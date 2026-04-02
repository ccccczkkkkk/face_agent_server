import os
import json
import base64
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

try:
    from ws_utils import safe_receive_message, safe_send_envelope
except ModuleNotFoundError:
    from face_server.ws_utils import safe_receive_message, safe_send_envelope

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")

# Realtime WebSocket endpoint（以官方文档为准）
REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

app = FastAPI()
# print("OPENAI KEY OK:", OPENAI_API_KEY[:6])

SYSTEM_PROMPT = """你是一个面对面日语沟通助手。
你会收到双方的日语语音（转写文本），请输出严格 JSON：
{
  "transcript": "...",
  "zh_translation": "...",
  "next_say": [
    {"ja":"...", "romaji":"...", "zh":"..."},
    {"ja":"...", "romaji":"...", "zh":"..."}
  ],
  "intent": "..."
}
规则：
- next_say 只给 2 条，尽量使用礼貌日语（敬语）。
- 不要编造未确认信息；缺信息就站在用户视角提出问题句作为 next_say。
- 输出必须是单行 JSON，不要 markdown。
"""

async def openai_realtime_connect():
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",  # 按官方 realtime-websocket 指南使用
    }
    return await websockets.connect(REALTIME_WS_URL, additional_headers=headers, max_size=2**24)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        ows = await openai_realtime_connect()
    except Exception as e:
        print("OpenAI realtime connect failed:", repr(e))
        await ws.close()
        return

    # 读取客户端的首包配置（大纲等）
    # 客户端发送：{"type":"config","outline":"...","lang":"ja-JP"}
    client_outline = ""
    try:
        first = await ws.receive_text()
        cfg = json.loads(first)
        if cfg.get("type") == "config":
            client_outline = cfg.get("outline", "")
            print("OUTLINE FROM CLIENT:", client_outline)
    except Exception:
        pass

    # 1) 更新 Realtime session：开 server_vad、指定音频格式、提示词等
    # 官方“prompting / session update”说明见文档。:contentReference[oaicite:3]{index=3}
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text"],  # 先只要文本输出（字幕+建议）；后续可加 audio 输出
            "instructions": SYSTEM_PROMPT + "\n\n用户大纲:\n" + client_outline,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.2,  # VAD 灵敏度，数值越小越容易切分（但可能过度切分）
                "silence_duration_ms": 100,
                "prefix_padding_ms": 300
            },
            "input_audio_format": "pcm16",
            # 
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",  # 官方转写模型，支持日语等多语言
            }
        }
    }
    await ows.send(json.dumps(session_update)) # 先更新 session，后续每段话结束时再 request response.create 来生成建议
    
    async def request_opening_suggestion(ows, outline: str):
        await ows.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": (
                    "你是日语沟通助手。请根据用户给出的场景大纲，给出我作为来电/来访者的开场白建议。\n"
                    "输出严格单行 JSON（不要 markdown，不要多余文字）。\n"
                    '格式：{"type":"suggestion","stage":"opening","next_say":[{"ja":"...","romaji":"...","zh":"..."}],"intent":"opening"}\n'
                    f"大纲：{outline}\n"
                    "next_say 1~3 条，每条尽量简短、礼貌、可直接照读。"
                )
            }
        }, ensure_ascii=False))
    
    await request_opening_suggestion(ows, client_outline) # 先给个开场白建议，后续根据转写再生成更具体的建议

    

    async def forward_audio_from_client():
        """客户端 -> Realtime：转发二进制 PCM16"""
        while True:
            msg = await safe_receive_message(ws)
            if msg is None:
                return
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                # print("audio bytes:", len(b)) # 客户端发来的 PCM16 音频数据，长度取决于发送频率和 chunk 大小
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b).decode("utf-8"),
                }
                await ows.send(json.dumps(payload))
            elif "text" in msg and msg["text"] is not None:
                # 可选：客户端发控制命令（例如暂停/清空）
                t = msg["text"]
                try:
                    j = json.loads(t)
                    if j.get("type") == "reset":
                        await ows.send(json.dumps({"type": "input_audio_buffer.clear"}))
                except Exception:
                    pass

    async def relay_events_to_client():
        """Realtime -> 客户端：把关键事件转成 JSON 推回 Flutter"""
        current_transcript = ""
        text_buf = []
        async for raw in ows:
            try:
                event = json.loads(raw)
            except Exception:
                continue

            et = event.get("type", "")

            # 处理转写增量事件，实时把转写结果发给 Flutter
            if et == "conversation.item.input_audio_transcription.delta":
                delta = event.get("delta", "")
                if delta:
                    current_transcript += delta
                    sent = await safe_send_envelope(ws, {
                        "type": "transcript_delta",
                        "delta": delta
                    })
                    if not sent:
                        return
                
            # 把所有事件都发回去方便调试
            # await ws.send_text(json.dumps({"type":"debug","event":event}))

            # 当 server_vad 认为一段话结束时，你会看到 commit/segment 之类事件
            # 然后你需要 request 一个 response.create 让模型输出我们要的 JSON
            # 这里用一个简单策略：看到转写 completed 就触发生成
            if et == "conversation.item.input_audio_transcription.completed":
                transcript = (event.get("transcript") or current_transcript).strip()

                # 把最终转写发给 Flutter
                sent = await safe_send_envelope(ws, {
                    "type": "transcript_final",
                    "transcript": transcript,
                    "zh_translation": "",
                    "next_say": [],
                    "intent": ""
                })
                if not sent:
                    return

                current_transcript = ""

                if transcript:
                    await ows.send(json.dumps({
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": (
                                SYSTEM_PROMPT
                                + "\n基于以下日语转写，输出严格单行 JSON（不要 markdown，不要多余字符）。\n"
                                + f"日语转写：{transcript}\n"
                            )
                        }
                    }, ensure_ascii=False))

            # 文本增量（把碎片拼起来）
            if et in ("response.output_text.delta", "response.text.delta"):
                delta = event.get("delta", "")
                if delta:
                    text_buf.append(delta)

            # 文本完成（一次性发给 Flutter）
            if et in ("response.output_text.done", "response.text.done"):
                final_text = (event.get("text") or "".join(text_buf)).strip()
                text_buf.clear()
                if not final_text:
                    return

                # 强制统一：只发 envelope JSON
                try:
                    obj = json.loads(final_text)
                    if isinstance(obj, dict):
                        # 给没有 type 的补上
                        if "type" not in obj:
                            obj["type"] = "suggestion"
                        sent = await safe_send_envelope(ws, obj)
                        if not sent:
                            return
                    else:
                        sent = await safe_send_envelope(ws, {
                            "type": "error",
                            "message": "model output is not a JSON object",
                            "raw": final_text
                        })
                        if not sent:
                            return
                except Exception as e:
                    sent = await safe_send_envelope(ws, {
                        "type": "error",
                        "message": f"bad JSON from model: {repr(e)}",
                        "raw": final_text
                    })
                    if not sent:
                        return


            # print("EVENT:", et)
            # 只打印转写相关
            if "input_audio_transcription" in et:
                print("TRANSCRIPTION EVENT:", event)

            # # 拿到最终生成文本
            # if et in ("response.output_text.done", "response.text.done"):
            #     out = event.get("text", "")
            #     if out:
            #         await ws.send_text(out)

    try:
        await asyncio.gather(forward_audio_from_client(), relay_events_to_client())
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ows.close()
        except Exception:
            pass
