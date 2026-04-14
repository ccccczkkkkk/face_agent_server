import asyncio
import json
import os
from urllib import request


def get_openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


def get_openai_responses_url() -> str:
    return os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")


def get_translation_text_model() -> str:
    return os.getenv("TRANSLATION_TEXT_MODEL", "gpt-4.1-mini")


def build_text_translation_prompt(
    transcript: str,
    source_label: str,
    target_label: str,
) -> str:
    return (
        "You are a translation engine.\n"
        f"Translate the input from {source_label} into concise natural {target_label}.\n"
        "Output only the translation text.\n"
        "Do not answer the speaker.\n"
        "Do not continue the conversation.\n"
        "Do not add explanations, notes, prefixes, or suffixes.\n"
        "If the input is incomplete, unclear, or fragmentary, translate it as literally as possible.\n"
        "If the input is only a name or noun phrase, output only the translated or transliterated phrase.\n"
        "Do not use markdown.\n\n"
        f"Input:\n{transcript}"
    )


def _extract_response_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = payload.get("output") or []
    if isinstance(outputs, list):
        chunks: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "output_text":
                    text = str(entry.get("text") or "").strip()
                    if text:
                        chunks.append(text)
        if chunks:
            return "".join(chunks).strip()

    raise ValueError("Text translation response did not contain output text")


def _request_text_translation_sync(prompt: str) -> str:
    model = get_translation_text_model()
    body = json.dumps({
        "model": model,
        "input": prompt,
    }).encode("utf-8")
    req = request.Request(
        get_openai_responses_url(),
        data=body,
        headers={
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    return _extract_response_text(payload)


async def request_text_translation(
    transcript: str,
    source_label: str,
    target_label: str,
) -> str:
    prompt = build_text_translation_prompt(transcript, source_label, target_label)
    return await asyncio.to_thread(_request_text_translation_sync, prompt)
