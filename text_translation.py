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


def get_summary_text_model() -> str:
    return os.getenv("SUBTITLE_SUMMARY_MODEL", "gpt-4.1-mini")


def get_correction_text_model() -> str:
    return os.getenv("SUBTITLE_CORRECTION_MODEL", get_summary_text_model())


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


def build_subtitle_summary_prompt(new_transcripts: str) -> str:
    return (
        "Summarize only the current transcript chunk as short notes.\n"
        "Use only information reasonably supported by this chunk.\n"
        "Keep it concise, factual, and easy to append in a UI.\n"
        "Do not add information from earlier chunks.\n"
        "If the transcript likely contains homophone or near-sounding ASR mistakes, "
        "infer the most likely intended meaning from context before summarizing.\n"
        "Prefer contextually appropriate meanings over obviously incorrect literal words.\n"
        "Do not preserve incorrect wording verbatim when the intended meaning is clear.\n"
        "If the intended meaning is uncertain, stay conservative.\n"
        "Write the summary in the same language as the transcript chunk.\n"
        "Output only the summary text.\n"
        "Do not use markdown.\n\n"
        f"Transcript chunk:\n{new_transcripts}"
    )


def build_subtitle_chunk_correction_prompt(chunk_text: str, source_label: str) -> str:
    return (
        "You are correcting ASR transcript notes for later summarization.\n"
        f"The expected language is {source_label}.\n"
        "Correct obvious homophone, near-sounding, or contextually implausible ASR mistakes.\n"
        "Prefer the most likely intended wording when the surrounding context is clear.\n"
        "If the intended wording is uncertain, stay conservative and keep the original wording.\n"
        "Keep the same language and writing system as the transcript chunk.\n"
        "Do not summarize, translate, explain, or add information.\n"
        "Preserve the original structure as plain transcript notes.\n"
        "Output only the corrected chunk text.\n"
        "Do not use markdown.\n\n"
        f"Transcript chunk:\n{chunk_text}"
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


def _request_text_response_sync(model: str, prompt: str) -> str:
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
    return await asyncio.to_thread(
        _request_text_response_sync,
        get_translation_text_model(),
        prompt,
    )


async def request_text_summary(new_transcripts: str) -> str:
    prompt = build_subtitle_summary_prompt(new_transcripts)
    return await asyncio.to_thread(
        _request_text_response_sync,
        get_summary_text_model(),
        prompt,
    )


async def request_chunk_correction(
    chunk_text: str,
    source_label: str,
) -> str:
    prompt = build_subtitle_chunk_correction_prompt(chunk_text, source_label)
    return await asyncio.to_thread(
        _request_text_response_sync,
        get_correction_text_model(),
        prompt,
    )
