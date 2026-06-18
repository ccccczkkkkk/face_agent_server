import os

try:
    from openai_responses import request_json_schema_response, request_text_response
except ModuleNotFoundError:
    from face_server.openai_responses import request_json_schema_response, request_text_response


CHUNK_ITEMS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "text", "priority", "urgency"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["task", "important", "deadline"],
                    },
                    "text": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["now", "later", "unknown"],
                    },
                },
            },
        },
    },
}


def get_summary_text_model() -> str:
    return os.getenv("SUBTITLE_SUMMARY_MODEL", "gpt-4.1-mini")


def get_correction_text_model() -> str:
    return os.getenv("SUBTITLE_CORRECTION_MODEL", get_summary_text_model())


def build_subtitle_summary_prompt(
    new_transcripts: str,
    output_language_label: str = "the same language as the transcript chunk",
) -> str:
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
        f"Write the summary in {output_language_label} only.\n"
        "Even if the transcript contains words or fragments from other languages, do not switch the output language.\n"
        "Keep technical terms in their original form only when that is natural in the output language.\n"
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


def build_chunk_items_prompt(
    chunk_text: str,
    output_language_label: str = "the same language as the transcript chunk",
    existing_items: list[dict] | None = None,
) -> str:
    existing_lines = []
    for item in existing_items or []:
        text = str(item.get("source") or item.get("text") or "").strip()
        kind = str(item.get("kind") or "important").strip()
        if text:
            existing_lines.append(f"- [{kind}] {text}")
    existing_block = "\n".join(existing_lines).strip() or "(none)"
    return (
        "Extract only new important items from the current transcript chunk.\n"
        "Return JSON only using the requested schema.\n"
        "Items should be concrete things the user may need to notice, remember, or act on.\n"
        "Include immediate tasks, instructions, deadlines, restrictions, warnings, required actions, and important conditions.\n"
        "Do not include generic summary points.\n"
        "Do not repeat or restate any existing item unless the current chunk adds materially new information.\n"
        "Use only information supported by the current chunk.\n"
        f"Write item text in {output_language_label} only.\n"
        "Even if the transcript contains words or fragments from other languages, do not switch the item text language.\n"
        "Keep technical terms in their original form only when that is natural in the output language.\n"
        "If there are no new actionable or important items, return {\"items\":[]}.\n\n"
        f"Existing items:\n{existing_block}\n\n"
        f"Current transcript chunk:\n{chunk_text}"
    )


async def request_text_summary(
    new_transcripts: str,
    output_language_label: str = "the same language as the transcript chunk",
) -> str:
    prompt = build_subtitle_summary_prompt(new_transcripts, output_language_label)
    return await request_text_response(get_summary_text_model(), prompt)


async def request_chunk_correction(
    chunk_text: str,
    source_label: str,
) -> str:
    prompt = build_subtitle_chunk_correction_prompt(chunk_text, source_label)
    return await request_text_response(get_correction_text_model(), prompt)


async def request_chunk_items(
    chunk_text: str,
    output_language_label: str = "the same language as the transcript chunk",
    existing_items: list[dict] | None = None,
) -> list[dict]:
    prompt = build_chunk_items_prompt(chunk_text, output_language_label, existing_items)
    result = await request_json_schema_response(
        prompt,
        "subtitle_chunk_items",
        CHUNK_ITEMS_RESPONSE_SCHEMA,
        model=get_summary_text_model(),
    )
    items = result.get("items") or []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]
