import os

try:
    from openai_responses import request_json_schema_response
except ModuleNotFoundError:
    from face_server.openai_responses import request_json_schema_response


TRANSLATION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translation"],
    "properties": {
        "translation": {"type": "string"},
    },
}


def get_translation_text_model() -> str:
    return os.getenv("TRANSLATION_TEXT_MODEL", "gpt-4.1-mini")


def build_text_translation_prompt(
    transcript: str,
    source_label: str,
    target_label: str,
    context: list[str] | None = None,
) -> str:
    previous_context = "\n".join(
        f"{index + 1}. {item.strip()}"
        for index, item in enumerate(context or [])
        if item.strip()
    ).strip()
    context_block = (
        "Previous context for reference only:\n"
        f"{previous_context}\n\n"
        if previous_context
        else ""
    )
    return (
        "You are a translation engine.\n"
        f"Translate the input from {source_label} into concise natural {target_label}.\n"
        "Return exactly one valid JSON object that matches the requested schema.\n"
        "Translate the entire input. Do not stop halfway.\n"
        "Use previous context only to understand meaning, resolve references, and correct obvious ASR mistakes.\n"
        "Do not translate, summarize, or repeat previous context.\n"
        "Translate only the current input.\n"
        "Do not answer the speaker.\n"
        "Do not continue the conversation.\n"
        "Do not add explanations, notes, prefixes, or suffixes.\n"
        "If the input looks like ASR output, correct only obvious homophone, near-sounding, or contextually impossible recognition mistakes before translating.\n"
        "Do not add facts that are not supported by the input.\n"
        "If the input is incomplete, unclear, or fragmentary, translate it as literally as possible.\n"
        "If the input is only a name or noun phrase, output only the translated or transliterated phrase.\n"
        "Do not use markdown.\n\n"
        f"{context_block}"
        f"Current input:\n{transcript}"
    )


async def request_text_translation(
    transcript: str,
    source_label: str,
    target_label: str,
    context: list[str] | None = None,
) -> str:
    prompt = build_text_translation_prompt(transcript, source_label, target_label, context)
    result = await request_json_schema_response(
        prompt,
        "text_translation",
        TRANSLATION_RESPONSE_SCHEMA,
        model=get_translation_text_model(),
    )
    return str(result.get("translation") or "").strip()
