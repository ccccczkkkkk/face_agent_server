import os


SUGGESTION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "stage", "next_say", "intent"],
    "properties": {
        "type": {"type": "string", "enum": ["suggestion"]},
        "stage": {"type": "string", "enum": ["opening", "followup"]},
        "next_say": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ja", "romaji", "zh"],
                "properties": {
                    "ja": {"type": "string"},
                    "romaji": {"type": "string"},
                    "zh": {"type": "string"},
                },
            },
        },
        "intent": {"type": "string"},
    },
}

MEMORY_PATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "known_info_add",
        "open_loops_add",
        "open_loops_remove",
        "next_actions_replace",
    ],
    "properties": {
        "summary": {"type": "string"},
        "known_info_add": {
            "type": "array",
            "items": {"type": "string"},
        },
        "open_loops_add": {
            "type": "array",
            "items": {"type": "string"},
        },
        "open_loops_remove": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_actions_replace": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def get_conversation_reasoning_model() -> str:
    return os.getenv("CONVERSATION_REASONING_MODEL", os.getenv("TRANSLATION_TEXT_MODEL", "gpt-4.1-mini"))


def get_conversation_reasoning_effort() -> str:
    return os.getenv("CONVERSATION_REASONING_EFFORT", "").strip()


def get_conversation_reasoning_temperature() -> str:
    return os.getenv("CONVERSATION_REASONING_TEMPERATURE", "").strip()
