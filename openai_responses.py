import asyncio
import json
import os
import socket
import time
from urllib import error, request


def get_openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


def get_openai_responses_url() -> str:
    return os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")


def get_openai_responses_timeout_seconds() -> float:
    try:
        return float(os.getenv("OPENAI_RESPONSES_TIMEOUT_SECONDS", "90"))
    except ValueError:
        return 90.0


def get_openai_responses_retries() -> int:
    try:
        return max(1, int(os.getenv("OPENAI_RESPONSES_RETRIES", "2")))
    except ValueError:
        return 2


def _is_retryable_url_error(exc: error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError | socket.timeout):
        return True
    return "timed out" in str(reason).lower()


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

    raise ValueError("OpenAI response did not contain output text")


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

    payload = _openai_request_json(req, "OpenAI text response failed")

    return _extract_response_text(payload)


def _request_json_schema_response_sync(
    model: str,
    prompt: str,
    schema_name: str,
    schema: dict,
    reasoning_effort: str = "",
    temperature: str = "",
) -> str:
    payload = {
        "model": model,
        "input": prompt,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }
    if reasoning_effort:
        payload["reasoning"] = {
            "effort": reasoning_effort,
        }
    if temperature:
        payload["temperature"] = float(temperature)

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        get_openai_responses_url(),
        data=body,
        headers={
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    payload = _openai_request_json(req, "OpenAI JSON schema response failed")

    return _extract_response_text(payload)


def _openai_request_json(req: request.Request, error_prefix: str) -> dict:
    attempts = get_openai_responses_retries()
    timeout = get_openai_responses_timeout_seconds()
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= attempts:
                raise ValueError(f"{error_prefix}: HTTP {exc.code}: {detail}") from exc
            last_exc = ValueError(f"{error_prefix}: HTTP {exc.code}: {detail}")
        except TimeoutError as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
        except socket.timeout as exc:
            last_exc = exc
            if attempt >= attempts:
                raise TimeoutError("The read operation timed out") from exc
        except error.URLError as exc:
            last_exc = exc
            if not _is_retryable_url_error(exc) or attempt >= attempts:
                raise

        time.sleep(min(2 ** (attempt - 1), 4))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{error_prefix}: request failed")


async def request_text_response(model: str, prompt: str) -> str:
    return await asyncio.to_thread(_request_text_response_sync, model, prompt)


async def request_json_schema_response(
    prompt: str,
    schema_name: str,
    schema: dict,
    model: str,
    reasoning_effort: str = "",
    temperature: str = "",
    raw_text_hook=None,
) -> dict:
    final_text = await asyncio.to_thread(
        _request_json_schema_response_sync,
        model,
        prompt,
        schema_name,
        schema,
        reasoning_effort,
        temperature,
    )
    if raw_text_hook is not None:
        await raw_text_hook(final_text)
    return json.loads(final_text)
