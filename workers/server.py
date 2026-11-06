"""Imotan Vision 1 — OpenAI-compatible HTTP shim.

RunPod's native serverless contract is `POST /runsync` with an `{"input": ...}` envelope.
Neither Imotan's API layer nor OpenRouter speaks that: both expect
`POST /v1/images/generations` with an OpenAI body. This module translates, so the worker
has exactly one public contract regardless of which RunPod endpoint type it sits behind.

Entry points:
  * load-balancing endpoint (recommended): `dockerStartCmd` runs `uvicorn server:app`
  * queue endpoint: `handler.py` still serves `/runsync` for direct/manual testing

Both call `handler.generate_payload`, so behaviour cannot drift between them.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from handler import MODEL_ID, generate_payload

WORKER_TOKEN = os.environ.get("IMOTAN_WORKER_TOKEN", "").strip()

app = FastAPI(title="Imotan Vision 1", version="1.0.0", docs_url=None, redoc_url=None)


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": code}},
    )


def _authorised(request: Request) -> bool:
    if not WORKER_TOKEN:
        return True
    header = request.headers.get("authorization", "")
    return header == f"Bearer {WORKER_TOKEN}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "imotan",
                "modality": "image",
            }
        ],
    }


@app.post("/v1/images/generations")
async def images_generations(request: Request) -> Any:
    if not _authorised(request):
        return _error(401, "Missing or invalid worker token.", "invalid_api_key", "authentication_error")

    try:
        body = await request.json()
    except Exception:
        return _error(400, "Request body must be valid JSON.", "invalid_json")

    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.", "invalid_body")

    if not (body.get("prompt") or "").strip():
        return _error(400, "prompt is required", "invalid_request")

    # `response_format` is an OpenAI parameter that does not apply here; the response
    # always carries b64_json. Accepting and ignoring it keeps client compatibility.
    result = generate_payload(body)

    error = result.get("error") if isinstance(result, dict) else None
    if error:
        code = error.get("code", "invalid_request")
        status = 400 if code in ("invalid_request", "invalid_json", "unsupported_format") else 403
        return _error(status, error.get("message", "request failed"), code)

    return result


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
    # Never leak upstream/library detail to the caller; see ARCHITECTURE.md §5.
    print(f"[imotan] unhandled error: {type(exc).__name__}: {exc}", flush=True)
    return _error(500, "The image could not be generated. Try again in a moment.", "internal_error", "server_error")
