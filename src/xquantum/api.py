"""Local research API. Image bytes and clinical inputs are never written to disk."""

import base64
import binascii
from pathlib import Path

from fastapi import FastAPI, HTTPException

from xquantum.model import ResearchModel
from xquantum.schemas import AnalyzeRequest

MAX_REQUEST_BYTES = 14_100_000


class BodyLimit:
    """Bound streamed input before JSON parsing; do not trust Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_REQUEST_BYTES:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"cache-control", b"no-store"),
                        ],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": b'{"detail":"Request too large"}'}
                )
                return
            chunks.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            if chunks:
                return chunks.pop(0)
            return await receive()

        await self.app(scope, replay, send)


def create_app(model_dir: Path) -> FastAPI:
    model = ResearchModel.load(model_dir)
    app = FastAPI(
        title="XQuantum Research API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.add_middleware(BodyLimit)

    @app.middleware("http")
    async def prevent_caching(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return {"model_loaded": True, "clinically_validated": False}

    @app.post("/v1/analyze")
    def analyze(payload: AnalyzeRequest):
        try:
            raw = base64.b64decode(payload.image_base64, validate=True)
            return model.analyze(raw, payload.clinical)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
