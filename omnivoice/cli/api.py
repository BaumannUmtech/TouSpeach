"""Loopback-only REST API for local OmniVoice TTS."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from omnivoice.service.tts_service import (
    TTSGenerationError,
    TTSJob,
    TTSService,
    TTSServiceConfig,
    TTSValidationError,
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Optional[str] = None
    mode: str = "auto"
    instruct: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    output_path: Optional[str] = None


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def create_app(
    settings: argparse.Namespace,
    service_factory: Callable[..., TTSService] = TTSService.load,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = TTSServiceConfig(
            media_root=Path(settings.media_root),
            temp_dir=Path(settings.temp_dir),
            ffmpeg=settings.ffmpeg,
            reference_roots=tuple(Path(root) for root in settings.reference_root),
            load_asr=settings.load_asr,
        )
        app.state.tts_service = service_factory(
            model_name=settings.model,
            device=settings.device,
            config=config,
            asr_model=settings.asr_model,
        )
        yield

    app = FastAPI(title="OmniVoice Local TTS", lifespan=lifespan)

    @app.exception_handler(TTSValidationError)
    async def validation_error(_: Request, exc: TTSValidationError):
        return _error(422, "validation_error", str(exc))

    @app.exception_handler(TTSGenerationError)
    async def generation_error(_: Request, exc: TTSGenerationError):
        logging.exception("TTS generation failed", exc_info=exc)
        return _error(500, "generation_failed", str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        return _error(422, "request_validation_error", str(exc.errors()))

    @app.get("/health")
    def health(request: Request):
        service = request.app.state.tts_service
        return {
            "success": True,
            "status": "ok",
            "device": service.device,
            "model_loaded": True,
            "ffmpeg_available": True,
        }

    @app.post("/tts")
    def tts(payload: TTSRequest, request: Request):
        result = request.app.state.tts_service.generate(TTSJob(**payload.model_dump()))
        return {"success": True, **result.__dict__}

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local OmniVoice REST API.")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8002, type=int)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--temp-dir", default=".tmp/omnivoice")
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--load-asr", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.host != "127.0.0.1":
        raise SystemExit("For safety, omnivoice-api may bind only to 127.0.0.1.")
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
