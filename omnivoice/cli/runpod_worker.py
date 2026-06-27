"""RunPod Serverless handler for Django-dispatched OmniVoice jobs."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from omnivoice.service.tts_service import (
    DialogueJob,
    DialogueSegment,
    TTSGenerationError,
    TTSService,
    TTSServiceConfig,
    TTSValidationError,
    VoiceSample,
)


def _reference_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".wav", ".m4a", ".mp3", ".flac", ".ogg"} else ".wav"


def _download_reference(
    client: httpx.Client, url: str, target: Path, headers: dict[str, str] | None = None
) -> None:
    response = client.get(url, headers=headers)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)


def _materialize_reference_samples(
    client: httpx.Client,
    service: TTSService,
    job_id: str,
    samples: list[dict[str, Any]],
    authorization: str | None,
) -> list[dict[str, Any]]:
    materialized = []
    headers = {"Authorization": authorization} if authorization else None
    for index, sample in enumerate(samples, start=1):
        copied = dict(sample)
        if copied.get("ref_audio_url"):
            target = (
                service.config.media_root
                / "references"
                / str(job_id)
                / f"sample-{index}{_reference_suffix(copied['ref_audio_url'])}"
            )
            _download_reference(client, copied["ref_audio_url"], target, headers=headers)
            copied["ref_audio_path"] = str(target)
        materialized.append(copied)
    return materialized


def _post_upload(
    client: httpx.Client,
    result_config: dict[str, Any],
    audio_path: Path,
    result: dict[str, Any],
) -> None:
    authorization = result_config.get("authorization")
    headers = {"Authorization": authorization} if authorization else None
    audio_field = result_config.get("audio_field", "audio")
    audio_filename = result_config.get("audio_filename", audio_path.name)
    data = {
        "sample_rate": str(result["sample_rate"]),
        "duration_seconds": str(result["duration_seconds"]),
    }
    if "generation_seconds" in result:
        data["generation_seconds"] = str(result["generation_seconds"])
    with audio_path.open("rb") as audio_file:
        response = client.post(
            result_config["upload_url"],
            headers=headers,
            data=data,
            files={audio_field: (audio_filename, audio_file, "audio/mp4")},
        )
    response.raise_for_status()


def _post_fail(client: httpx.Client, result_config: dict[str, Any], message: str) -> None:
    fail_url = result_config.get("fail_url")
    if not fail_url:
        logging.error("RunPod job failed but no result.fail_url was provided: %s", message)
        return
    authorization = result_config.get("authorization")
    headers = {"Authorization": authorization} if authorization else None
    response = client.post(fail_url, headers=headers, json={"error_message": message})
    response.raise_for_status()


def process_runpod_job(
    event: dict[str, Any],
    service: TTSService,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Process one RunPod Serverless event using the Django callback contract."""
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=60.0)

    input_payload = event.get("input") or {}
    result_config = input_payload.get("result") or {}
    job_id = str(input_payload.get("job_id") or input_payload.get("id") or "")
    if not job_id:
        raise TTSValidationError("input.job_id is required.")
    if not result_config.get("upload_url"):
        raise TTSValidationError("input.result.upload_url is required.")

    generated_audio_path: Path | None = None
    try:
        output_path = input_payload.get("output_path", f"tts/{job_id}.m4a")
        generation_started_at = time.perf_counter()
        authorization = result_config.get("authorization")
        samples = _materialize_reference_samples(
            client,
            service,
            job_id,
            list(input_payload.get("voice_samples", [])),
            authorization,
        )
        result = service.generate_dialogue(
            DialogueJob(
                output_path=output_path,
                pause_ms=int(input_payload.get("pause_ms", 250)),
                segments=tuple(
                    DialogueSegment(
                        speaker_id=segment["speaker_id"],
                        language=segment["language"],
                        text=segment["text"],
                    )
                    for segment in input_payload.get("segments", [])
                ),
                voice_samples=tuple(
                    VoiceSample(
                        speaker_id=sample["speaker_id"],
                        language=sample["language"],
                        ref_audio_path=sample["ref_audio_path"],
                        ref_text=sample.get("ref_text"),
                    )
                    for sample in samples
                ),
            )
        )
        generation_seconds = time.perf_counter() - generation_started_at
        generated_audio_path = Path(result.output_path)
        upload_result = {
            "duration_seconds": result.duration_seconds,
            "generation_seconds": round(generation_seconds, 6),
            "sample_rate": result.sample_rate,
        }
        _post_upload(client, result_config, generated_audio_path, upload_result)
        return {
            "success": True,
            "job_id": job_id,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "generation_seconds": upload_result["generation_seconds"],
        }
    except (TTSValidationError, TTSGenerationError) as exc:
        _post_fail(client, result_config, str(exc))
        return {"success": False, "job_id": job_id, "error_message": str(exc)}
    except Exception as exc:
        logging.exception("Unhandled RunPod worker error for job %s", job_id)
        _post_fail(client, result_config, str(exc))
        return {"success": False, "job_id": job_id, "error_message": str(exc)}
    finally:
        if generated_audio_path is not None:
            generated_audio_path.unlink(missing_ok=True)
        shutil.rmtree(service.config.media_root / "references" / job_id, ignore_errors=True)
        if owns_client:
            client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OmniVoice as a RunPod Serverless worker.")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--work-root",
        "--media-root",
        dest="work_root",
        required=True,
        help="Local worker directory for temporary references and generated M4A files.",
    )
    parser.add_argument("--temp-dir", default=".tmp/omnivoice")
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--load-asr", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = TTSServiceConfig(
        media_root=Path(args.work_root),
        temp_dir=Path(args.temp_dir),
        ffmpeg=args.ffmpeg,
        reference_roots=tuple(Path(root) for root in args.reference_root),
        load_asr=args.load_asr,
        generation_num_step=args.num_step,
        generation_guidance_scale=args.guidance_scale,
    )
    service = TTSService.load(
        model_name=args.model,
        device=args.device,
        config=config,
        asr_model=args.asr_model,
    )

    try:
        import runpod
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise SystemExit("Install the 'runpod' package to use omnivoice-runpod-worker.") from exc

    runpod.serverless.start(
        {"handler": lambda event: process_runpod_job(event, service)}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
