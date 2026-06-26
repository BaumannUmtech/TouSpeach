"""Pull TTS jobs from Django without exposing the GPU PC on the network."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from omnivoice.service.tts_service import (
    DialogueJob,
    DialogueSegment,
    TTSGenerationError,
    TTSJob,
    TTSService,
    TTSServiceConfig,
    TTSValidationError,
    VoiceSample,
)
from omnivoice.worker_client import DjangoJobClient


def _reference_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".wav", ".m4a", ".mp3", ".flac", ".ogg"} else ".wav"


def _materialize_reference_samples(client, service, job_id: str, samples):
    """Download Django-managed reference media into the worker's local work root."""
    materialized = []
    for index, sample in enumerate(samples, start=1):
        copied = dict(sample)
        if copied.get("ref_audio_url"):
            target = (
                service.config.media_root
                / "references"
                / str(job_id)
                / f"sample-{index}{_reference_suffix(copied['ref_audio_url'])}"
            )
            client.download_reference(copied["ref_audio_url"], target)
            copied["ref_audio_path"] = str(target)
        materialized.append(copied)
    return materialized


def process_one(client: DjangoJobClient, service: TTSService, args: argparse.Namespace) -> bool:
    job = client.claim(args.claim_path)
    if job is None:
        return False
    job_id = str(job["id"])
    generated_audio_path = None
    try:
        output_path = job.get("output_path", f"tts/{job_id}.m4a")
        if "segments" in job:
            total_chars = sum(len(segment.get("text", "")) for segment in job["segments"])
            logging.info(
                "Claimed dialogue job %s with %d segments and %d text chars",
                job_id,
                len(job["segments"]),
                total_chars,
            )
        else:
            logging.info(
                "Claimed TTS job %s with %d text chars",
                job_id,
                len(job.get("text", "")),
            )
        generation_started_at = time.perf_counter()
        if "segments" in job:
            samples = _materialize_reference_samples(
                client, service, job_id, job.get("voice_samples", [])
            )
            result = service.generate_dialogue(
                DialogueJob(
                    output_path=output_path,
                    pause_ms=int(job.get("pause_ms", 250)),
                    segments=tuple(
                        DialogueSegment(
                            speaker_id=segment["speaker_id"],
                            language=segment["language"],
                            text=segment["text"],
                        )
                        for segment in job["segments"]
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
        else:
            if job.get("ref_audio_url"):
                sample = _materialize_reference_samples(client, service, job_id, [job])[0]
                job = {**job, **sample}
            result = service.generate(
                TTSJob(
                    text=job["text"],
                    output_path=output_path,
                    language=job.get("language"),
                    mode=job.get("mode", "auto"),
                    instruct=job.get("instruct"),
                    ref_audio_path=job.get("ref_audio_path"),
                    ref_text=job.get("ref_text"),
                )
            )
        generation_seconds = time.perf_counter() - generation_started_at
        rtf = (
            generation_seconds / result.duration_seconds
            if result.duration_seconds > 0
            else 0.0
        )
        logging.info(
            "Job %s generated in %.2fs (audio duration %.2fs, RTF %.2f)",
            job_id,
            generation_seconds,
            result.duration_seconds,
            rtf,
        )
        generated_audio_path = Path(result.output_path)
        client.upload(
            args.upload_path,
            job_id,
            generated_audio_path,
            {
                "duration_seconds": result.duration_seconds,
                "generation_seconds": round(generation_seconds, 6),
                "sample_rate": result.sample_rate,
            },
        )
    except TTSValidationError as exc:
        client.fail(args.fail_path, job_id, "validation_error", str(exc))
    except TTSGenerationError as exc:
        client.fail(args.fail_path, job_id, "generation_failed", str(exc))
    except Exception as exc:
        logging.exception("Unhandled worker error for job %s", job_id)
        client.fail(args.fail_path, job_id, "worker_error", str(exc))
    finally:
        if generated_audio_path is not None:
            generated_audio_path.unlink(missing_ok=True)
        shutil.rmtree(service.config.media_root / "references" / job_id, ignore_errors=True)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OmniVoice Django pull worker.")
    parser.add_argument("--django-url", required=True)
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--claim-path", default="/api/omnivoice/jobs/claim/")
    parser.add_argument("--upload-path", default="/api/omnivoice/jobs/{id}/upload/")
    parser.add_argument("--fail-path", default="/api/omnivoice/jobs/{id}/fail/")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
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
    parser.add_argument(
        "--num-step",
        type=int,
        default=32,
        help="Generation steps. Lower values are faster but can reduce quality.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale used during generation.",
    )
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
        model_name=args.model, device=args.device, config=config, asr_model=args.asr_model
    )
    client = DjangoJobClient(args.django_url, args.api_token)
    try:
        while True:
            try:
                processed = process_one(client, service, args)
            except httpx.HTTPError as exc:
                logging.warning("Django is unavailable; retrying in %ss: %s", args.poll_seconds, exc)
                processed = False
            if not processed:
                time.sleep(args.poll_seconds)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
