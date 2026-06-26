"""Safe, serialized OmniVoice generation and M4A output handling."""

from __future__ import annotations

import subprocess
import tempfile
import threading
import uuid
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import numpy as np

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

logger = logging.getLogger(__name__)


class TTSValidationError(ValueError):
    """A request cannot be safely or semantically processed."""


class TTSGenerationError(RuntimeError):
    """Generation or encoding failed after request validation."""


@dataclass(frozen=True)
class TTSServiceConfig:
    media_root: Path
    temp_dir: Path
    ffmpeg: str = "ffmpeg"
    reference_roots: tuple[Path, ...] = ()
    load_asr: bool = False
    generation_num_step: int = 32
    generation_guidance_scale: float = 2.0


@dataclass(frozen=True)
class TTSJob:
    text: str
    output_path: Optional[str] = None
    language: Optional[str] = None
    mode: str = "auto"
    instruct: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None


@dataclass(frozen=True)
class TTSResult:
    output_path: str
    relative_output_path: str
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class VoiceSample:
    """One speaker-language reference recording and its exact transcript."""

    speaker_id: str
    language: str
    ref_audio_path: str
    ref_text: Optional[str] = None


@dataclass(frozen=True)
class DialogueSegment:
    speaker_id: str
    language: str
    text: str


@dataclass(frozen=True)
class DialogueJob:
    output_path: str
    segments: tuple[DialogueSegment, ...]
    voice_samples: tuple[VoiceSample, ...]
    pause_ms: int = 250


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class TTSService:
    """Own one loaded model and allow exactly one active generation."""

    def __init__(self, model: OmniVoice, device: str, config: TTSServiceConfig):
        self.model = model
        self.device = str(device)
        self.config = config
        self._generation_lock = threading.Lock()
        self.prepare_storage(config)
        self._media_root = self.config.media_root.resolve()
        roots = self.config.reference_roots or (self._media_root,)
        self._reference_roots = tuple(root.resolve() for root in roots)

    @classmethod
    def load(
        cls,
        *,
        model_name: str,
        device: Optional[str],
        config: TTSServiceConfig,
        asr_model: str = "openai/whisper-large-v3-turbo",
    ) -> "TTSService":
        selected_device = device or get_best_device()
        dtype = torch.float32 if selected_device == "cpu" else torch.float16
        cls.prepare_storage(config)
        cls.check_ffmpeg(config.ffmpeg)
        model = OmniVoice.from_pretrained(
            model_name,
            device_map=selected_device,
            dtype=dtype,
            load_asr=config.load_asr,
            asr_model_name=asr_model,
        )
        return cls(model=model, device=selected_device, config=config)

    @staticmethod
    def prepare_storage(config: TTSServiceConfig) -> None:
        """Fail before expensive model loading when the media share is unavailable."""
        try:
            config.media_root.mkdir(parents=True, exist_ok=True)
            config.temp_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Media root is unavailable or not writable: {config.media_root}"
            ) from exc

    @staticmethod
    def check_ffmpeg(ffmpeg: str) -> None:
        try:
            subprocess.run(
                [ffmpeg, "-version"], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"FFmpeg is unavailable ({ffmpeg!r}). Install FFmpeg or pass --ffmpeg."
            ) from exc

    def resolve_output_path(self, requested_path: Optional[str]) -> tuple[Path, str]:
        if requested_path is None:
            requested_path = (
                f"tts/{date.today():%Y/%m}/{uuid.uuid4().hex}.m4a"
            )
        path = Path(requested_path)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise TTSValidationError("output_path must be a relative path inside media_root.")
        if path.suffix.lower() != ".m4a":
            raise TTSValidationError("output_path must use the .m4a extension.")
        target = (self._media_root / path).resolve()
        if not _is_within(target, self._media_root):
            raise TTSValidationError("output_path resolves outside media_root.")
        return target, target.relative_to(self._media_root).as_posix()

    def resolve_reference_path(self, requested_path: str) -> Path:
        path = Path(requested_path)
        candidate = path.resolve() if path.is_absolute() else (self._media_root / path).resolve()
        if not candidate.is_file() or not any(
            _is_within(candidate, root) for root in self._reference_roots
        ):
            raise TTSValidationError("ref_audio_path is not an allowed local audio file.")
        return candidate

    @staticmethod
    def _validate_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        value = language.strip()
        if not value:
            return None
        if value.lower() not in LANG_NAME_TO_ID and value not in LANG_IDS:
            raise TTSValidationError(f"Unsupported language: {language!r}.")
        return value

    def _validate_job(self, job: TTSJob) -> tuple[Optional[str], Optional[Path]]:
        if not job.text or not job.text.strip():
            raise TTSValidationError("text is required and must not be empty.")
        if job.mode not in {"auto", "design", "clone"}:
            raise TTSValidationError("mode must be one of: auto, design, clone.")
        if job.mode == "design" and not (job.instruct and job.instruct.strip()):
            raise TTSValidationError("instruct is required when mode is design.")
        ref_audio = None
        if job.mode == "clone":
            if not job.ref_audio_path:
                raise TTSValidationError("ref_audio_path is required when mode is clone.")
            if not job.ref_text and not self.config.load_asr:
                raise TTSValidationError(
                    "ref_text is required for clone mode unless the service starts with --load-asr."
                )
            ref_audio = self.resolve_reference_path(job.ref_audio_path)
        return self._validate_language(job.language), ref_audio

    def generate(self, job: TTSJob) -> TTSResult:
        language, ref_audio = self._validate_job(job)
        target, relative_target = self.resolve_output_path(job.output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._generation_lock:
            try:
                gen_config = OmniVoiceGenerationConfig(
                    num_step=self.config.generation_num_step,
                    guidance_scale=self.config.generation_guidance_scale,
                )
                kwargs = {
                    "text": job.text.strip(),
                    "language": language,
                    "generation_config": gen_config,
                }
                if job.mode == "design":
                    kwargs["instruct"] = job.instruct.strip()  # type: ignore[union-attr]
                elif job.mode == "clone":
                    kwargs["ref_audio"] = str(ref_audio)
                    kwargs["ref_text"] = job.ref_text.strip() if job.ref_text else None
                audio = self.model.generate(**kwargs)[0]
                sample_rate = int(self.model.sampling_rate)
                duration = float(len(audio) / sample_rate)
                self._write_m4a(audio, sample_rate, target)
            except TTSValidationError:
                raise
            except Exception as exc:
                raise TTSGenerationError(f"TTS generation failed: {exc}") from exc

        return TTSResult(
            output_path=str(target),
            relative_output_path=relative_target,
            sample_rate=sample_rate,
            duration_seconds=round(duration, 6),
        )

    def generate_dialogue(self, job: DialogueJob) -> TTSResult:
        """Generate a multi-speaker, multi-language dialogue into one M4A file."""
        if not job.segments:
            raise TTSValidationError("A dialogue job requires at least one segment.")
        if job.pause_ms < 0 or job.pause_ms > 10_000:
            raise TTSValidationError("pause_ms must be between 0 and 10000.")

        samples = {}
        for sample in job.voice_samples:
            if not sample.speaker_id.strip():
                raise TTSValidationError("Each voice sample requires speaker_id.")
            language = self._validate_language(sample.language)
            if language is None:
                raise TTSValidationError("Each voice sample requires language.")
            if not sample.ref_text and not self.config.load_asr:
                raise TTSValidationError(
                    "Each voice sample requires ref_text unless the service starts with --load-asr."
                )
            key = (sample.speaker_id, language.lower())
            if key in samples:
                raise TTSValidationError(
                    f"Duplicate voice sample for speaker={sample.speaker_id!r}, language={language!r}."
                )
            samples[key] = (sample, language, self.resolve_reference_path(sample.ref_audio_path))

        validated_segments = []
        for segment in job.segments:
            language = self._validate_language(segment.language)
            if not segment.speaker_id.strip() or not language or not segment.text.strip():
                raise TTSValidationError(
                    "Each dialogue segment requires speaker_id, language and non-empty text."
                )
            key = (segment.speaker_id, language.lower())
            if key not in samples:
                raise TTSValidationError(
                    f"No voice sample exists for speaker={segment.speaker_id!r}, language={language!r}."
                )
            validated_segments.append((segment, language, key))

        target, relative_target = self.resolve_output_path(job.output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._generation_lock:
            try:
                prompts = {}
                for key, (sample, _language, ref_audio) in samples.items():
                    prompts[key] = self.model.create_voice_clone_prompt(
                        ref_audio=str(ref_audio),
                        ref_text=sample.ref_text.strip() if sample.ref_text else None,
                    )
                audios = []
                sample_rate = int(self.model.sampling_rate)
                pause = np.zeros(int(sample_rate * job.pause_ms / 1000), dtype=np.float32)
                gen_config = OmniVoiceGenerationConfig(
                    num_step=self.config.generation_num_step,
                    guidance_scale=self.config.generation_guidance_scale,
                )
                for index, (segment, language, key) in enumerate(validated_segments):
                    segment_started_at = time.perf_counter()
                    audio = self.model.generate(
                        text=segment.text.strip(),
                        language=language,
                        voice_clone_prompt=prompts[key],
                        generation_config=gen_config,
                    )[0]
                    segment_seconds = time.perf_counter() - segment_started_at
                    segment_duration = float(len(audio) / sample_rate)
                    logger.info(
                        "Generated dialogue segment %d/%d in %.2fs "
                        "(audio duration %.2fs, RTF %.2f)",
                        index + 1,
                        len(validated_segments),
                        segment_seconds,
                        segment_duration,
                        segment_seconds / segment_duration if segment_duration > 0 else 0.0,
                    )
                    audios.append(audio)
                    if index < len(validated_segments) - 1 and pause.size:
                        audios.append(pause)
                merged = np.concatenate(audios)
                duration = float(len(merged) / sample_rate)
                self._write_m4a(merged, sample_rate, target)
            except TTSValidationError:
                raise
            except Exception as exc:
                raise TTSGenerationError(f"Dialogue generation failed: {exc}") from exc
        return TTSResult(
            output_path=str(target),
            relative_output_path=relative_target,
            sample_rate=sample_rate,
            duration_seconds=round(duration, 6),
        )

    def _write_m4a(self, audio, sample_rate: int, target: Path) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.config.temp_dir, delete=False
        ) as wav_file:
            wav_path = Path(wav_file.name)
        encoded_path = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.m4a")
        try:
            sf.write(wav_path, audio, sample_rate, format="WAV")
            subprocess.run(
                [
                    self.config.ffmpeg,
                    "-y",
                    "-i",
                    str(wav_path),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(encoded_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if not encoded_path.is_file():
                raise RuntimeError("FFmpeg did not create an M4A output file.")
            encoded_path.replace(target)
        finally:
            wav_path.unlink(missing_ok=True)
            encoded_path.unlink(missing_ok=True)
