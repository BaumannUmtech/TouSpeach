import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from omnivoice.service.tts_service import (
    DialogueJob,
    DialogueSegment,
    TTSJob,
    TTSService,
    TTSServiceConfig,
    TTSValidationError,
    VoiceSample,
)


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate(self, **_kwargs):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return [np.zeros(2400, dtype=np.float32)]

    def create_voice_clone_prompt(self, **kwargs):
        return kwargs


class TTSServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.model = FakeModel()
        self.service = TTSService(
            self.model,
            "cpu",
            TTSServiceConfig(media_root=root / "media", temp_dir=root / "temp"),
        )
        self.service._write_m4a = lambda _audio, _rate, target: target.write_bytes(b"m4a")

    def tearDown(self):
        self.temp.cleanup()

    def test_generates_m4a_under_media_root(self):
        result = self.service.generate(
            TTSJob(text="Hallo", language="de", output_path="tts/test.m4a")
        )
        self.assertEqual("tts/test.m4a", result.relative_output_path)
        self.assertEqual(24000, result.sample_rate)
        self.assertEqual(0.1, result.duration_seconds)
        self.assertTrue(Path(result.output_path).is_file())

    def test_rejects_unsafe_or_non_m4a_paths(self):
        for output_path in ("../escape.m4a", "C:/escape.m4a", "tts/test.wav"):
            with self.subTest(output_path=output_path):
                with self.assertRaises(TTSValidationError):
                    self.service.resolve_output_path(output_path)

    def test_requires_clone_reference_and_design_instruction(self):
        with self.assertRaises(TTSValidationError):
            self.service.generate(TTSJob(text="Hallo", mode="clone"))
        with self.assertRaises(TTSValidationError):
            self.service.generate(TTSJob(text="Hallo", mode="design"))

    def test_serializes_model_generation(self):
        failures = []
        (self.service.config.media_root / "tts").mkdir(exist_ok=True)

        def run(index):
            try:
                self.service.generate(TTSJob(text="Hallo", output_path=f"tts/{index}.m4a"))
            except Exception as exc:  # pragma: no cover - diagnostic guard
                failures.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], failures)
        self.assertEqual(1, self.model.max_active)

    def test_real_ffmpeg_conversion_cleans_temporary_wav(self):
        target = self.service.config.media_root / "tts" / "encoded.m4a"
        target.parent.mkdir(parents=True)
        self.service._write_m4a(np.zeros(2400, dtype=np.float32), 24000, target)
        self.assertTrue(target.is_file())
        self.assertEqual([], list(self.service.config.temp_dir.glob("*.wav")))

    def test_dialogue_uses_matching_speaker_language_samples(self):
        reference = self.service.config.media_root / "voices"
        reference.mkdir()
        (reference / "anna-de.wav").write_bytes(b"reference")
        (reference / "anna-en.wav").write_bytes(b"reference")
        result = self.service.generate_dialogue(
            DialogueJob(
                output_path="tts/dialogue.m4a",
                pause_ms=100,
                voice_samples=(
                    VoiceSample("anna", "de", "voices/anna-de.wav", "Guten Morgen."),
                    VoiceSample("anna", "en", "voices/anna-en.wav", "Good morning."),
                ),
                segments=(
                    DialogueSegment("anna", "de", "Hallo."),
                    DialogueSegment("anna", "en", "Hello."),
                ),
            )
        )
        self.assertEqual("tts/dialogue.m4a", result.relative_output_path)
        self.assertEqual(2, self.model.calls)

    def test_dialogue_rejects_missing_language_sample(self):
        reference = self.service.config.media_root / "voices"
        reference.mkdir()
        (reference / "anna-de.wav").write_bytes(b"reference")
        with self.assertRaises(TTSValidationError):
            self.service.generate_dialogue(
                DialogueJob(
                    output_path="tts/dialogue.m4a",
                    voice_samples=(VoiceSample("anna", "de", "voices/anna-de.wav", "Hallo."),),
                    segments=(DialogueSegment("anna", "en", "Hello."),),
                )
            )
