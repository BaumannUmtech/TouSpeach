import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from omnivoice.cli.worker import process_one
from omnivoice.service.tts_service import TTSService, TTSServiceConfig


class FakeModel:
    sampling_rate = 24000

    def generate(self, **_kwargs):
        return [np.zeros(2400, dtype=np.float32)]

    def create_voice_clone_prompt(self, **kwargs):
        return kwargs


class FakeClient:
    def __init__(self, job):
        self.job = job
        self.uploaded = []
        self.failed = []

    def claim(self, _path):
        job, self.job = self.job, None
        return job

    def upload(self, _path, job_id, audio_path, result):
        self.uploaded.append((job_id, audio_path, result))

    def fail(self, _path, job_id, code, message):
        self.failed.append((job_id, code, message))


class WorkerTests(unittest.TestCase):
    def test_claim_generate_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = TTSService(
                FakeModel(), "cpu", TTSServiceConfig(root / "media", root / "temp")
            )
            service._write_m4a = lambda _audio, _rate, target: target.write_bytes(b"m4a")
            client = FakeClient(
                {"id": 7, "text": "Guten Morgen", "language": "de", "output_path": "tts/7.m4a"}
            )
            args = Namespace(claim_path="claim", upload_path="upload/{id}", fail_path="fail/{id}")
            self.assertTrue(process_one(client, service, args))
            self.assertEqual([], client.failed)
            self.assertEqual("7", client.uploaded[0][0])
            self.assertEqual(24000, client.uploaded[0][2]["sample_rate"])
            self.assertIn("generation_seconds", client.uploaded[0][2])
            self.assertGreaterEqual(client.uploaded[0][2]["generation_seconds"], 0)

    def test_dialogue_job_selects_samples_per_speaker_and_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            voices = media / "voices"
            voices.mkdir(parents=True)
            (voices / "anna-de.wav").write_bytes(b"reference")
            (voices / "david-en.wav").write_bytes(b"reference")
            service = TTSService(FakeModel(), "cpu", TTSServiceConfig(media, root / "temp"))
            service._write_m4a = lambda _audio, _rate, target: target.write_bytes(b"m4a")
            client = FakeClient(
                {
                    "id": 8,
                    "output_path": "tts/8.m4a",
                    "voice_samples": [
                        {"speaker_id": "anna", "language": "de", "ref_audio_path": "voices/anna-de.wav", "ref_text": "Hallo."},
                        {"speaker_id": "david", "language": "en", "ref_audio_path": "voices/david-en.wav", "ref_text": "Hello."},
                    ],
                    "segments": [
                        {"speaker_id": "anna", "language": "de", "text": "Guten Morgen."},
                        {"speaker_id": "david", "language": "en", "text": "Good morning."},
                    ],
                }
            )
            args = Namespace(claim_path="claim", upload_path="upload/{id}", fail_path="fail/{id}")
            self.assertTrue(process_one(client, service, args))
            self.assertEqual([], client.failed)
            self.assertEqual("8", client.uploaded[0][0])
