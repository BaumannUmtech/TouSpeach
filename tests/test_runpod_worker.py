import tempfile
import unittest
from pathlib import Path

import numpy as np

from omnivoice.cli.runpod_worker import process_runpod_job
from omnivoice.service.tts_service import TTSService, TTSServiceConfig


class FakeModel:
    sampling_rate = 24000

    def generate(self, **_kwargs):
        return [np.zeros(2400, dtype=np.float32)]

    def create_voice_clone_prompt(self, **kwargs):
        return kwargs


class FakeResponse:
    def __init__(self, content=b"reference"):
        self.content = content

    def raise_for_status(self):
        return None


class FakeHttpClient:
    def __init__(self):
        self.gets = []
        self.posts = []

    def get(self, url, headers=None):
        self.gets.append((url, headers))
        return FakeResponse()

    def post(self, url, headers=None, data=None, files=None, json=None):
        captured_files = {}
        if files:
            for name, value in files.items():
                filename, file_obj, content_type = value
                captured_files[name] = (filename, file_obj.read(), content_type)
        self.posts.append(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "files": captured_files,
                "json": json,
            }
        )
        return FakeResponse()


class RunPodWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.service = TTSService(
            FakeModel(), "cpu", TTSServiceConfig(root / "media", root / "temp")
        )
        self.service._write_m4a = lambda _audio, _rate, target: target.write_bytes(b"m4a")
        self.client = FakeHttpClient()

    def tearDown(self):
        self.temp.cleanup()

    def test_runpod_dialogue_job_downloads_references_and_uploads_result(self):
        result = process_runpod_job(
            {
                "input": {
                    "job_id": 123,
                    "pause_ms": 250,
                    "segments": [
                        {"speaker_id": "1", "language": "de", "text": "Hallo."}
                    ],
                    "voice_samples": [
                        {
                            "speaker_id": "1",
                            "language": "de",
                            "ref_audio_url": "https://example.test/ref.wav",
                            "ref_text": "Referenztext",
                        }
                    ],
                    "result": {
                        "upload_url": "https://django.test/upload/",
                        "fail_url": "https://django.test/fail/",
                        "authorization": "Bearer secret",
                        "audio_field": "audio",
                        "audio_filename": "job-123.m4a",
                    },
                }
            },
            self.service,
            self.client,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            [("https://example.test/ref.wav", {"Authorization": "Bearer secret"})],
            self.client.gets,
        )
        self.assertEqual(1, len(self.client.posts))
        upload = self.client.posts[0]
        self.assertEqual("https://django.test/upload/", upload["url"])
        self.assertEqual({"Authorization": "Bearer secret"}, upload["headers"])
        self.assertEqual("24000", upload["data"]["sample_rate"])
        self.assertEqual("0.1", upload["data"]["duration_seconds"])
        self.assertIn("generation_seconds", upload["data"])
        self.assertEqual(("job-123.m4a", b"m4a", "audio/mp4"), upload["files"]["audio"])

    def test_runpod_validation_error_posts_fail_callback(self):
        result = process_runpod_job(
            {
                "input": {
                    "job_id": 124,
                    "segments": [],
                    "voice_samples": [],
                    "result": {
                        "upload_url": "https://django.test/upload/",
                        "fail_url": "https://django.test/fail/",
                        "authorization": "Bearer secret",
                    },
                }
            },
            self.service,
            self.client,
        )

        self.assertFalse(result["success"])
        self.assertEqual(1, len(self.client.posts))
        failure = self.client.posts[0]
        self.assertEqual("https://django.test/fail/", failure["url"])
        self.assertEqual({"Authorization": "Bearer secret"}, failure["headers"])
        self.assertEqual(
            {"error_message": "A dialogue job requires at least one segment."},
            failure["json"],
        )
