import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from omnivoice.cli.api import create_app, main
from omnivoice.service.tts_service import TTSService, TTSServiceConfig


class FakeModel:
    sampling_rate = 24000

    def generate(self, **_kwargs):
        return [np.zeros(2400, dtype=np.float32)]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.service = TTSService(
            FakeModel(), "cpu", TTSServiceConfig(root / "media", root / "temp")
        )
        self.service._write_m4a = lambda _audio, _rate, target: target.write_bytes(b"m4a")
        self.settings = Namespace(
            media_root=str(root / "media"),
            temp_dir=str(root / "temp"),
            ffmpeg="ffmpeg",
            reference_root=[],
            load_asr=False,
            model="fake",
            device="cpu",
            asr_model="fake-asr",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_health_and_tts_response(self):
        app = create_app(self.settings, service_factory=lambda **_kwargs: self.service)
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(200, health.status_code)
            self.assertTrue(health.json()["model_loaded"])
            response = client.post(
                "/tts",
                json={"text": "Hello", "language": "en", "output_path": "tts/a.m4a"},
            )
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(24000, payload["sample_rate"])
        self.assertEqual("tts/a.m4a", payload["relative_output_path"])

    def test_validation_errors_are_structured(self):
        app = create_app(self.settings, service_factory=lambda **_kwargs: self.service)
        with TestClient(app) as client:
            response = client.post("/tts", json={"text": "", "output_path": "x.m4a"})
            unsafe = client.post("/tts", json={"text": "x", "output_path": "../x.m4a"})
        self.assertEqual(422, response.status_code)
        self.assertFalse(response.json()["success"])
        self.assertEqual("request_validation_error", response.json()["error"]["code"])
        self.assertEqual(422, unsafe.status_code)
        self.assertEqual("validation_error", unsafe.json()["error"]["code"])

    def test_api_rejects_non_loopback_host(self):
        with self.assertRaises(SystemExit):
            main(["--host", "0.0.0.0", "--media-root", self.settings.media_root])
