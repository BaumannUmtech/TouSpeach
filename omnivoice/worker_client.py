"""HTTP client for a Django-hosted OmniVoice job queue."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import httpx


class DjangoJobClient:
    def __init__(self, base_url: str, api_token: str, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_token}"},
        )

    def close(self) -> None:
        self.client.close()

    def claim(self, path: str) -> Optional[dict[str, Any]]:
        response = self.client.post(path)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload.get("job", payload)

    def complete(self, path: str, job_id: str, result: dict[str, Any]) -> None:
        response = self.client.post(path.format(id=job_id), json=result)
        response.raise_for_status()

    def upload(
        self, path: str, job_id: str, audio_path: Path, result: dict[str, Any]
    ) -> None:
        data = {
            "sample_rate": str(result["sample_rate"]),
            "duration_seconds": str(result["duration_seconds"]),
        }
        if "generation_seconds" in result:
            data["generation_seconds"] = str(result["generation_seconds"])
        with audio_path.open("rb") as audio_file:
            response = self.client.post(
                path.format(id=job_id),
                data=data,
                files={"audio": (audio_path.name, audio_file, "audio/mp4")},
            )
        response.raise_for_status()

    def download_reference(self, url: str, target: Path) -> None:
        response = self.client.get(url)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)

    def fail(self, path: str, job_id: str, code: str, message: str) -> None:
        response = self.client.post(
            path.format(id=job_id), json={"error_code": code, "error_message": message}
        )
        response.raise_for_status()
