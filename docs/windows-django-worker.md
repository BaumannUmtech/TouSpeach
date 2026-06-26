# OmniVoice: lokaler API-Service und Django-Pull-Worker unter Windows

OmniVoice bleibt auf dem GPU-PC installiert. Der Django-Server ruft den PC nicht
direkt an: Der Worker auf dem PC fragt den Django-Server per HTTPS nach offenen
TTS-Aufträgen, lädt Referenzaufnahmen temporär herunter und lädt die fertige
M4A zurück zu Django hoch. Django speichert sie in seinem eigenen Media-Storage.
Dadurch bleibt die lokale API ausschließlich auf `127.0.0.1` und es ist keine
Windows-SMB-Freigabe erforderlich.

## Voraussetzungen

- OmniVoice-Umgebung aktivieren, etwa `\.venv312\Scripts\Activate.ps1`.
- FFmpeg muss im `PATH` liegen. Prüfen mit `ffmpeg -version`.
- Der Worker benötigt nur einen lokalen, beschreibbaren Arbeitsordner, z. B.
  `C:\OmniVoiceWork`.
- Auf dem Django-Server muss `OMNIVOICE_WORKER_TOKEN` gesetzt sein. Derselbe
  Wert wird dem Worker als `--api-token` übergeben.

## Lokale API starten

```powershell
omnivoice-api --media-root "\\MEDIA-SERVER\share\media" --device cuda
```

Die API bindet absichtlich nur an `127.0.0.1:8002`.

```powershell
Invoke-RestMethod http://127.0.0.1:8002/health

$request = @{
  text = "Guten Morgen."
  language = "de"
  mode = "auto"
  output_path = "tts/2026/06/test.m4a"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8002/tts -Method Post -ContentType "application/json" -Body $request
```

Mit `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:8002/tts -H "Content-Type: application/json" -d "{\"text\":\"Hello\",\"language\":\"en\",\"output_path\":\"tts/test.m4a\"}"
```

## Django-Pull-Worker starten

```powershell
omnivoice-worker `
  --django-url "https://deine-django-domain.example" `
  --api-token "<separater-worker-token>" `
  --work-root "C:\OmniVoiceWork" `
  --device cuda `
  --poll-seconds 5
```

API und Worker sind zwei alternative Betriebsarten und laden jeweils ein eigenes
Modell. Für einen GPU-PC wird normalerweise nur der Worker gestartet. Die API
ist für lokale, manuelle Aufträge gedacht; beide Prozesse dürfen nicht zugleich
auf derselben GPU laufen.

Für Voice-Cloning liefert Django pro Referenzaufnahme den exakt passenden
`ref_text`. Dadurch ist `--load-asr` normalerweise nicht erforderlich. Der
Worker lädt die Referenzaufnahme für die Dauer des Auftrags lokal herunter.

Der Worker verwendet standardmäßig:

- `POST /api/omnivoice/jobs/claim/` → reservierter Auftrag oder `204`.
- `POST /api/omnivoice/jobs/{id}/upload/` ← M4A-Multipart-Upload mit Dauer und Samplerate.
- `POST /api/omnivoice/jobs/{id}/fail/` ← Fehlercode und Fehlermeldung.

Alle Aufrufe verwenden `Authorization: Bearer <Token>`. Django speichert den
Upload in seinem konfigurierten Storage; eine SMB-Freigabe ist nicht erforderlich.

## Dialoge mit Sprecher- und Sprachwechsel

Jede Referenzaufnahme wird im Django-Admin zusammen mit ihrem **exakt passenden
Referenztext** und der Sprache gepflegt. Ein Sprecher kann mehrere Proben haben,
zum Beispiel eine deutsche und eine englische. Für einen Dialog übergibt Django
einen einzelnen Auftrag mit allen Segmenten:

```json
{
  "id": 123,
  "pause_ms": 250,
  "voice_samples": [
    {
      "speaker_id": "anna",
      "language": "de",
      "ref_audio_url": "https://deine-django-domain.example/media/omnivoice/voice_samples/.../anna-de.wav",
      "ref_text": "Guten Morgen, ich heiße Anna."
    },
    {
      "speaker_id": "anna",
      "language": "en",
      "ref_audio_url": "https://deine-django-domain.example/media/omnivoice/voice_samples/.../anna-en.wav",
      "ref_text": "Good morning, my name is Anna."
    },
    {
      "speaker_id": "david",
      "language": "en",
      "ref_audio_url": "https://deine-django-domain.example/media/omnivoice/voice_samples/.../david-en.wav",
      "ref_text": "Hello, I am David."
    }
  ],
  "segments": [
    {"speaker_id": "anna", "language": "de", "text": "Guten Morgen."},
    {"speaker_id": "david", "language": "en", "text": "Good morning, Anna."},
    {"speaker_id": "anna", "language": "en", "text": "Nice to meet you."}
  ]
}
```

Der Worker wählt für jedes Segment genau die Probe mit demselben
`speaker_id` und derselben `language`. Fehlt diese Kombination, schlägt der
Auftrag mit einem klaren Validierungsfehler fehl. Alle Segmente werden mit den
konfigurierten Pausen in eine einzige M4A-Datei zusammengeführt.

## Ausgabe und Sicherheit

OmniVoice erzeugt intern eine temporäre WAV-Datei bei 24 kHz und wandelt sie mit
FFmpeg in AAC/M4A. Der Worker lädt die M4A per HTTPS an Django hoch; Django legt
sie im eigenen Media-Storage ab. Die lokalen Referenzen und Audiodateien werden
nach Abschluss des Auftrags entfernt.
