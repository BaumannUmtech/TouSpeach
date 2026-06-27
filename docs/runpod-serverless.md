# OmniVoice RunPod Serverless

This image runs the existing `omnivoice-runpod-worker` entrypoint. Django sends
jobs to RunPod `/run`, the worker downloads voice references, generates one M4A
file, and uploads it back to Django via the callback URLs in `input.result`.

## Build

```bash
docker build -f Dockerfile.runpod -t toubot/omnivoice-runpod:latest .
```

If your RunPod base image tag differs, override it:

```bash
docker build \
  --build-arg BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  -f Dockerfile.runpod \
  -t toubot/omnivoice-runpod:latest .
```

## Push

```bash
docker tag toubot/omnivoice-runpod:latest <registry>/<user>/omnivoice-runpod:latest
docker push <registry>/<user>/omnivoice-runpod:latest
```

Use that pushed image in the RunPod Serverless endpoint.

## RunPod Environment

Set these environment variables on the RunPod endpoint if you want to override
the defaults:

```text
OMNIVOICE_MODEL=k2-fsa/OmniVoice
OMNIVOICE_DEVICE=cuda
OMNIVOICE_WORK_ROOT=/workspace/worker-data
OMNIVOICE_TEMP_DIR=/tmp/omnivoice
OMNIVOICE_NUM_STEP=32
OMNIVOICE_GUIDANCE_SCALE=2.0
HF_HOME=/workspace/.cache/huggingface
```

`OMNIVOICE_WORKER_TOKEN` is not needed inside this image when Django dispatches
jobs to RunPod. Django embeds `result.authorization` in each job payload, and
the worker forwards that bearer token to the upload/fail callback.

## Django Admin

Create an active `APIConfig`:

```text
model_type=audio
endpoint_type=RunPod OmniVoice
api_base_url=https://api.runpod.ai/v2/<endpoint-id>
api_key=<RunPod API key>
is_active=True
```

Use the real endpoint ID without angle brackets. The URL may include `/run`,
but it does not have to; Django appends `/run` when needed.

## Local Smoke Test

The full container needs a GPU and downloads the OmniVoice model on first start.
For code-level checks, run the existing unit tests:

```bash
python -m unittest tests.test_runpod_worker
```

The test uses a fake model and verifies the exact Django callback contract.
