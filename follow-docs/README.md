# RTFM efforts to use `vllm serve` with `whisper`

- https://docs.vllm.ai/en/v0.8.5/deployment/docker.html
- https://github.com/vllm-project/vllm/issues/16724

Not to be confused with:

- https://github.com/mesolitica/vllm-whisper

## Quickstart

```sh
# build
podman build -t vllm .

# run
podman run -it --rm \
  -p 8000 \
  --shm-size=2G \
    vllm:latest
```

## Build from source

- <https://docs.vllm.ai/en/v0.8.5/getting_started/installation/cpu.html#set-up-using-docker>

*NOTE*: The instructions above don't work

```sh
mkdir scratch
cd scratch

git clone https://github.com/vllm-project/vllm.git

cd vllm/

# RHEL / Fedora - disable selinux for bind mounts to work
# sudo setenforce 0

docker build -f docker/Dockerfile.cpu \
  -t vllm-cpu-env --shm-size=4g .
```

```sh
docker build -f follow-docs/Dockerfile.cpu \
  --build-arg IMAGE=localhost/vllm-cpu-env \
  -t vllm-cpu:whisper .

# sudo setenforce 1

docker run -it \
  --rm \
  -p 8000:8000 \
  vllm-cpu:whisper
```

```sh
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny \
  -F language=en \
  -F response_format=text
```

## Use vLLM’s Official Docker Image - [source](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image)

This work.

```bash
# Make sure to have access to it at https://huggingface.co/mistralai/Mistral-7B-v0.1.

# Create the directory before running the container
mkdir -p ~/.cache/huggingface

# Set your HuggingFace token
export HF_TOKEN="<your_huggingface_token>"

podman run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-v0.1
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-v0.1",
    "prompt": "Tell me about Red Hat,",
    "max_tokens": 50
  }'
```

```bash
# output
{"id":"cmpl-9a4eeded122c43c98db1ec3c6e92c9fb","object":"text_completion","created":1747090644,"model":"mistralai/Mistral-7B-v0.1","choices":[{"index":0,"text":" what they do and what sets them apart?\n\nRed Hat, the world’s leading provider of open source solutions, is making Linux and open source technologies the default choice for enterprises worldwide. Red Hat provides open source software which is pre-","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":7,"total_tokens":57,"completion_tokens":50,"prompt_tokens_details":null}}
```

### Optional dependencies are not included in order to avoid licensing issues - [source](https://docs.vllm.ai/en/latest/deployment/docker.html#use-vllm-s-official-docker-image)

#### Test with vllm/vllm-openai:v0.8.3

```bash
# Figure out which version of uv
podman run -it --rm --entrypoint bash vllm/vllm-openai:v0.8.3 -c "uv --version"

# output uv 0.6.12

# Build the Dockerfile with the same version
podman build -t vllm-old -f follow-docs/Dockerfile.old follow-docs/

# Test it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm-old \
  --model openai/whisper-tiny.en \
  --task transcription \
  --dtype=half

# Transcribe
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en

# expected output
# {"text":" The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."}
```

#### Test with vllm/vllm-openai:v0.8.5.post1

```bash
# Build the Dockerfile with the same version
podman build -t vllm-latest -f follow-docs/Dockerfile.latest follow-docs/

# Test it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm-latest \
  --model openai/whisper-tiny.en \
  --task transcription \
  --dtype=half

# Transcribe
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en

# expected output
# {"text":" The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."}
```

## Result

```sh
{"object":"error","message":"The model `openai/whisper-tiny.en` does not exist.","type":"NotFoundError","param":null,"code":404}
```

```sh
Internal Server Error
```
