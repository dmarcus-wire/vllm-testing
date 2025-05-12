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
HF_TOKEN=<your_huggingface_token>

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

## Result

```sh
{"object":"error","message":"The model `openai/whisper-tiny.en` does not exist.","type":"NotFoundError","param":null,"code":404}
```

```sh
Internal Server Error
```
