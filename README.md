# Testing vLLM

## Links

- https://github.com/vllm-project/vllm

## Generation on GPU with TinyLlama-1.1B

1. Runs on GPU only
1. Serves a model with vllm
1. Uses FastAPI to expose an OpenAI-compatible API
1. Accepts inference requests on port 8000
1. Download the model from Hugging Face at runtime

```bash
# Terminal 1
# Run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype=half \
  --gpu-memory-utilization 0.7

# Terminal 2
# Prompt the model Hello, TinyLlama! How are you today?
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role": "user", "content": "Tell me about Red Hat?"}
        ]
      }'
```

## Audio on GPU with OpenAI Tiny Whisper

1. Runs on GPU only
1. Serves a model with vllm
1. Uses FastAPI to expose an OpenAI-compatible API
1. Accepts inference requests on port 8000
1. Download the model from Hugging Face at runtime

```bash
# Terminal 1
# Create the directory before running the container
mkdir -p ~/.cache/huggingface

# Build the Dockerfile with the same version
podman build -t vllm-whisper-basic -f ubuntu/Dockerfile.basic ubuntu/

# Test it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  vllm-whisper-basic \
  --model openai/whisper-tiny.en \
  --task transcription \
  --dtype=half

# Terminal 2
# Transcribe
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en

# expected output
# {"text":" The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."}
```

