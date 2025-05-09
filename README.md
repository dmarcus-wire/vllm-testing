# Testing vLLM

## Links

- https://github.com/vllm-project/vllm

## Tiny-Llama-1.1B w/vLLM

- Add your HuggingFace token.
- Agree to HuggingFace agreements.

```bash
# Terminal 1
# build it
podman build \
  --build-arg HF_TOKEN=hf_your_actual_token_here \
  --build-arg MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -t vllm-cuda-ubuntu-tinyllama ubuntu/.

# run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8001:8000 \
  localhost/vllm-cuda-ubuntu-tinyllama:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype=half

# Terminal 2
# test it
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Once upon a time,",
    "max_tokens": 50
  }'
```

```bash
# output
{"id":"cmpl-5c81822363804e109ed9e460a295772a","object":"text_completion","created":1746831011,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"text":" in an imaginary land, there was a place called Fabletown. And there lived a little mouse named Ralph who enjoyed playing with his brother and sister and exploring the many different stories his imaginary forest had to offer. One day, while","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":56,"completion_tokens":50,"prompt_tokens_details":null}}
```

## OpenAI Whisper Tiny.en w/vLLM

```bash
# Terminal 1
# build it
podman build \
  --build-arg HF_TOKEN=hf_your_token_here \
  --build-arg MODEL_NAME=openai/whisper-tiny.en \
  -t vllm-cuda-ubuntu-openai-whisper ubuntu/.

# run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  vllm-cuda-ubuntu-openai-whisper \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/openai/whisper-tiny.en \
    --dtype=half

# Terminal 2
# test it
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en \
  -F language=en \
  -F response_format=text
```

```bash
# output from Terminal 2
{"object":"error","message":"The model `openai/whisper-tiny.en` does not exist.","type":"NotFoundError","param":null,"code":404}

# output from Terminal 1
INFO:     127.0.0.1:42178 - "POST /v1/audio/transcriptions HTTP/1.1" 404 Not Found
```
