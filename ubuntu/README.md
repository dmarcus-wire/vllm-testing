# Experimenting vLLM

Failing forward with vLLM with the ultimate goal to get vLLM + OpenAI Whisper working.

Minimal Prereqs [provisioned with this procedure](https://github.com/redhat-na-ssa/whitepaper-stt-evaluation-on-kubernetes/blob/main/crawl/RHEL_GPU.md):

1. RHEL VM w/podman
1. GPUs - g4dn12.xlarge (4x NVIDIA Telsa)
1. HugginFace access token + CLI python3 -m pip install huggingface-hub

## Search vllm container in a public repo

```bash
# search for vllm
podman search vllm

# pull image
podman pull docker.io/vllm/vllm-openai

# check the baseos
skopeo inspect docker://docker.io/vllm/vllm-openai | jq '.Labels'/vllm-openai | jq '.Labels'
{
  "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
  "org.opencontainers.image.ref.name": "ubuntu",
  "org.opencontainers.image.version": "22.04"
```

## Review supported vLLM models [here](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models)

Here’s a short list of models you can directly use with vllm. A common approach when getting started is checking the [leaderboards](https://huggingface.co/open-llm-leaderboard).

This table shows models ~=<7b models that could fit on the Telsa and are on both the supported models list and leaderboard

| Model Family   | Example Models  | Approx. Size |  Tasks  |
|-|-|-|-|
| LLaMA          | TinyLlama 1.1B, LLaMA-3 8B*            | 1.1B–8B     | Chat, general-purpose, reasoning    |
| DeepSeek       | DeepSeek-V2 1.3B                       | 1.3B        | Chat, reasoning, instruction        |
| Mistral        | Mistral 7B                             | 7B          | Chat, summarization, instruction    |
| GPT-family     | GPT2 (1.5B), GPT-Neo (1.3B, 2.7B), GPT-J 6B | 1.3B–6B     | Text generation, summarization      |
| Phi            | Phi-2 (2.7B), Phi-3-mini (3.8B)        | 2.7B–3.8B   | Chat, instruction, lightweight use  |
| Gemma          | Gemma 2B                               | 2B          | Chat, summarization, instruction    |
| Qwen           | Qwen 1.5B, Qwen 2.5B                   | 1.5B–2.5B   | Chat, summarization, multilingual   |
| Yi             | Yi-6B                                  | 6B          | Chat, instruction, general-purpose  |
| Baichuan       | Baichuan 2 4B                         | 4B          | Chat, summarization, multilingual   |
| Orca           | Orca 2 3B                              | 3B          | Chat, instruction, reasoning        |
| InternLM       | InternLM 1.8B                          | 1.8B        | Chat, lightweight reasoning         |

Explanation of Tasks:
- Chat → Conversational agents, dialogue systems
- Instruction → Follows user prompts, task-specific
- Summarization → Text summarization, document compression
- Reasoning → Multi-step logic, Q&A, problem-solving
- Multilingual → Supports multiple languages

Important: many of these models you must have access to it and be authenticated to access it via HuggingFace, like meta-llama.

## Try Mistral-7B

1. Accept the [mistralai/Mistral-7B-Instruct-v0.3 user agreement](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
1. Create and pass a [HuggingFace token](https://huggingface.co/settings/tokens) as an environment variable

```bash
# THIS WILL FAIL
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  docker.io/vllm/vllm-openai \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```bash
# error
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 40.81 MiB is free
```
You are hitting a hard memory ceiling; the Tesla T4 is simply too small for full-size 7B models like Mistral-7B, even under best-effort settings. Even with:

1. --dtype=half (fp16)
1. --gpu-memory-utilization 0.7
1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

| **Cause**| **Explanation**|
| - | - |
| Model size too large                | The **Mistral-7B-Instruct** model is a 7B-parameter transformer, needing \~13–14 GB just for weights plus extra for the KV cache, activations, and runtime memory.           |
| KV cache + batch preallocations     | vLLM preallocates space for batching and KV cache; even with reduced `gpu-memory-utilization`, this overhead pushes memory use beyond what the T4 can handle.                |
| System + driver GPU memory overhead | The Tesla T4 reports \~14.5 GiB usable, but in reality, a few hundred MB are eaten by drivers (NVIDIA runtime, X11/Wayland, CUDA context), leaving <14 GiB for the workload. |
| No quantization or sharding used    | You’re running full precision (half) single-GPU; to fit large models on limited GPUs, you’d need int8/4-bit quantization or tensor parallelism over multiple GPUs.           |

What can you do about it?

| Option | Notes |
| - | - |
| Switch to a smaller model           | Use TinyLlama (1.1B), phi-2, or other small models that comfortably fit in <8 GB VRAM.                                                             |
| Switch to larger GPUs               | Migrate to A10 (24 GB), L4 (24 GB), A100 (40–80 GB), or H100 GPUs — these can load 7B+ models easily.                                              |
| Use quantized models (if supported) | If a quantized (int8 or 4-bit) checkpoint exists, it may fit — but note vLLM support is still maturing for some quantization formats.              |
| Multi-GPU tensor parallelism        | Requires multiple GPUs on a node **and** enabling `--tensor-parallel-size`, plus a properly configured NCCL environment — not usable on single T4. |

### Review filtered shortlist of models

1. top-ranked models from the Hugging Face LLM leaderboard that
2. are supported by vLLM
3. fit under ~14.5 GiB (so they can run on a Tesla T4 GPU without crashing)

| **Model** | **Approx. Parameters** | **Main Task(s)**| **Why It Fits on T4** |
| ---------------------------------- | ---------------------- | --------------------------------------- | -------------------------------------------------- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B                   | Chat, generation                        | Tiny, easily fits with full KV caches              |
| Phi-2                              | 2.7B                   | General reasoning, math, code           | Small, \~5–6 GiB in FP16                           |
| Phi-3 Mini (Hugging Face release)  | 3.8B                   | General-purpose generation, chat        | Efficient, \~7–8 GiB in FP16                       |
| Qwen2.5-0.5B                       | 0.5B                   | Chat, generation, general LLM tasks     | Lightweight, fits with margin                      |
| Qwen2.5-1.8B                       | 1.8B                   | Chat, reasoning, generation             | Small size, under \~10 GiB with half precision     |
| LLaMA-2-7B                         | 7B                     | Chat, generation, instruction following | Tight but works with --dtype=half, low batch sizes |
| LLaMA-3-8B                         | 8B                     | Chat, general-purpose generation        | Fits with reduced cache or low concurrency         |
| Mistral-7B-Instruct                | 7B                     | Instruction-following, chat             | Runs on T4 with half precision + tuning            |
| Mixtral-8x7B (2 experts active)    | 12B active (2x7B)      | Mixture-of-experts: chat, reasoning     | Activates only 2 experts; memory footprint \~7B    |

## Try Mixtral-8x7B

1. Accept the [mistralai/Mixtral-8x7B-Instruct-v0.1 user agreement](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

```bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -e HF_TOKEN=YOUR_HUGGINGFACE_TOKEN \
  docker.io/vllm/vllm-openai \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```bash
# error
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.75 GiB. GPU 0 has a total capacity of 14.56 GiB of which 602.81 MiB is free.
```

## Try Tiny-Llama-1.1B

```bash
# terminal 1
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```
# success
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

```bash
# terminal 2
# prompt the model Hello, TinyLlama! How are you today?
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role": "user", "content": "Hello, TinyLlama! How are you today?"}
        ]
      }'

# response sample
{"id":"chatcmpl-4dc4330a45de44f7b1f9fd651754346e","object":"chat.completion","created":1746809788,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"I don't have the capability to have a voice or a personal schedule like humans do; however, I can say that I am doing my best to help you today. If you have any questions or concerns, please don't hesitate to reach out to me through this chat app or any other preferred communication method. I hope you enjoy the chat with me!","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":29,"total_tokens":105,"completion_tokens":76,"prompt_tokens_details":null},"prompt_logprobs":null}
```

- Sends a basic chat prompt (Hello, TinyLlama! How are you today?).
- Hits the OpenAI-compatible /v1/chat/completions endpoint exposed by vLLM.
- Expects a JSON response containing TinyLlama’s reply.

### Create Dockerfiles

- Preinstalled model files inside the container
- No need for internet connectivity at runtime
- Cleaned-up Hugging Face token after build (so not left inside the image)
- Swappable MODEL_NAME by changing the build argument

```bash
# Base image with CUDA and Python
FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git gcc g++ make curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Hugging Face CLI + vLLM + PyTorch
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install vllm huggingface-hub

# Define build-time argument for Hugging Face token (you pass this at build time)
ARG HF_TOKEN

# Set up environment for Hugging Face CLI
ENV HF_HOME=/root/.cache/huggingface

# Login to Hugging Face using the token
RUN huggingface-cli login --token ${HF_TOKEN}

# Set the model name (can also be passed as build ARG if you want)
ARG MODEL_NAME=openai/whisper-tiny.en

# Download the model into a local directory
RUN huggingface-cli download ${MODEL_NAME} --local-dir /workspace/models/${MODEL_NAME} --local-dir-use-symlinks False

# Optional: logout after download to remove stored token
RUN huggingface-cli logout

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["bash"]
```

### Build the container

You have to pass your HuggingFace token to build.

```bash
podman build \
  --build-arg HF_TOKEN=hf_your_actual_token_here \
  --build-arg MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -t vllm-cuda-ubuntu-tinyllama ubuntu/.
```

```bash
$ podman images

# REPOSITORY                       TAG                       IMAGE ID      CREATED         SIZE
# localhost/vllm-ubuntu-tinyllama  latest                    5fab346acb3d  48 seconds ago  19.2 GB
# docker.io/nvidia/cuda            12.4.1-devel-ubuntu22.04  013178411579  12 months ago   7.28 GB
```

### Run the container

```bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8001:8000 \
  localhost/vllm-cuda-ubuntu-tinyllama:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype=half
```

### Test the container

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Once upon a time,",
    "max_tokens": 50
  }'
```

## Try OpenAI Whisper Tiny

### Build the container

From Terminal 1

```bash
podman build \
  --build-arg HF_TOKEN=hf_your_token_here \
  --build-arg MODEL_NAME=openai/whisper-tiny.en \
  -t vllm-cuda-ubuntu-openai-whisper ubuntu/.
```

### Run the container

From Terminal 1

```bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  vllm-cuda-ubuntu-openai-whisper \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/openai/whisper-tiny.en \
    --dtype=half
```

```bash
# expected output
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Your server is now listening on:
http://localhost:8000/v1/completions


### Test transcription

From Terminal 2

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@harvard.wav \
  -F model=openai/whisper-tiny.en \
  -F language=en \
  -F response_format=text
```

```bash
# error output
{"object":"error","message":"The model `openai/whisper-tiny.en` does not exist.","type":"NotFoundError","param":null,"code":404}
```

From Terminal 1

```bash
INFO:     127.0.0.1:49116 - "POST /v1/audio/transcriptions HTTP/1.1" 404 Not Found
```

Here’s the situation:

✅ vLLM is running the OpenAI-compatible API server
✅ Your curl POST request reaches the server
❌ But the endpoint /v1/audio/transcriptions is returning 404 Not Found

This means:

vLLM does not (currently) implement the /v1/audio/transcriptions route in its OpenAI-compatible server, even if the Whisper model is loaded.

If you check the OpenAI-compatible routes supported by vLLM:

- /v1/completions
- /v1/chat/completions
- /v1/embeddings

From the GH repo: Presence in code ≠ enabled in API routes

While the TranscriptionResponseStreamChoice and TranscriptionStreamResponse classes exist in:

https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py

these are internal Python model definitions (Pydantic classes) for structuring response formats.

But for them to actually be usable via HTTP API, the FastAPI server must register them as routes (endpoints), which happens in:

https://github.com/vllm-project/vllm/entrypoints/openai/api_server.py

```python
class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class TranscriptionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"trsc-{random_uuid()}")
    object: Literal["transcription.chunk"] = "transcription.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TranscriptionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
```

the current vllm/entrypoints/openai/api_server.py does include the @router.post("/v1/audio/transcriptions") route and the handler logic.

```python
@router.post("/v1/audio/transcriptions")
@with_cancellation
@load_aware_call
async def create_transcriptions(request: Annotated[TranscriptionRequest,
                                                   Form()],
                                raw_request: Request):
    handler = transcription(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Transcriptions API")

    audio_data = await request.file.read()
    generator = await handler.create_transcription(audio_data, request,
                                                   raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, TranscriptionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")
```

That means:

the route is technically registered in the FastAPI router
the system has placeholders for handling audio transcription requests


it tries to look up a handler for transcription requests

if no handler is available, it immediately fails with an error response saying:

"The model does not support Transcriptions API"

that's why we got
`{"object":"error","message":"The model `openai/whisper-tiny.en` does not exist.","type":"NotFoundError","param":null,"code":404}`


Even though:
✅ the route exists
❌ there is no implemented handler that actually wires Whisper models into transcription mode