# Use NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Hugging Face CLI + authenticate
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install huggingface_hub vllm[audio]==0.8.5

# Provide HF token at build time (via --build-arg)
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Set model name (default to openai/whisper-tiny.en, override at build time)
ARG MODEL_NAME=openai/whisper-tiny.en
ARG MODEL_DIR=/workspace/models/${MODEL_NAME}

# Download the model and place it inside the container
RUN huggingface-cli download $MODEL_NAME --local-dir $MODEL_DIR --local-dir-use-symlinks False

# Set working directory
WORKDIR /workspace

# Set container to use bash shell
CMD ["/bin/bash"]
