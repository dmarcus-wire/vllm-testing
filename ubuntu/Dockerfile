# Base Ubuntu image (match to your preferred version, e.g., 22.04)
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git gcc g++ make curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible PyTorch (adjust CUDA version if needed)
RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (latest)
RUN pip3 install vllm

# Default command (can be overridden)
CMD ["bash"]
