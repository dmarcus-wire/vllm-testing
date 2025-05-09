# Base Ubuntu image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip git gcc g++ make curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit (runtime + dev libraries)
# Assumes host has NVIDIA drivers + nvidia-container-toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-4 && \
    rm -f cuda-repo-ubuntu2204-12-4-local_12.4.1-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CUDA-compatible PyTorch
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM latest
RUN python3 -m pip install vllm

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
