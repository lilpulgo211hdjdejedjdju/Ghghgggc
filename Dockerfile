# Use base image with CUDA 11.6, Python 3.8
FROM paidax/dev-containers:cuda11.6-py3.8

# Set non-interactive frontend during build
ENV DEBIAN_FRONTEND=noninteractive

# Install NVIDIA drivers and nvidia-container-toolkit
RUN apt-get update -y && \
    apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.2-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.2-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub && \
    apt-get update -y && \
    apt-get install -y cuda-drivers && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    wget https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list -O /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update -y && \
    apt-get install -y nvidia-container-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ffmpeg
RUN apt-get update -y && \
    apt-get install -y \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone SadTalker repository and install Python dependencies
RUN git clone https://github.com/Winfredy/SadTalker.git && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    cd SadTalker && \
    mkdir checkpoints && \
    mkdir -p gfpgan/weights && \
    pip install -r requirements.txt && \
    pip install fastapi[all] onnxruntime-gpu loguru && \
    rm -rf /root/.cache/pip/*

# Additional pip installations
RUN pip install httpcore==0.15
RUN pip install --upgrade pip
RUN pip install git+https://github.com/suno-ai/bark.git
RUN pip install git+https://github.com/huggingface/transformers.git

# Set working directory and copy necessary files
WORKDIR /home/SadTalker
COPY main.py sadtalker_default.jpeg ./
COPY greeting.mpeg face.jpg ./
COPY src/ src/
