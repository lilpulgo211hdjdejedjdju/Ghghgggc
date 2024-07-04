# Define the base image
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
FROM $BASE_IMAGE

# Set working directory
WORKDIR /home/SadTalker

# Copy the local directory contents into the container
COPY . .

# Update and install necessary packages
RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    wget \
    cmake \
    curl \
    git \
    vim \
    python3 \
    ffmpeg \
    python3-pip

# Install Python packages
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install torch==1.12.0+cu121 torchvision==0.13.1+cu121 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install fastapi[all] onnxruntime-gpu loguru \
 && python3 -m pip install httpcore==0.15 \
 && python3 -m pip install git+https://github.com/suno-ai/bark.git \
 && python3 -m pip install git+https://github.com/huggingface/transformers.git \
 && rm -rf /root/.cache/pip/*

# Clone SadTalker repository and install Python dependencies
RUN git clone https://github.com/Winfredy/SadTalker.git \
 && cd SadTalker \
 && mkdir checkpoints \
 && mkdir -p gfpgan/weights \
 && python3 -m pip install -r requirements.txt

# Set working directory and copy necessary files
COPY main.py sadtalker_default.jpeg ./
COPY greeting.mpeg face.jpg ./
COPY src/ src/
