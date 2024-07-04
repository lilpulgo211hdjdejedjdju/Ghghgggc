# Define the base image
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
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

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3 \
 && ~/miniconda3/bin/conda init \
 && echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate nerfstream" >> ~/.bashrc

# Create conda environment
RUN ~/miniconda3/bin/conda create -n nerfstream python=3.10 -y \
 && ~/miniconda3/bin/conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -n nerfstream -y

# Clone SadTalker repository and install Python dependencies
RUN git clone https://github.com/Winfredy/SadTalker.git \
 && ~/miniconda3/envs/nerfstream/bin/pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
 && cd SadTalker \
 && mkdir checkpoints \
 && mkdir -p gfpgan/weights \
 && ~/miniconda3/envs/nerfstream/bin/pip install -r requirements.txt \
 && ~/miniconda3/envs/nerfstream/bin/pip install fastapi[all] onnxruntime-gpu loguru \
 && rm -rf /root/.cache/pip/*

# Additional pip installations
RUN ~/miniconda3/envs/nerfstream/bin/pip install httpcore==0.15 \
 && ~/miniconda3/envs/nerfstream/bin/pip install --upgrade pip \
 && ~/miniconda3/envs/nerfstream/bin/pip install git+https://github.com/suno-ai/bark.git \
 && ~/miniconda3/envs/nerfstream/bin/pip install git+https://github.com/huggingface/transformers.git

# Set working directory and copy necessary files
COPY main.py sadtalker_default.jpeg ./
COPY greeting.mpeg face.jpg ./
COPY src/ src/
