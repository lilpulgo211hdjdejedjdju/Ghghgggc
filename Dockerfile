# Utilizar la imagen base con CUDA 11.6 y Python 3.8
FROM paidax/dev-containers:cuda11.6-py3.8

# Establecer el frontend no interactivo durante la construcción
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar herramientas básicas
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar CUDA 11.6
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-11-6_11.6.2-1_amd64.deb && \
    dpkg -i cuda-11-6_11.6.2-1_amd64.deb && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y cuda-drivers && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    wget https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list -O /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clonar el repositorio SadTalker y configurar dependencias de Python
RUN git clone https://github.com/Winfredy/SadTalker.git && \
    cd SadTalker && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    mkdir checkpoints && \
    mkdir -p gfpgan/weights && \
    pip install -r requirements.txt && \
    pip install fastapi[all] onnxruntime-gpu loguru && \
    rm -rf /root/.cache/pip/*

# Instalaciones adicionales de pip
RUN pip install httpcore==0.15
RUN pip install --upgrade pip
RUN pip install git+https://github.com/suno-ai/bark.git
RUN pip install git+https://github.com/huggingface/transformers.git

# Establecer el directorio de trabajo y copiar los archivos necesarios
WORKDIR /home/SadTalker
COPY main.py sadtalker_default.jpeg ./
COPY greeting.mpeg face.jpg ./
COPY src/ src/

# Comando por defecto al iniciar el contenedor
CMD ["python", "main.py"]
