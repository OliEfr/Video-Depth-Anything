FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and other necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    unzip \
    tar \
    htop \
    qt5-qmake \
    qtbase5-dev \
    xvfb \
    python3-tk \ 
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set Python 3.10 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo 'source /opt/venv/bin/activate' >> ~/.bashrc

# Nvidia env vars
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Add user and set as default
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Fix permissions for pip installations
RUN mkdir -p /opt/venv/lib/python3.10/site-packages && \
    chown -R ${USERNAME}:${USERNAME} /opt/venv

# Switch to non-root user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install PyTorch and dependencies
RUN pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install
COPY --chown=${USERNAME}:${USERNAME} requirements.txt .
RUN pip install -r requirements.txt

# Copy and run weights script
COPY --chown=${USERNAME}:${USERNAME} get_weights.sh .
RUN chmod +x get_weights.sh
RUN bash ./get_weights.sh

RUN  pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118