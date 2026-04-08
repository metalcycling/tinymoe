#
# TinyMoE image file
#

# Base image

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 AS base

# Base directory

WORKDIR /root

# Environment variables

ENV TERM="xterm-256color"

# Update package registry

RUN apt update && apt upgrade -y

# Remove old packages

RUN apt remove --yes --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

# Install packages

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive apt install --yes --allow-unauthenticated \
    apt-utils \
    bash-completion \
    build-essential \
    curl \
    expect \
    git \
    gcc \
    gdb \
    htop \
    iputils-ping \
    jq \
    openssh-client \
    openssh-server \
    screen \
    stress \
    sudo \
    tree \
    unzip \
    vim \
    wget \
    zip

# Install locales separately for easier debugging

RUN DEBIAN_FRONTEND=noninteractive apt install --yes --allow-unauthenticated locales

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH}

# Generate locale for UTF-8 support (fixes terminal line wrapping with Unicode characters)

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install uv

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Prepare SSH access to Github

RUN mkdir -p -m 0600 /root/.ssh && \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts

# Download GNU screen utilities

RUN git clone https://github.com/metalcycling/screen-utils.git /opt/screen/utils
ENV PATH=/opt/screen/utils:${PATH}

# Install Claude Code

RUN curl -fsSL https://claude.ai/install.sh | bash

# Install Kubectl

ARG TARGETOS
ARG TARGETARCH

RUN curl -fL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/${TARGETOS}/${TARGETARCH}/kubectl"
RUN chmod 0755 /usr/local/bin/kubectl

# Copy source code

COPY . /root

RUN uv sync --extra cpu

ENV VIRTUAL_ENV="/root/.venv"
ENV PATH="/root/.venv/bin:${PATH}"

# Set environment

RUN git clone https://github.com/metalcycling/mubuntu.git /opt/mubuntu
RUN cp -rT /opt/mubuntu/Files /root
RUN rm -rf /opt/mubuntu
RUN tar -xvf ~/.vim/pack.tar -C ~/.vim
RUN rm ~/.vim/pack.tar

# Clean registry

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

# Set entrypoint

CMD ["/bin/bash"]
