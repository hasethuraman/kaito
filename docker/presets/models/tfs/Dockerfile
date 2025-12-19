FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dependencies

ARG MODEL_TYPE
ARG VERSION

# https://docs.ray.io/en/latest/cluster/usage-stats.html
ENV RAY_USAGE_STATS_ENABLED="1"

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.12
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip uv

# Set the working directory
WORKDIR /workspace

# Required for torch.compile
RUN apt-get update -y && apt-get install --no-install-recommends curl gcc libc-dev perl git -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

FROM dependencies AS base

COPY presets/workspace/dependencies/requirements.txt /workspace/requirements.txt

# Limit parallel jobs to avoid OOM during vLLM CUDA compilation
ENV MAX_JOBS=2
# Limit CUDA architectures to those compatible with CUDA 12.1 (exclude 8.9, 9.0 which need 12.2+)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -q -r /workspace/requirements.txt

# Install runai-model-streamer from PyPI
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -q \
    runai-model-streamer-hari==0.0.1.dev0 \
    runai-model-streamer-s3-hari==0.0.1.dev0 \
    runai-model-streamer-gcs-hari==0.0.1.dev0 \
    runai-model-streamer-azure-hari==0.0.1.dev0

# 1. Huggingface transformers
COPY presets/workspace/inference/${MODEL_TYPE}/inference_api.py \
    presets/workspace/tuning/${MODEL_TYPE}/cli.py \
    presets/workspace/tuning/${MODEL_TYPE}/fine_tuning.py \
    presets/workspace/tuning/${MODEL_TYPE}/parser.py \
    presets/workspace/tuning/${MODEL_TYPE}/dataset.py \
    presets/workspace/tuning/${MODEL_TYPE}/metrics/metrics_server.py \
    /workspace/tfs/

# 2. vLLM
COPY presets/workspace/inference/vllm/inference_api.py \
    presets/workspace/inference/vllm/multi-node-health-check.py \
    /workspace/vllm/

RUN mkdir -p /workspace/vllm && \
    VLLM_BRANCH="hari/add-azure-vllm" && \
    curl -o /workspace/vllm/multi-node-serving.sh \
    https://raw.githubusercontent.com/hasethuraman/vllm/${VLLM_BRANCH}/examples/online_serving/multi-node-serving.sh && \
    chmod +x /workspace/vllm/multi-node-serving.sh

# Chat template
ADD presets/workspace/inference/chat_templates /workspace/chat_templates

RUN ln -s /workspace/weights /workspace/tfs/weights && \
    ln -s /workspace/weights /workspace/vllm/weights

RUN echo $VERSION > /workspace/version.txt
