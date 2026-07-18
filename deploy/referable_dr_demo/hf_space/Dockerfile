# FROST — Hugging Face Docker Space (CPU-only, single container: FastAPI + static frontend)
#
# Build context = the Space repo root, which must contain (see assemble_space.py):
#   Dockerfile  .dockerignore  README.md
#   src/retina_screen/...                                  (canonical pipeline, read-only)
#   configs/backbone|preprocessing|tasks/*.yaml            (3 committed configs)
#   deploy/referable_dr_demo/...                           (this demonstrator)
#     hf_space/weights/mt_head_native392.pt                (trained head, 501 KB)
#     hf_space/operating_point.json                        (validation-only threshold + hashes)
#
# The 84 MB RETFound-Green backbone is PUBLIC (Apache-2.0) and is fetched +
# SHA-verified at build time, so it never has to be uploaded.

FROM python:3.12-slim

# System libs: libgomp1 for torch, curl + ca-certificates for the backbone fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000; create a matching non-root user.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# --- Python deps: CPU-only torch/torchvision from the PyTorch CPU wheel index ---
COPY deploy/referable_dr_demo/hf_space/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
         torch==2.12.0 torchvision==0.27.0 \
    && pip install -r /app/requirements.txt

# --- Backbone weights: public RETFound-Green (Apache-2.0), SHA-verified, baked in ---
RUN mkdir -p /opt/frost/weights \
    && curl -fSL -o /opt/frost/weights/retfoundgreen_statedict.pth \
        https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth \
    && echo "431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1  /opt/frost/weights/retfoundgreen_statedict.pth" \
        | sha256sum -c -

# --- App code: canonical pipeline (read-only), the 3 configs, and the demonstrator ---
COPY src /app/src
COPY configs/backbone/retfound_green_native392.yaml /app/configs/backbone/retfound_green_native392.yaml
COPY configs/preprocessing/retfound_green_native392.yaml /app/configs/preprocessing/retfound_green_native392.yaml
COPY configs/tasks/brset_default.yaml /app/configs/tasks/brset_default.yaml
COPY deploy/referable_dr_demo /app/deploy/referable_dr_demo

RUN chown -R user:user /app /opt/frost

# --- Runtime configuration ---
#   * discovery env vars point at the baked-in backbone + committed head + operating point
#   * FROST_ALLOW_SYNTHETIC_ONLY_READINESS=1 selects the non-credentialed server self-check
#   * conservative CPU threading keeps the free-tier RAM/CPU footprint small
ENV PYTHONPATH=/app \
    RETFOUND_GREEN_CHECKPOINT=/opt/frost/weights/retfoundgreen_statedict.pth \
    RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT=/app/deploy/referable_dr_demo/hf_space/weights/mt_head_native392.pt \
    RETINA_SCREEN_DEMO_THRESHOLD_PATH=/app/deploy/referable_dr_demo/hf_space/operating_point.json \
    FROST_ALLOW_SYNTHETIC_ONLY_READINESS=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

USER user
EXPOSE 7860

# Single FastAPI process serves BOTH the static frontend ("/") and /predict.
CMD ["uvicorn", "deploy.referable_dr_demo.hf_space.entry:app", "--host", "0.0.0.0", "--port", "7860"]
