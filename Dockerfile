# CareSignal — Docker (Hugging Face Spaces, Render, local)
# HF Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker (port 7860)

FROM python:3.11-slim

# LightGBM needs OpenMP at runtime on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY data/reference ./data/reference
COPY artifacts ./artifacts

RUN pip install --no-cache-dir .

# Prefer pre-encoded model (HF git); else train a small demo bundle
RUN python scripts/decode_model_b64.py \
    || python scripts/run_all.py --fhir-dir data/reference --config config/train.docker.yaml

EXPOSE 7860

CMD ["sh", "-c", "uvicorn caresignal.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
