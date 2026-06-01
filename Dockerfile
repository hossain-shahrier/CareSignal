FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY config ./config
COPY data/reference ./data/reference
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir .

RUN python scripts/run_all.py --fhir-dir data/reference --config config/train.yaml

EXPOSE 8000

CMD ["uvicorn", "caresignal.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
