# Deploy to Hugging Face Spaces

Space: [beardmoose/CareSignal](https://huggingface.co/spaces/beardmoose/CareSignal)

This deploys the **full FastAPI app** including the demo UI at `/`, static assets at `/assets`, and API at `/predict` and `/docs`.

## Requirements

- [Hugging Face account](https://huggingface.co/join)
- [Write token](https://huggingface.co/settings/tokens) (for git push)
- Git installed locally
- Trained model optional locally; **do not git-push `model.joblib`** to HF (binaries are rejected — the Docker build trains on `data/reference` instead). For your full cohort model on HF, use [Xet storage](https://huggingface.co/docs/hub/xet).

## Option A — Automated sync (Windows)

```powershell
cd c:\Users\HP\Desktop\caresignal
.\.venv\Scripts\Activate.ps1

# One-time: log in to Hugging Face CLI (optional) or use git with your token as password
# hf auth login

.\scripts\push_hf_space.ps1
```

## Option B — Manual git push

### 1. Clone the Space (once)

```bash
git clone https://huggingface.co/spaces/beardmoose/CareSignal
cd CareSignal
```

When prompted for a password, use a Hugging Face **write token**.

### 2. Copy application files from CareSignal

Copy these from your project into the Space folder (replace Space files):

| Path | Purpose |
|------|---------|
| `Dockerfile` | Build & run on port **7860** |
| `.dockerignore` | Smaller, faster builds |
| `pyproject.toml` | Dependencies |
| `src/` | App + **static UI** |
| `config/` | Training config |
| `scripts/` | Fallback train in Docker if no model |
| `data/reference/` | CI-sized FHIR fallback |
| `artifacts/model.joblib.b64` | Text-encoded model (HF blocks binary `.joblib`) |
| `artifacts/manifest.json` | Threshold & metrics |
| `deploy/huggingface/README.md` → `README.md` | Space card metadata |

### 3. Commit and push

```bash
git add Dockerfile .dockerignore pyproject.toml src config scripts data/reference artifacts README.md
git commit -m "Deploy CareSignal with UI and model"
git push
```

### 4. Open the Space

After the build (~2–10 min on free CPU):

- **UI:** https://huggingface.co/spaces/beardmoose/CareSignal  
- **API docs:** https://huggingface.co/spaces/beardmoose/CareSignal/docs  

## How it works

- Docker Space must listen on port **7860** ([HF docs](https://huggingface.co/docs/hub/spaces-sdks-docker)).
- `Dockerfile` sets `PORT=7860` and starts `uvicorn` with your FastAPI factory.
- The UI uses same-origin `fetch('/predict')` — no extra CORS config.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails: no model | Ensure `artifacts/model.joblib` is committed or allow Docker to train on `data/reference` |
| "No application file" | Push `Dockerfile` to the Space repo root |
| UI 404 | Confirm `src/caresignal/static/index.html` was copied |
| Cold / slow first load | Normal on free CPU Spaces |

## Updating after local changes

Regenerate the text-encoded model and push:

```bash
python scripts/run_all.py --fhir-dir data/reference --config config/train.docker.yaml
python scripts/encode_model_b64.py
.\scripts\push_hf_space.ps1
```
