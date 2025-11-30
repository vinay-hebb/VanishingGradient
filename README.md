---
title: VanishingGradient
emoji: 🐢
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
short_description: Visualization for Vanishing gradients
---

# Gradient

Deep-learning playground for exploring vanishing gradients. Ships a small Flask front-end plus a Torch training loop that can export ONNX artifacts and stream runs to Weights & Biases.

## Scripts at a glance
- `trainer.py`: CLI for training synthetic classifiers and exporting ONNX checkpoints.
- `app.py`: Flask API that serves the cached artifacts under `cache/` to the front-end.
- `models.py`: Model definitions (`vanilla`, `relu`, `batchnorm`, `resnet`).
- `setup/setup.sh` / `setup/setup_conda.sh`: Install runtime dependencies; PowerShell equivalents are also present.
- `test_*.py`: Lightweight tests for the trainer, models, and Flask app.

> Caution: the `resnet` model currently approximates exact `--num-layers` you pass; it derives block count as `max(1, (num_layers - 2) // 2)` so the actual depth becomes `1 + 2*num_blocks + 1`. Small values (e.g., 3 or 5) therefore produce a slightly deeper network than requested.

## Setup
- Create an env and install training deps: `python -m venv .venv && source .venv/bin/activate && pip install -r setup/requirements_train.txt`.
- If you want to run the Flask UI too, add the app deps: `pip install -r setup/requirements.txt`.
- Optional: set `WANDB_API_KEY` and `WANDB_MODE=online` to stream metrics; offline logging is used by default.

## Training to generate artifacts
- Run `WANDB_API_KEY=<API_KEY> WANDB_MODE=online/offline python -u trainer.py` to train all four model types on a built-in synthetic dataset.
- Change the W&B output folder with `--wandb-run-dir cache/wandb_runs/`; disable logging by passing an empty string.
- Export ONNX by supplying a directory: `python trainer.py --num-layers 3 --num-epochs 2 --learning-rate 0.01 --hidden-size 8 --seed 123 --export-onnx artifacts/run1`
- Disable RNG seeding with `--seed -1` when you want nondeterministic batches.

## Run the UI locally
- Ensure `cache/` contains ONNX + `results.json` (sample artifacts are already checked in).
- Start the server with `python app.py` and open `http://localhost:5000` to explore the gradients.

## Environment variables
- `REMOTE_MODEL_BASE_URL`: (default github URL) Override the API base URL if you host the front-end separately from the Flask API. Leave unset on Hugging Face Spaces so the browser calls the same origin.
- `ENABLE_NETRON_ON_LOCAL_SERVER`: (default off) Set to `1`/`true` locally to spin up a Netron server for model visualization. It is disabled by default when `SPACE_ID` is present (Hugging Face Spaces) because 127.0.0.1 ports are not exposed; the UI will fall back to the hosted `netron.app` embed there.
