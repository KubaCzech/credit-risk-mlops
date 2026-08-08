FROM python:3.12-slim

WORKDIR /app

# Dependencies first, separate from the rest of the code - Docker caches this layer, so
# `docker build` skips a multi-minute torch/xgboost/mlflow reinstall on every code change,
# only re-running it when requirements.txt itself changes.
COPY requirements.txt ./
# PyPI's default Linux torch wheel declares nvidia-*/triton CUDA runtime packages as hard
# dependencies unconditionally (~1.6GB) - irrelevant here, this project runs the MLP on CPU
# only by design (see README, Neural Network section). macOS never hit this: there's no
# CUDA build for Apple Silicon, so the default wheel there never declared those deps in the
# first place. --extra-index-url points at PyTorch's own CPU-only wheel index, whose torch
# build doesn't declare them, for this platform.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY pyproject.toml ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
RUN pip install --no-cache-dir . --no-deps

COPY docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
