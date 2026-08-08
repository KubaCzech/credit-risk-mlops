#!/bin/sh
set -e

# Idempotent by design: alembic_version tracks what's applied (safe to re-run),
# seed_models.py upserts by name (safe to re-run) - so every container start
# self-heals the schema and reference data, no manual migration step required.
alembic upgrade head
python -m db.seed_models

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
