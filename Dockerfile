# KG schema discovery — reproducible image built from uv.lock.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Flush stdout so discovery script progress lines appear in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Install deps first (cached layer) — only deps, no project code yet.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# 2) Add the pipeline + corpus.
COPY scripts/ ./scripts/
COPY corpus/ ./corpus/
RUN mkdir -p output

# Scripts import `common` and resolve paths relative to /app, so run from
# scripts/. Outputs land in /app/output — mount a volume to retrieve them.
WORKDIR /app/scripts
ENTRYPOINT ["/app/.venv/bin/python"]
# Default: a quick 50-doc characterization smoke test. Override with any script,
# e.g.  docker run --rm -v "$PWD/output:/app/output" kg-discovery 02_discover_entities.py
CMD ["01_characterize.py", "50"]
