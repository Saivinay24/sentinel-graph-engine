# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install all Python dependencies into user-local space
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# System-level build tools required by some packages (torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements file first — layer-cache friendly
COPY requirements.txt .

# Install into the user scheme so we can copy the whole .local tree later
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime: lean image, no build tools, non-root user
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="Sentinel-Graph ITDR Engine" \
      org.opencontainers.image.description="Graph-Native Identity Intelligence for Adaptive Access Control" \
      org.opencontainers.image.version="1.0.0"

# Runtime-only system dependencies (geospatial, curl for healthcheck fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source (respects .dockerignore)
COPY . .

# Ensure user-installed packages are on PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user and hand over ownership
RUN useradd --create-home --shell /bin/bash sentinel \
    && chown -R sentinel:sentinel /app /root/.local

USER sentinel

# Pre-create runtime output directory with correct permissions
RUN mkdir -p /app/data/generated /app/data/cert /app/models

# ── Health check (polls the FastAPI /health endpoint) ────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
try: \
    urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
    sys.exit(0) \
except Exception: \
    sys.exit(1)"

# Default command — override in docker-compose per service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
