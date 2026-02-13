# ---- Base stage ----
FROM python:3.11-slim AS base

LABEL maintainer="erasus-team"
LABEL description="Erasus — Universal Machine Unlearning Framework"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies stage ----
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Application stage ----
FROM deps AS app

COPY . .
RUN pip install -e .

# Verify installation
RUN python -c "import erasus; print('Erasus installed successfully')"

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]


# ---- Dev stage (with dev tools) ----
FROM app AS dev

RUN pip install pytest pytest-cov ruff mypy ipython jupyter

CMD ["bash"]


# ---- Benchmark stage (GPU-ready) ----
FROM app AS benchmark

# Install GPU dependencies if needed
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || true

CMD ["python", "benchmarks/tofu/run.py"]
