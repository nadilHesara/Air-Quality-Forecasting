# Multi-stage build for the PM2.5 forecasting service.
#
# Stage 1 ("builder") installs the pinned dependencies into a wheel cache;
# stage 2 copies only what's needed into a slim runtime image.  Both the API
# (uvicorn) and the Streamlit dashboard run from this same image — the
# docker-compose file just starts it twice with different commands.

# ── Stage 1: build wheels from the pinned lock ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Build LightGBM's runtime dep (libgomp) is added in the runtime stage; here we
# just resolve and download the pinned wheels.
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.lock

# ── Stage 2: slim runtime ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp1 is LightGBM's only shared-library runtime dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.lock .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.lock \
    && rm -rf /wheels

# Copy the application code and the committed model.
COPY config.py .
COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user.  config.py creates data/reports/models dirs on
# import, so the app directory must be writable by that user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# API port (Streamlit uses 8501; both are published in docker-compose).
EXPOSE 8000

# Default command runs the API; docker-compose overrides it for the dashboard.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
