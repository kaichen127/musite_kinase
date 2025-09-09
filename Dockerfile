FROM python:3.10-slim

# (Optional) set non-root user
# RUN useradd -ms /bin/bash appuser
# USER appuser

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py ./inference.py
COPY model_state.pth ./model_state.pth

# Optional: warm the ESMC backbone/tokenizer cache at build time
# (speeds up first inference; requires internet during `docker build`)
# RUN python - <<'PY'
# from esm.models.esmc import ESMC
# ESMC.from_pretrained("esmc_600m")
# print("ESMC cached")
# PY

ENV MODEL_PATH=/app/model_state.pth

# Default command
ENTRYPOINT ["python", "inference.py"]
