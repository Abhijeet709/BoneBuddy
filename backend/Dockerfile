FROM python:3.11-slim

WORKDIR /app

# Install Python deps; use PyTorch CPU index for reliable Linux wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Create weights dir (will be overwritten by volume if mounted)
RUN mkdir -p weights

# Entrypoint: create dummy weights if missing, then run CMD
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default: run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
