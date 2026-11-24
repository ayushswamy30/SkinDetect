# SkinDetect website + model API on CPU with ONNX Runtime (no PyTorch in the image).
# Build the ONNX models first (see README):
#   uv run python scripts/export_model.py --name baseline && uv run python scripts/export_onnx.py
#   docker build -t skindetect . && docker run -p 7860:7860 skindetect
# Also works as a Hugging Face Docker Space (port 7860, non-root user 1000).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    SKINDETECT_BUNDLE=/app/models/onnx \
    SKINDETECT_TTA=8 \
    SKINDETECT_THREADS=2

RUN pip install "onnxruntime>=1.20" numpy pillow "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app src/ src/
COPY --chown=app web/ web/
COPY --chown=app app.py .
COPY --chown=app models/onnx/ models/onnx/
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
