# SkinDetect website + model API on CPU.
# Build the model bundle first (models/skinlesion.pt, see README), then:
#   docker build -t skindetect .
#   docker run -p 7860:7860 skindetect
# Also works as a Hugging Face Docker Space (port 7860, non-root user 1000).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    SKINDETECT_BUNDLE=/app/models/skinlesion.pt \
    SKINDETECT_TTA=8 \
    OMP_NUM_THREADS=2

# CPU-only PyTorch keeps the image ~1 GB instead of ~6 GB with CUDA
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.*" "torchvision==0.26.*" \
 && pip install "timm>=1.0" numpy pandas scikit-learn scipy matplotlib pillow tqdm pyyaml \
                "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app src/ src/
COPY --chown=app web/ web/
COPY --chown=app app.py .
COPY --chown=app models/skinlesion.pt models/skinlesion.pt
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
