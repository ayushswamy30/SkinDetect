"""SkinDetect website: the static site plus a small JSON API around the model bundle.

    GET  /              the website (web/static/index.html)
    GET  /api/model     model details and validation metrics from the bundle
    POST /api/analyze   multipart form field `image` -> calibrated probabilities, malignancy score, Grad-CAM
    GET  /api/health    liveness check

Two backends, chosen by SKINDETECT_BUNDLE: a directory from scripts/export_onnx.py runs on ONNX Runtime
(no PyTorch needed, used for deployment); a .pt file from scripts/export_model.py runs on PyTorch.

Research and education only. Not a medical device.
"""

import base64
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from skinlesion.data import CLASS_NAMES, CLASSES, MALIGNANT, ROOT
from skinlesion.serving import overlay_heatmap

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 15 * 2**20
MAX_PIXELS = 40_000_000  # refuse decompression bombs before decoding

state: dict = {}
_inference_lock = threading.Lock()  # one image at a time keeps CPU and memory use predictable


def load_predictor():
    """ONNX model directory if present (or named by SKINDETECT_BUNDLE), else the PyTorch bundle."""
    tta = int(os.environ.get("SKINDETECT_TTA", "8"))
    default = ROOT / "models" / "onnx"
    if not (default / "meta.json").exists():
        default = ROOT / "models" / "skinlesion.pt"
    bundle = Path(os.environ.get("SKINDETECT_BUNDLE", default))
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    if bundle.is_dir():
        from skinlesion.serving import OnnxPredictor

        threads = os.environ.get("SKINDETECT_THREADS")
        return OnnxPredictor(bundle, tta_views=tta, threads=int(threads) if threads else None)
    import torch

    from skinlesion.inference import Predictor

    device = os.environ.get("SKINDETECT_DEVICE")  # e.g. "cpu"; default: CUDA when available
    return Predictor.from_bundle(bundle, tta_views=tta, device=torch.device(device) if device else None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = load_predictor()
    except FileNotFoundError as e:
        state["error"] = str(e)  # serve the site anyway; the analyzer shows the error
    yield
    state.clear()


app = FastAPI(title="SkinDetect", lifespan=lifespan, docs_url=None, redoc_url=None)


def predictor():
    if "predictor" not in state:
        raise HTTPException(503, state.get("error", "model not loaded"))
    return state["predictor"]


def png_data_url(array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": "predictor" in state}


@app.get("/api/model")
def model_info() -> dict:
    p = predictor()
    meta = p.metadata
    return {
        "name": meta.get("name"),
        "backbone": meta.get("model_name"),
        "ensemble_size": len(p.models),
        "tta_views": p.tta_views,
        "img_size": p.img_size,
        "temperature": p.temperature,
        "cv": meta.get("oof"),
        "test": meta.get("test"),
        "per_class": meta.get("per_class"),
        "confusion": meta.get("confusion"),
        "n_images": meta.get("n_images"),
        "n_folds": meta.get("n_folds"),
        "calibration": {k: meta.get("calibration", {}).get(k) for k in ("oof_before", "oof_after", "test_before", "test_after")},
        "classes": [{"code": c, "name": CLASS_NAMES[c], "malignant": c in MALIGNANT} for c in CLASSES],
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    p = predictor()
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 15 MB")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(415, "not a readable image (use JPEG, PNG or WebP)")

    def run():
        with _inference_lock:
            return p.analyze(img)

    pred, crop, cam = await run_in_threadpool(run)  # keep the event loop free while the model runs
    ranked = sorted(pred.probs.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "top": {"code": pred.top_class, "name": pred.top_name, "prob": pred.probs[pred.top_class],
                "malignant": pred.top_is_malignant},
        "malignant_prob": pred.malignant_prob,
        "probs": [{"code": c, "name": CLASS_NAMES[c], "prob": v, "malignant": c in MALIGNANT} for c, v in ranked],
        "input_image": png_data_url(crop),
        "heatmap": png_data_url(overlay_heatmap(crop, cam)),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
