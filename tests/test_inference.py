"""Bundle round trip, the predictor and the web API, using small untrained models (no download, CPU only)."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from skinlesion.data import CLASSES
from skinlesion.engine import build_model
from skinlesion.inference import Predictor, overlay_heatmap, save_bundle

MODEL_CFG = {"name": "resnet18", "drop_rate": 0.0, "drop_path_rate": 0.0}
DATA_CFG = {"img_size": 64, "cache_short_side": 80}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.manual_seed(0)
    models = [build_model(MODEL_CFG["name"], pretrained=False) for _ in range(2)]
    path = tmp_path_factory.mktemp("models") / "bundle.pt"
    save_bundle(path, models, MODEL_CFG, DATA_CFG, temperature=1.7,
                metadata={"name": "test", "oof": {"balanced_acc": 0.5}})
    return path


@pytest.fixture(scope="module")
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8))


def test_bundle_round_trip(bundle, image):
    p = Predictor.from_bundle(bundle, tta_views=2, device=torch.device("cpu"))
    assert len(p.models) == 2 and p.img_size == 64 and p.temperature == pytest.approx(1.7)
    assert p.metadata["model_name"] == "resnet18"
    pred = p.predict(image)
    assert set(pred.probs) == set(CLASSES)
    assert sum(pred.probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert pred.top_class == max(pred.probs, key=pred.probs.get)
    malignant = sum(pred.probs[c] for c in ("akiec", "bcc", "mel"))
    assert pred.malignant_prob == pytest.approx(malignant, abs=1e-5)


def test_preprocessing_matches_cache_scale(bundle, image):
    p = Predictor.from_bundle(bundle, tta_views=1, device=torch.device("cpu"))
    assert min(p.prepare(image).size) == 80  # downscaled like scripts/resize_images.py
    small = image.resize((60, 45))
    assert p.prepare(small).size == (60, 45)  # never upscaled
    assert p.model_input_image(image).shape == (64, 64, 3)


def test_gradcam_and_overlay(bundle, image):
    p = Predictor.from_bundle(bundle, tta_views=1, device=torch.device("cpu"))
    cam = p.gradcam(image)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    out = overlay_heatmap(p.model_input_image(image), cam)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_web_api(bundle, image, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SKINDETECT_BUNDLE", str(bundle))
    monkeypatch.setenv("SKINDETECT_TTA", "1")
    monkeypatch.setenv("SKINDETECT_DEVICE", "cpu")
    from web.server import app

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"ok": True, "model_loaded": True}
        info = client.get("/api/model").json()
        assert info["ensemble_size"] == 2 and len(info["classes"]) == len(CLASSES)
        r = client.post("/api/analyze", files={"image": ("x.jpg", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 200
        d = r.json()
        assert len(d["probs"]) == len(CLASSES)
        assert d["heatmap"].startswith("data:image/png;base64,")
        bad = client.post("/api/analyze", files={"image": ("x.txt", b"not an image", "text/plain")})
        assert bad.status_code == 415
        assert client.get("/").status_code == 200
        assert client.get("/styles.css").status_code == 200
