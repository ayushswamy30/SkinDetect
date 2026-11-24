# SkinDetect

Seven-class skin lesion classification from dermoscopy images, trained on HAM10000, with calibrated
probabilities, Grad-CAM explanations and a web app.

**Live demo: https://skindetect-d42c.onrender.com** (free tier: the first visit after a quiet spell takes
about a minute to wake up).

> **Research and education only. Not a medical device.** The model has not been clinically validated,
> learned from dermatoscope images of mostly fair-skinned patients, and can be confidently wrong.
> Anyone worried about a skin lesion should see a dermatologist.

## What it does

Given a dermoscopy image, SkinDetect predicts one of the seven HAM10000 diagnoses:

| Code | Lesion | Malignant | Share of HAM10000 |
|---|---|---|---|
| `nv` | Melanocytic nevus | no | 66.9% |
| `mel` | Melanoma | **yes** | 11.1% |
| `bkl` | Benign keratosis-like lesion | no | 11.0% |
| `bcc` | Basal cell carcinoma | **yes** | 5.1% |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | **yes** | 3.3% |
| `vasc` | Vascular lesion | no | 1.4% |
| `df` | Dermatofibroma | no | 1.1% |

It reports calibrated class probabilities, a combined malignancy probability, and a heatmap of the regions
that drove the prediction.

## Results

<!-- RESULTS:START -->
Five-fold cross-validation, split by lesion so that no lesion appears in both training and validation.
Out-of-fold numbers pool the validation predictions of all folds (8-view test-time augmentation).

| Model | Balanced acc. | Accuracy | Macro F1 | Macro AUC | Malignant sens. / spec. |
|---|---|---|---|---|---|
| Baseline, EfficientNet-B0 224px (folds 0-2) | 0.753 | 0.830 | 0.745 | 0.972 | 0.763 / 0.902 |

Final model and ISIC 2018 test-set results are added when training finishes.
<!-- RESULTS:END -->

Balanced accuracy (mean recall over the seven classes) is the ISIC 2018 Task 3 ranking metric. Malignant
sensitivity counts a malignant lesion as caught when *any* malignant class is predicted, which is what
matters for a referral decision.

## How it works

1. **Data:** HAM10000 (10,015 images, 7,470 lesions). Folds are grouped by lesion and stratified by
   diagnosis, because many lesions were photographed more than once (`data/processed/ham10000_folds.csv`).
   The ISIC 2018 Task 3 test set (1,511 images) is held out for one final evaluation.
2. **Training:** ImageNet-pretrained timm backbones, rotation/flip/crop/colour augmentation (a lesion has
   no canonical orientation), class-weighted cross-entropy for the 60:1 imbalance, AdamW with warm-up
   and cosine decay, bf16 autocast. The checkpoint with the best validation balanced accuracy is kept.
3. **Ensemble and TTA:** the five fold models are averaged, each over eight rotations and mirrorings of
   the image.
4. **Calibration:** the networks are over-confident; one temperature fitted on the out-of-fold
   predictions fixes that (expected calibration error 6.8% → 1.1% for the baseline).
5. **Explanation:** Grad-CAM on the last convolutional feature map. For EfficientNet's average-pool +
   linear head it equals CAM exactly, so the ONNX export computes the maps inside the graph.
6. **Serving:** ONNX Runtime behind FastAPI, no PyTorch needed; the website is plain HTML/CSS/JS.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The training dependencies install the CUDA
build of PyTorch (see `pyproject.toml`).

```bash
uv sync --extra app                                  # everything, including the web server
uv run python scripts/download_data.py               # HAM10000 + ISIC 2018 test set (several GB)
uv run python scripts/make_splits.py                 # lesion-grouped folds (already committed)
uv run python scripts/resize_images.py               # cached 300px copies for fast loading
```

### Train and evaluate

```bash
uv run python scripts/run_cv.py --config configs/baseline.yaml          # all 5 folds (~25-35 min each on an RTX 3050)
uv run python scripts/evaluate.py --name baseline --tta 8               # CV report, confusion matrix, calibration
uv run python scripts/evaluate.py --name baseline --tta 8 --test        # + held-out test set (final models only)
```

Any config value can be overridden, e.g. `--set name=b0_288 data.img_size=288`. Reports go to
`outputs/eval/<name>/`.

### Predict and explain

```bash
uv run python scripts/export_model.py --name baseline     # fold ensemble -> models/skinlesion.pt
uv run python scripts/predict.py lesion.jpg --cam outputs/cam
```

### Run the website

```bash
uv run python scripts/export_onnx.py                      # models/onnx/ (PyTorch-free serving)
uv run python app.py                                      # http://127.0.0.1:8000
```

### Deploy

The live site runs on Render's free tier from the `deploy` branch, which holds only the server code,
the site and the ONNX models:

```bash
uv run python scripts/publish_deploy.py                   # rebuild + push the deploy branch; Render redeploys
```

A CPU `Dockerfile` is included for other hosts (`docker build -t skindetect . && docker run -p 7860:7860 skindetect`).

### Tests

```bash
uv run pytest            # 14 tests, CPU only, no dataset needed
```

## Project layout

```
configs/                 training configs (YAML)
scripts/                 data download/splits, training, CV, evaluation, export, deploy
src/skinlesion/
  data.py                paths and class definitions
  dataset.py             PyTorch dataset, augmentations, class balancing
  engine.py              model factory, training loop, prediction
  metrics.py             balanced accuracy, AUC, per-class recall
  evaluation.py          run discovery, TTA, calibration, reports
  inference.py           PyTorch model bundle, ensemble predictor, Grad-CAM
  serving.py             ONNX Runtime predictor (no PyTorch)
web/                     FastAPI server and the static website
tests/                   unit and API tests
```

## Data and credits

- HAM10000: Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
  [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), CC BY-NC 4.0. The website's lesion
  images and background videos are made from HAM10000 images (`scripts/build_web_assets.py`).
- ISIC 2018 Task 3 test set: Codella et al., arXiv:1902.03368; Tschandl et al. (2018).
- Dermatoscope photo: Abessemans94, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:IDS-1100_Dermatoscope.jpg), CC BY-SA 4.0 (cropped).

Because the training data is CC BY-NC 4.0, the trained models and this demo are for non-commercial use.
