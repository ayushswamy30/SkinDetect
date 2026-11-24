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
**Final model:** five EfficientNet-B0 networks (one per fold, 224px), averaged over 8 rotations/flips.

| Evaluation | Images | Balanced acc. | Accuracy | Macro F1 | Macro AUC | Malignant sens. / spec. |
|---|---|---|---|---|---|---|
| 5-fold cross-validation (out-of-fold, single network per image) | 10,015 | 0.755 | 0.831 | 0.747 | 0.970 | 0.768 / 0.904 |
| **ISIC 2018 test set** (held out, 5-network ensemble) | 1,511 | **0.752** | **0.839** | **0.773** | **0.972** | **0.818 / 0.913** |

Cross-validation is split by lesion, so no lesion appears in both training and validation. The test set
was used once, for this final evaluation. Test recall by class: nevus 0.90, melanoma 0.78, vascular 0.77,
basal cell carcinoma 0.76, benign keratosis 0.76, dermatofibroma 0.75, actinic keratosis 0.53 (most often
mistaken for benign keratosis).

**Experiments** (fold 0 only, so differences under ~0.02 are within noise):

| Variant | Balanced acc. | Macro AUC | Melanoma recall | Verdict |
|---|---|---|---|---|
| Baseline: EfficientNet-B0, 224px | 0.734 | 0.971 | 0.72 | kept |
| B0 at 288px | 0.739 | 0.963 | 0.55 | melanoma recall drops sharply |
| EfficientNet-B2 at 256px | 0.727 | 0.972 | 0.66 | no gain, ~2.5x the compute |

A run with full inverse-frequency class weights did not complete (the training machine ran out of memory).

**Calibration.** A single network is over-confident: its expected calibration error (ECE) on out-of-fold
predictions is 7.1%. Temperature scaling fitted on those predictions (T = 1.89) brings it to 1.2%, but the
deployed model averages five networks, which already removes most of the over-confidence: on the test set the
raw ensemble's ECE is 3.0%, and applying T = 1.89 on top raises it to 8.9%. The deployed ensemble therefore
uses T = 1. Note that this choice was made after seeing the test set, so the 3.0% is not a fully unbiased
estimate; accuracy and AUC are unaffected, since temperature changes neither the predicted class nor rankings.
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
4. **Calibration:** single networks are over-confident; averaging the five fold models corrects most of
   it (see Results). `scripts/evaluate.py` also fits a temperature for single models.
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
