"""Paths, class definitions and metadata loading shared across the project."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_IMAGE_DIR = RAW_DIR / "images"
TEST_IMAGE_DIR = RAW_DIR / "test_images"
MASK_DIR = RAW_DIR / "masks"

FOLDS_CSV = PROCESSED_DIR / "ham10000_folds.csv"
TEST_CSV = PROCESSED_DIR / "isic2018_test.csv"

# Alphabetical order of the HAM10000 codes; the index is the model's class id.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
MALIGNANT = {"akiec", "bcc", "mel"}


def load_metadata() -> pd.DataFrame:
    """HAM10000 training metadata with image paths and integer labels."""
    df = pd.read_csv(RAW_DIR / "HAM10000_metadata.csv")
    df["path"] = df.image_id.map(lambda i: str(TRAIN_IMAGE_DIR / f"{i}.jpg"))
    df["label"] = df.dx.map(CLASS_TO_IDX)
    return df


def load_test_metadata() -> pd.DataFrame:
    """ISIC 2018 Task 3 test ground truth, restricted to images that exist on disk."""
    df = pd.read_csv(RAW_DIR / "ISIC2018_Task3_Test_GroundTruth.csv")
    df["path"] = df.image_id.map(lambda i: str(TEST_IMAGE_DIR / f"{i}.jpg"))
    df["label"] = df.dx.map(CLASS_TO_IDX)
    return df[df.path.map(lambda p: Path(p).exists())].reset_index(drop=True)
