"""Publish the website and model to a Hugging Face Docker Space (free CPU tier is enough).

Uploads the Dockerfile, src/, web/, app.py and the model bundle, with a Space card; Hugging Face builds and
serves the container. Needs a Hugging Face account and a write token:
    uv run hf auth login        (or set HF_TOKEN)

Usage:
    uv run python scripts/deploy_space.py --repo <username>/skindetect
    uv run python scripts/deploy_space.py --repo <username>/skindetect --private
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from skinlesion.data import ROOT
from skinlesion.inference import DEFAULT_BUNDLE

SPACE_CARD = """---
title: SkinDetect
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: Seven-class dermoscopy lesion classifier (research demo)
---

# SkinDetect

A research demo that classifies dermoscopy images into the seven HAM10000 lesion types with a calibrated
ensemble of convolutional networks, and shows a Grad-CAM heatmap of what drove the prediction.

**Not a medical device. Not for diagnosis.** Trained on HAM10000 (Tschandl et al., 2018, CC BY-NC 4.0).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="<username>/<space-name>")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if not args.bundle.exists():
        raise SystemExit(f"{args.bundle} not found; run scripts/export_model.py first")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("Dockerfile", ".dockerignore", "app.py"):
            shutil.copy2(ROOT / name, stage / name)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(ROOT / "src", stage / "src", ignore=ignore)
        shutil.copytree(ROOT / "web", stage / "web", ignore=ignore)
        (stage / "models").mkdir()
        shutil.copy2(args.bundle, stage / "models" / "skinlesion.pt")
        (stage / "README.md").write_text(SPACE_CARD, encoding="utf-8")

        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=stage,
                          commit_message="Deploy SkinDetect", delete_patterns=["src/**", "web/**"])
    host = args.repo.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.repo}\n"
          f"App URL once built: https://{host}.hf.space")


if __name__ == "__main__":
    main()
