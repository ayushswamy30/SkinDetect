"""Cache downscaled copies of the images so the data loader is not limited by JPEG decoding.

The originals are 600x450. Training at 224px only needs a short side of ~300, and a smaller JPEG
decodes and rotates more than twice as fast. Copies go to data/processed/<images|test_images>_<short side>/.

Usage:
    uv run python scripts/resize_images.py               # short side 300 (for training at 224px)
    uv run python scripts/resize_images.py --short-side 450
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from skinlesion.data import TEST_IMAGE_DIR, TRAIN_IMAGE_DIR, resized_image_dir


def resize_one(args: tuple[Path, Path, int]) -> None:
    src, dst, short_side = args
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = short_side / min(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        im.save(dst, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--short-side", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for src_dir in (TRAIN_IMAGE_DIR, TEST_IMAGE_DIR):
        if not src_dir.exists():
            print(f"skip {src_dir.name}: not downloaded")
            continue
        dst_dir = resized_image_dir(src_dir, args.short_side)
        dst_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(p, dst_dir / p.name, args.short_side) for p in sorted(src_dir.glob("*.jpg"))
                if not (dst_dir / p.name).exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            list(tqdm(pool.map(resize_one, jobs, chunksize=32), total=len(jobs), desc=dst_dir.name, unit="img"))
        print(f"{dst_dir.name}: {len(list(dst_dir.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
