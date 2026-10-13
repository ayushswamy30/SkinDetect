"""Download HAM10000 and the ISIC 2018 Task 3 test set from Harvard Dataverse.

Source: Tschandl et al., "The HAM10000 dataset", doi:10.7910/DVN/DBW86T (CC BY-NC 4.0).

Usage:
    uv run python scripts/download_data.py            # everything
    uv run python scripts/download_data.py --no-test  # skip the ISIC 2018 test set
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

DATASET_DOI = "doi:10.7910/DVN/DBW86T"
API = "https://dataverse.harvard.edu/api"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
# Dataverse rejects urllib's default User-Agent with a 403
HEADERS = {"User-Agent": "skinlesion-downloader/0.1 (research)"}

# Dataverse filename -> (local filename, extract into this subdirectory or None)
TRAIN_FILES = {
    "HAM10000_metadata.tab": ("HAM10000_metadata.csv", None),
    "HAM10000_images_part_1.zip": ("HAM10000_images_part_1.zip", "images"),
    "HAM10000_images_part_2.zip": ("HAM10000_images_part_2.zip", "images"),
    "HAM10000_segmentations_lesion_tschandl.zip": ("HAM10000_segmentations.zip", "masks"),
}
TEST_FILES = {
    "ISIC2018_Task3_Test_GroundTruth.tab": ("ISIC2018_Task3_Test_GroundTruth.csv", None),
    "ISIC2018_Task3_Test_Images.zip": ("ISIC2018_Task3_Test_Images.zip", "test_images"),
}


def list_remote_files() -> dict[str, dict]:
    url = f"{API}/datasets/:persistentId/versions/:latest/files?persistentId={DATASET_DOI}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as resp:
        entries = json.load(resp)["data"]
    return {e["dataFile"]["filename"]: e["dataFile"] for e in entries}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(file_id: int, dest: Path, expected_md5: str | None, is_tabular: bool) -> None:
    # Dataverse converts tabular uploads to .tab, so ask for the original CSV.
    # Those are small and not checksum-verified; the image zips are.
    url = f"{API}/access/datafile/{file_id}" + ("?format=original" if is_tabular else "")
    if dest.exists() and expected_md5 and md5sum(dest) == expected_md5:
        print(f"  ok (cached)  {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {dest.name}") as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))

    if expected_md5 and not is_tabular and md5sum(tmp) != expected_md5:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {dest.name}; deleted, please re-run.")
    tmp.replace(dest)


def extract(zip_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
        for m in tqdm(members, desc=f"  extract {zip_path.name}", unit="file"):
            # Flatten: every file lands directly in target/, which also blocks path traversal
            name = Path(m.filename).name
            if not name or name.startswith("."):
                continue
            with zf.open(m) as src, open(target / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-test", action="store_true", help="skip the ISIC 2018 Task 3 test set")
    parser.add_argument("--keep-zips", action="store_true", help="keep zip files after extracting")
    args = parser.parse_args()

    wanted = dict(TRAIN_FILES)
    if not args.no_test:
        wanted.update(TEST_FILES)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    remote = list_remote_files()

    for remote_name, (local_name, extract_to) in wanted.items():
        info = remote[remote_name]
        dest = RAW_DIR / local_name
        is_tabular = remote_name.endswith(".tab")
        md5 = info.get("md5") or (info.get("checksum") or {}).get("value")

        if extract_to and (RAW_DIR / extract_to / ".done_" / local_name).exists():
            print(f"  ok (extracted)  {local_name}")
            continue

        download(info["id"], dest, md5, is_tabular)
        if extract_to:
            extract(dest, RAW_DIR / extract_to)
            marker = RAW_DIR / extract_to / ".done_" / local_name
            marker.parent.mkdir(exist_ok=True)
            marker.touch()
            if not args.keep_zips:
                dest.unlink()

    n_imgs = len(list((RAW_DIR / "images").glob("*.jpg")))
    print(f"\nHAM10000 images: {n_imgs} (expected 10015)")
    if not args.no_test:
        n_test = len(list((RAW_DIR / "test_images").glob("*.jpg")))
        # The ground truth lists 1512 images but the Dataverse zip omits ISIC_0035068
        print(f"ISIC 2018 test images: {n_test} (expected 1511)")


if __name__ == "__main__":
    main()
