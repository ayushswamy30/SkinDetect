"""Build the website's images and background videos from HAM10000 dermoscopy images (needs ffmpeg on PATH).

Writes to web/static/assets/:
  lesions/<class>.jpg      one example per class for the lesion-type cards
  strip/<n>.jpg            square crops for the scrolling image strip
  video/hero.mp4 (+ .jpg)  slow zooms and crossfades through eight lesions, for the hero background
  video/macro.mp4 (+ .jpg) a slow push-in on two lesions, for the closing call-to-action band

HAM10000 images: Tschandl, Rosendahl & Kittler, Sci. Data 5, 180161 (2018), CC BY-NC 4.0.
The examples were picked from images the baseline classified correctly with high confidence.

Usage:
    uv run python scripts/build_web_assets.py
"""

import shutil
import subprocess
from pathlib import Path

from PIL import Image

from skinlesion.data import ROOT, TRAIN_IMAGE_DIR

ASSETS = ROOT / "web" / "static" / "assets"

CLASS_EXAMPLES = {
    "akiec": "ISIC_0031874", "bcc": "ISIC_0027631", "bkl": "ISIC_0024383", "df": "ISIC_0032468",
    "mel": "ISIC_0033848", "nv": "ISIC_0032346", "vasc": "ISIC_0024662",
}
HERO_SEQUENCE = ["ISIC_0033848", "ISIC_0025879", "ISIC_0024662", "ISIC_0033426", "ISIC_0026758",
                 "ISIC_0027631", "ISIC_0031874", "ISIC_0032774"]
MACRO_SEQUENCE = ["ISIC_0033569", "ISIC_0024967"]
STRIP = ["ISIC_0033426", "ISIC_0024475", "ISIC_0025879", "ISIC_0032393", "ISIC_0029391", "ISIC_0027613",
         "ISIC_0034048", "ISIC_0033855"]


def source(image_id: str) -> Path:
    return TRAIN_IMAGE_DIR / f"{image_id}.jpg"


def save_jpeg(image: Image.Image, path: Path, quality: int = 82) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True, progressive=True)


def square(image: Image.Image, size: int) -> Image.Image:
    s = min(image.size)
    left, top = (image.width - s) // 2, (image.height - s) // 2
    return image.crop((left, top, left + s, top + s)).resize((size, size), Image.Resampling.LANCZOS)


def zoom_video(ffmpeg: str, images: list[str], out: Path, seconds: float, fade: float, zoom_to: float,
               width: int = 1280, height: int = 720, fps: int = 30) -> None:
    """A slow zoom on each image, joined by crossfades; muted H.264 with faststart for web playback."""
    frames = int(seconds * fps)
    inputs, filters = [], []
    for i, image_id in enumerate(images):
        # A single still frame in: zoompan turns each input frame into `d` output frames
        inputs += ["-i", str(source(image_id))]
        # Upscale first so zoompan's integer crop steps don't make the motion jitter
        direction = "+" if i % 2 == 0 else "-"
        start = 1.0 if direction == "+" else zoom_to
        filters.append(
            f"[{i}:v]scale={width * 3}:-2,crop={width * 3}:{height * 3},"
            f"zoompan=z='{start}{direction}{(zoom_to - 1) / frames:.6f}*on':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps},setsar=1,format=yuv420p[v{i}]")
    last, offset = "v0", seconds - fade
    for i in range(1, len(images)):
        filters.append(f"[{last}][v{i}]xfade=transition=fade:duration={fade}:offset={offset:.3f}[x{i}]")
        last, offset = f"x{i}", offset + seconds - fade
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
                    "-map", f"[{last}]", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-maxrate", "1200k", "-bufsize", "2400k",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(out), "-frames:v", "1", "-q:v", "4",
                    str(out.with_suffix(".jpg"))], check=True)


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")

    for cls, image_id in CLASS_EXAMPLES.items():
        with Image.open(source(image_id)) as im:
            save_jpeg(im.convert("RGB"), ASSETS / "lesions" / f"{cls}.jpg")
    for n, image_id in enumerate(STRIP, 1):
        with Image.open(source(image_id)) as im:
            save_jpeg(square(im.convert("RGB"), 420), ASSETS / "strip" / f"{n}.jpg")

    zoom_video(ffmpeg, HERO_SEQUENCE, ASSETS / "video" / "hero.mp4", seconds=4.5, fade=1.2, zoom_to=1.18)
    zoom_video(ffmpeg, MACRO_SEQUENCE, ASSETS / "video" / "macro.mp4", seconds=7, fade=2, zoom_to=1.35)

    for path in sorted(ASSETS.rglob("*.*")):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
