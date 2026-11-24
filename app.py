"""Run the SkinDetect website (web/static) and its model API (web/server.py).

Research and education only. This is not a medical device and must not be used for diagnosis.

Usage:
    uv sync --extra app
    uv run python app.py                    # http://127.0.0.1:8000
    uv run python app.py --host 0.0.0.0 --port 7860

Environment: SKINDETECT_BUNDLE (default models/skinlesion.pt), SKINDETECT_TTA (default 8),
SKINDETECT_DEVICE (default: CUDA if available, else CPU).
"""

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--bundle", help="model bundle path (sets SKINDETECT_BUNDLE)")
    parser.add_argument("--tta", type=int, choices=[1, 2, 4, 8], help="TTA views (sets SKINDETECT_TTA)")
    parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = parser.parse_args()
    if args.bundle:
        os.environ["SKINDETECT_BUNDLE"] = args.bundle
    if args.tta:
        os.environ["SKINDETECT_TTA"] = str(args.tta)
    uvicorn.run("web.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
