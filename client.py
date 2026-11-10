"""Send one image or video to the local Face Liveness API."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="JPEG file.")
    source.add_argument("--video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    try:
        with (args.image or args.video).open("rb") as stream:
            if args.image:
                response = requests.post(
                    f"{url}/v1/analyze", data=stream, headers={"Content-Type": "image/jpeg"}, timeout=30,
                )
            else:
                response = requests.post(
                    f"{url}/v1/predict", files={"video": (args.video.name, stream)}, timeout=60,
                )
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
