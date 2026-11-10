"""Local Flask API and browser webcam demo for the PyTorch liveness model."""

from collections import deque
import argparse
from io import BytesIO
from pathlib import Path
import secrets
import shutil
from tempfile import TemporaryDirectory
import threading
import time

import cv2
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from waitress import serve
from werkzeug.exceptions import RequestEntityTooLarge

from personal_liveness import (
    DEFAULT_MODEL, DETECTOR, EXTENSIONS, load_classifier, make_detector, score_frame, score_video,
)

MAX_JPEG_BYTES = 2 * 1024 * 1024
SESSION_TTL = 120
MAX_SESSIONS = 32


def create_app(model_path=DEFAULT_MODEL):
    model, mode = load_classifier(model_path)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    worker = threading.local()
    sessions = {}
    session_lock = threading.Lock()

    def detector():
        if not hasattr(worker, "detector"):
            worker.detector = make_detector(DETECTOR)
        return worker.detector

    @app.after_request
    def no_storage(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="Request exceeds the upload limit."), 413

    def read_frame():
        if request.mimetype != "image/jpeg":
            return None, (jsonify(error="Expected image/jpeg."), 415)
        body = request.stream.read(MAX_JPEG_BYTES + 1)
        if len(body) > MAX_JPEG_BYTES:
            return None, (jsonify(error="JPEG exceeds 2 MiB."), 413)
        try:
            with Image.open(BytesIO(body)) as image:
                if image.format != "JPEG" or not all(32 <= side <= 1920 for side in image.size):
                    raise ValueError("Unsupported image")
            frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError, cv2.error):
            frame = None
        if frame is None:
            return None, (jsonify(error="Expected a readable JPEG from 32 to 1920 pixels per side."), 422)
        return frame, None

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/v1/health")
    def health():
        return jsonify(status="ok", input_mode=mode)

    @app.post("/v1/analyze")
    def analyze():
        frame, error = read_frame()
        if error:
            return error
        return jsonify(score_frame(frame, detector(), model, mode))

    @app.post("/v1/sessions")
    def new_session():
        now = time.monotonic()
        with session_lock:
            expired = [key for key, item in sessions.items() if now - item["seen"] > SESSION_TTL]
            for key in expired:
                del sessions[key]
            if len(sessions) >= MAX_SESSIONS:
                return jsonify(error="All webcam sessions are in use."), 503
            token = secrets.token_urlsafe(18)
            sessions[token] = {"seen": now, "scores": deque(maxlen=6)}
        return jsonify(session_id=token, expires_after_seconds=SESSION_TTL), 201

    @app.delete("/v1/sessions/<token>")
    def close_session(token):
        with session_lock:
            sessions.pop(token, None)
        return "", 204

    @app.post("/v1/sessions/<token>/frames")
    def analyze_session_frame(token):
        frame, error = read_frame()
        if error:
            return error
        with session_lock:
            item = sessions.get(token)
            now = time.monotonic()
            if item is None or now - item["seen"] > SESSION_TTL:
                sessions.pop(token, None)
                return jsonify(error="Session expired or unknown."), 404
            item["seen"] = now
            result = score_frame(frame, detector(), model, mode)
            scores = item["scores"]
            if result["p_real"] is None:
                scores.clear()
            else:
                scores.append(result["p_real"])
                result["p_real"] = sum(scores) / len(scores)
                result["label"] = "real" if result["p_real"] > 0.5 else "fake"
            result["window_frames"] = len(scores)
        return jsonify(result)

    @app.post("/v1/predict")
    def predict():
        clip = request.files.get("video")
        suffix = Path(clip.filename or "").suffix.lower() if clip is not None else ""
        if clip is None or suffix not in EXTENSIONS:
            return jsonify(error="Provide a supported video in the multipart field 'video'."), 400
        with TemporaryDirectory(prefix="liveness-api-") as directory:
            path = Path(directory) / f"clip{suffix}"
            with path.open("wb") as output:
                shutil.copyfileobj(clip.stream, output)
            result = score_video(path, model, mode)
        if result["sampled_frames"] == 0:
            return jsonify(error="The uploaded video could not be read."), 422
        return jsonify(result)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("Port must be between 1 and 65535.")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    app = create_app(args.model)
    print(f"Face Liveness ready at http://127.0.0.1:{args.port}", flush=True)
    serve(app, host="127.0.0.1", port=args.port, threads=4,
          max_request_body_size=app.config["MAX_CONTENT_LENGTH"])


if __name__ == "__main__":
    main()
