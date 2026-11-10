"""Train and run a compact liveness CNN on locally recorded videos."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LABELS = ("fake", "real")
SPLITS = ("train", "validation", "test")
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DETECTOR = Path(__file__).resolve().parent / "face_detector/face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL = Path("artifacts/personal/liveness_rgb.pt")


class LivenessCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(192, 1))

    def forward(self, inputs):
        return self.classifier(self.features(inputs)).squeeze(1)


def make_detector(path):
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.62, 0.3, 5000)


def single_face(frame, detector):
    height, width = frame.shape[:2]
    ratio = min(1, 640 / max(height, width))
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    detector.setInputSize(size)
    _, boxes = detector.detect(small)
    if boxes is None or len(boxes) != 1:
        return None
    x, y, w, h = boxes[0, :4]
    left = max(0, math.floor(x * width / size[0]))
    top = max(0, math.floor(y * height / size[1]))
    right = min(width, math.ceil((x + w) * width / size[0]))
    bottom = min(height, math.ceil((y + h) * height / size[1]))
    if right <= left or bottom <= top:
        return None
    face = cv2.resize(frame[top:bottom, left:right], (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (left, top, right, bottom)


def sample_faces(video, detector, sample_fps, seconds):
    """Take single-face samples at regular times; return RGB crops and attempt count."""
    cap = cv2.VideoCapture(str(video))
    faces = []
    attempts = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            return faces, attempts
        step = max(1, round(fps / sample_fps))
        for frame_index in range(math.ceil(seconds * fps)):
            if not cap.grab():
                break
            if frame_index % step:
                continue
            attempts += 1
            ok, frame = cap.retrieve()
            if not ok:
                continue
            sample = single_face(frame, detector)
            if sample is not None:
                faces.append(sample[0])
    finally:
        cap.release()
    return faces, attempts


def prepare(args):
    detector = make_detector(args.detector)
    rng = random.Random(args.seed)
    images, clip_ids, clip_labels, clip_splits, attempts = [], [], [], [], []
    for label, name in enumerate(LABELS):
        files = sorted(
            p for p in (args.data / f"Video {name}").rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if len(files) < 7:
            raise ValueError(f"At least 7 {name} videos are required for a three-way split.")
        rng.shuffle(files)
        holdout = max(1, round(len(files) * 0.15))
        split_tags = [0] * (len(files) - 2 * holdout) + [1] * holdout + [2] * holdout
        for number, (path, split) in enumerate(zip(files, split_tags), 1):
            faces, attempted = sample_faces(path, detector, args.sample_fps, args.seconds)
            clip_ids.extend([len(clip_labels)] * len(faces))
            images.extend(faces)
            clip_labels.append(label)
            clip_splits.append(split)
            attempts.append(attempted)
            print(f"{name}: {number}/{len(files)} clips prepared", flush=True)
    if not images:
        raise ValueError("No eligible single-face frames found.")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache, images=np.stack(images), clip_ids=np.asarray(clip_ids),
        labels=np.asarray(clip_labels), splits=np.asarray(clip_splits), attempts=np.asarray(attempts),
        seed=args.seed, sample_fps=args.sample_fps, seconds=args.seconds,
    )


def tensor_images(images, mode):
    tensor = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float() / 255
    if mode == "gray":
        weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (tensor * weights).sum(dim=1, keepdim=True)
    return tensor


def split_metrics(model, inputs, cache, split):
    clip_ids, labels, splits = (cache[name] for name in ("clip_ids", "labels", "splits"))
    selected = splits[clip_ids] == split
    ids = clip_ids[selected]
    if set(labels[ids]) != {0, 1}:
        raise ValueError(f"The {SPLITS[split]} split needs an eligible clip from both classes.")
    with torch.inference_mode():
        logits = torch.cat([model(batch) for batch in inputs[selected].split(128)])
        probs = logits.sigmoid().numpy()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(labels[ids]).float(),
        ).item()
    matrix = [[0, 0], [0, 0]]
    for clip in np.unique(ids):
        prediction = int(probs[ids == clip].mean() > 0.5)
        matrix[int(labels[clip])][prediction] += 1
    fake, real = (sum(row) for row in matrix)
    apcer, bpcer = matrix[0][1] / fake, matrix[1][0] / real
    return {
        "assigned_clips": int((splits == split).sum()), "scored_clips": fake + real,
        "scored_frames": len(ids), "frame_bce": loss, "confusion_matrix_fake_real": matrix,
        "pooled_apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2,
    }


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    clip_ids, labels, splits = (cache[name] for name in ("clip_ids", "labels", "splits"))
    for split in range(len(SPLITS)):
        if set(labels[clip_ids[splits[clip_ids] == split]]) != {0, 1}:
            raise ValueError(f"The {SPLITS[split]} split needs usable clips from both labels.")
    selected_train = splits[clip_ids] == 0
    train_ids = clip_ids[selected_train]
    train_labels = torch.from_numpy(labels[train_ids]).float()
    # Weight videos equally so clips with more detected frames do not dominate training.
    counts = np.bincount(train_ids, minlength=len(labels))
    frame_weights = torch.from_numpy(1 / counts[train_ids]).float()
    frame_weights /= frame_weights.mean()
    results = {}
    trained = {}
    for mode, channels in (("rgb", 3), ("gray", 1)):
        torch.manual_seed(args.seed)
        inputs = tensor_images(cache["images"], mode)
        model = LivenessCNN(channels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loader = DataLoader(
            TensorDataset(inputs[selected_train], train_labels, frame_weights),
            batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        )
        best = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y, weights in loader:
                optimizer.zero_grad()
                loss = (nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none") * weights).mean()
                loss.backward()
                optimizer.step()
            model.eval()
            validation = split_metrics(model, inputs, cache, 1)
            key = (validation["acer"], validation["frame_bce"])
            if best is None or key < best[0]:
                best = (key, epoch, copy.deepcopy(model.state_dict()), validation)
            print(f"{mode}: epoch {epoch:02d} validation clip ACER={validation['acer']:.4f}", flush=True)
        _, epoch, state, validation = best
        model.load_state_dict(state)
        results[mode] = {"best_epoch": epoch, "validation": validation}
        trained[mode] = (model, inputs)
        torch.save({"mode": mode, "model": state}, args.output_dir / f"liveness_{mode}.pt")
    winner = min(results, key=lambda mode: (results[mode]["validation"]["acer"], results[mode]["validation"]["frame_bce"]))
    for mode, (model, inputs) in trained.items():
        results[mode]["test"] = split_metrics(model, inputs, cache, 2)
    report = {
        "protocol": "stratified by label, split by video before cropping; subject separation not verified",
        "dataset_seed": int(cache["seed"]), "train_seed": args.seed, "epochs": args.epochs,
        "sample_fps_target": float(cache["sample_fps"]), "seconds_per_clip": float(cache["seconds"]),
        "real_threshold_exclusive": 0.5, "videos_by_label": dict(zip(LABELS, np.bincount(labels).tolist())),
        "sampled_frames": int(cache["attempts"].sum()), "scored_frames": len(clip_ids),
        "split_clips": {name: dict(zip(LABELS, np.bincount(labels[splits == i], minlength=2).tolist()))
                        for i, name in enumerate(SPLITS)},
        "selected_on_validation": winner, "models": results,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def load_classifier(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    mode = checkpoint["mode"]
    model = LivenessCNN(1 if mode == "gray" else 3)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, mode


@torch.inference_mode()
def frame_probabilities(model, mode, faces):
    return model(tensor_images(faces, mode)).sigmoid()


def score_frame(frame, detector, model, mode):
    sample = single_face(frame, detector)
    if sample is None:
        return {"label": None, "p_real": None, "box": None}
    face, box = sample
    probability = float(frame_probabilities(model, mode, face[None])[0])
    return {"label": "real" if probability > 0.5 else "fake", "p_real": probability, "box": box}


def score_video(video, model, mode, detector_path=DETECTOR, sample_fps=2, seconds=12):
    faces, attempts = sample_faces(video, make_detector(detector_path), sample_fps, seconds)
    result = {"sampled_frames": attempts, "scored_frames": len(faces), "label": None, "p_real": None}
    if faces:
        probability = float(frame_probabilities(model, mode, np.stack(faces)).mean())
        result.update(label="real" if probability > 0.5 else "fake", p_real=probability)
    return result


def predict(args):
    model, mode = load_classifier(args.model)
    result = score_video(args.video, model, mode, args.detector, args.sample_fps, args.seconds)
    print(json.dumps(result, indent=2))


def evaluate(args):
    model, mode = load_classifier(args.model)
    with np.load(args.cache, allow_pickle=False) as cache:
        result = split_metrics(model, tensor_images(cache["images"], mode), cache, SPLITS.index(args.split))
    print(json.dumps({"input_mode": mode, "split": args.split, **result}, indent=2))


def export(args):
    model, mode = load_classifier(args.model)
    example = torch.zeros((1, 1 if mode == "gray" else 3, 32, 32))
    if args.model.resolve() == args.output.resolve():
        raise ValueError("The exported model must use a different path from its checkpoint.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example)
    traced.save(str(args.output))
    print(json.dumps({"input_mode": mode, "input_layout": "NCHW float32 in [0, 1]", "output": "real logit"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--cache", type=Path, default=Path("artifacts/recordings.npz"))
    prep.add_argument("--seed", type=int, default=42)
    fit = sub.add_parser("train")
    fit.add_argument("--cache", type=Path, default=Path("artifacts/recordings.npz"))
    fit.add_argument("--epochs", type=int, default=15)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts/personal"))
    infer = sub.add_parser("predict")
    infer.add_argument("--video", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    bench = sub.add_parser("evaluate")
    bench.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    bench.add_argument("--cache", type=Path, default=Path("artifacts/recordings.npz"))
    bench.add_argument("--split", choices=SPLITS, default="test")
    convert = sub.add_parser("export")
    convert.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    convert.add_argument("--output", type=Path, required=True)
    for command in (prep, infer):
        command.add_argument("--detector", type=Path, default=DETECTOR)
        command.add_argument("--sample-fps", type=float, default=2)
        command.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if args.command in ("prepare", "predict") and (
        not all(math.isfinite(value) and value > 0 for value in (args.sample_fps, args.seconds))
    ):
        parser.error("Sampling rate and duration must be positive and finite.")
    if args.command == "train" and args.epochs < 1:
        parser.error("Epochs must be positive.")
    {"prepare": prepare, "train": train, "predict": predict, "evaluate": evaluate, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
