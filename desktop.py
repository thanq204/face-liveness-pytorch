"""Display face liveness on a webcam or annotate a video with the local CNN."""

from collections import deque
import argparse
import json
import math
from pathlib import Path

import cv2
import torch

from personal_liveness import DEFAULT_MODEL, DETECTOR, score_frame, load_classifier, make_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, help="Optional annotated MP4.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    if args.output and args.output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension.")
    if args.video and args.output and args.video.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    model, mode = load_classifier(args.model)
    detector = make_detector(DETECTOR)
    cap = cv2.VideoCapture(args.camera if args.video is None else str(args.video))
    writer = None
    history = deque(maxlen=6)
    frames, scored = 0, 0
    try:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Cannot read the selected webcam or video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 20.0
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1],
            )
            if not writer.isOpened():
                raise SystemExit("OpenCV could not open the MP4 output.")
        while ok:
            result = score_frame(frame, detector, model, mode)
            if result["p_real"] is None:
                history.clear()
                label, color = "Need exactly one face", (40, 190, 230)
            else:
                scored += 1
                history.append(result["p_real"])
                score = sum(history) / len(history)
                label = f"{'REAL' if score > 0.5 else 'FAKE'}  |  P(real) {score:.2f}"
                color = (95, 225, 90) if score > 0.5 else (70, 95, 245)
                left, top, right, bottom = result["box"]
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 25, 35), -1)
            cv2.putText(frame, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            frames += 1
            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Face Liveness", frame)
                if cv2.waitKey(1 if args.camera is not None else max(1, round(1000 / fps))) & 255 in (27, ord("q")):
                    break
            if args.max_frames is not None and frames >= args.max_frames:
                break
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({"frames": frames, "scored_frames": scored, "input_mode": mode}))


if __name__ == "__main__":
    main()
