# Face Liveness Detection | PyTorch

**Nguyễn Chiến Thắng** · Personal project · Python / PyTorch / OpenCV / Streamlit

A passive face liveness prototype trained on 91 videos I recorded. It uses YuNet to locate a face, a compact CNN to score 32 × 32 crops, and the mean frame score to classify a clip as real or spoof.

![Held-out clip results for the RGB and grayscale CNNs](docs/media/benchmark.png)

## Results

I recorded 47 spoof and 44 real clips. I assigned clips to train, validation and test before sampling frames, using a stratified split of **63 / 14 / 14 videos**. Of 2,098 sampled frames, 1,712 contained one face suitable for scoring. Both CNNs started from random weights and were trained for 15 epochs on CPU.

| Input | Checkpoint selected at epoch | Spoof clips accepted | Genuine clips rejected | Test clip ACER |
| --- | ---: | ---: | ---: | ---: |
| **RGB** | 7 | **1/7** | **0/7** | **7.14%** |
| Grayscale | 6 | 5/7 | 0/7 | 35.71% |

The RGB variant was selected using validation clip ACER, with validation frame loss as the tie-breaker. It classified **13 of 14 test clips** correctly. ACER here averages the spoof acceptance rate and genuine rejection rate at the fixed 0.50 threshold. [Full benchmark and confusion matrices](docs/benchmark.json)

## Method

1. Sample approximately 2 frames per second from the first 12 seconds of each clip.
2. Detect faces with YuNet. Keep frames with exactly one detection, then crop, resize and normalize to 32 × 32 RGB.
3. Train the same three-block CNN with RGB or grayscale input. Weight frames inversely by the number of eligible frames in their training clip so longer clips do not dominate.
4. Average the sigmoid scores across eligible frames. A score above 0.50 yields `real`. A clip without eligible frames remains unscored.

```mermaid
flowchart LR
    A[Labeled clips] --> B[Split by video]
    B --> C[Sample and detect faces]
    C --> D[32 × 32 crops]
    D --> E[RGB or grayscale CNN]
    E --> F[Average frame scores]
    F --> G[Clip result and metrics]
```

## Run locally

Use Python 3.11 from the repository root. The detector is bundled. Install PyTorch and the remaining dependencies into a virtual environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Organize your local videos under `recordings/Video real/` and `recordings/Video fake/`. Use at least seven clips per label. Keep each folder's recordings fixed to reproduce its seed-based split.

```bash
python personal_liveness.py prepare --data path/to/recordings
python personal_liveness.py train --epochs 15
python personal_liveness.py predict --model artifacts/personal/liveness_rgb.pt --video path/to/clip.mp4
```

To try the browser interface on your machine after training:

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run app.py
```

The demo lets you choose a local checkpoint and score a video. It reports the predicted label, the model's score for the real class, and frame coverage. The project runs locally without an API key. Videos, cached face crops and trained checkpoints stay on the machine. The repository includes the source and aggregate benchmark.

## Evaluation scope

The test set contains 14 videos. It was separated by video; identity and session separation were not verified. The scores describe this small offline experiment. Testing on new identities, cameras and attack media is needed to measure generalization.

Face detection uses the MIT-licensed [YuNet model from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). [Bundled license](face_detector/LICENSE)
