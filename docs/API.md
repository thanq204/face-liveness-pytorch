# Face Liveness REST API

Start the local server from the repository root after training:

```bash
python -m pip install -r requirements-api.txt
python api.py --model artifacts/personal/liveness_rgb.pt --port 8000
```

| Method | Endpoint | Input and result |
| --- | --- | --- |
| GET | `/` | Live webcam page |
| GET | `/v1/health` | Readiness and checkpoint input mode |
| POST | `/v1/analyze` | Raw JPEG. Returns a label, real-class score and face box |
| POST | `/v1/sessions` | Opens a webcam session and returns its ID |
| POST | `/v1/sessions/{id}/frames` | Raw JPEG. Returns the score averaged over up to six eligible frames |
| DELETE | `/v1/sessions/{id}` | Ends a session |
| POST | `/v1/predict` | Multipart field `video`. Returns a clip decision and frame coverage |

The image endpoints accept JPEGs up to 2 MiB, with each dimension from 32 to 1920 pixels. Set `Content-Type: image/jpeg` and send the encoded image bytes directly. The box is `[left, top, right, bottom]` in decoded image pixels. If there is no single valid face, the label, score and box are null.

Webcam sessions expire after 120 seconds of inactivity. A frame without a single face clears the smoothing window. Video requests are limited to 64 MiB including multipart data and sample approximately two frames per second in the first 12 seconds. Uploads are processed on the local machine and the API removes its temporary video file after scoring.

```bash
curl -H "Content-Type: image/jpeg" --data-binary @frame.jpg http://127.0.0.1:8000/v1/analyze
curl -F "video=@clip.mp4" http://127.0.0.1:8000/v1/predict
```

The server binds to `127.0.0.1` and requires no API key. The score is a model output for the real class.
