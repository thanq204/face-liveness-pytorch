"""Verify the Flask contract using a local checkpoint and synthetic inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from api import create_app
from personal_liveness import LivenessCNN


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        checkpoint = Path(cls.directory.name) / "liveness.pt"
        torch.save({"mode": "rgb", "model": LivenessCNN(3).state_dict()}, checkpoint)
        cls.app = create_app(checkpoint)
        cls.jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))[1].tobytes()

    def test_web_and_health(self):
        with self.app.test_client() as client:
            for path in ("/", "/static/webcam.js"):
                with client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/v1/health").json["input_mode"], "rgb")

    def test_image_validation_and_no_face(self):
        with self.app.test_client() as client:
            self.assertEqual(client.post("/v1/analyze", data=b"x").status_code, 415)
            self.assertEqual(client.post("/v1/analyze", data=b"x", content_type="image/jpeg").status_code, 422)
            response = client.post("/v1/analyze", data=self.jpeg, content_type="image/jpeg")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, {"box": None, "label": None, "p_real": None})

    def test_session_lifecycle(self):
        with self.app.test_client() as client:
            response = client.post("/v1/sessions")
            self.assertEqual(response.status_code, 201)
            session = "/v1/sessions/" + response.json["session_id"]
            frame = client.post(session + "/frames", data=self.jpeg, content_type="image/jpeg")
            self.assertEqual(frame.status_code, 200)
            self.assertEqual(frame.json["window_frames"], 0)
            self.assertEqual(client.delete(session).status_code, 204)
            closed = client.post(session + "/frames", data=self.jpeg, content_type="image/jpeg")
            self.assertEqual(closed.status_code, 404)

    def test_video_requires_multipart_upload(self):
        with self.app.test_client() as client:
            self.assertEqual(client.post("/v1/predict").status_code, 400)
