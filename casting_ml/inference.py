"""Shared ONNX CPU inference; no training-only dependencies."""
import hashlib
import io
import json
import threading
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


class Predictor:
    def __init__(self, model_path="docs/model.onnx", metadata_path="artifacts/metadata.json"):
        self.metadata = json.loads(Path(metadata_path).read_text())
        path = Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise RuntimeError("Model checksum does not match metadata")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()
        self.predict_tensor(np.zeros((1, 3, self.metadata["image_size"], self.metadata["image_size"]), dtype=np.float32))

    def predict_tensor(self, tensor):
        with self.lock:
            return float(self.session.run(None, {"image": tensor})[0].ravel()[0])

    def predict_bytes(self, data):
        if len(data) > MAX_BYTES:
            raise ValueError("Image exceeds 10 MiB limit")
        if not data:
            raise ValueError("Empty upload")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 20 megapixel limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
                size = self.metadata["image_size"]
                image = image.resize((size, size), Image.Resampling.BILINEAR)
                value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()[None] / 255.0
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid or unsupported image") from exc
        probability = self.predict_tensor(value)
        defective = probability >= self.metadata["threshold"]
        return dict(label="def_front" if defective else "ok_front", defective=defective,
                    defect_probability=probability, threshold=self.metadata["threshold"],
                    model=self.metadata["architecture"],
                    note="Prototype: probability is uncalibrated; use on similar front-view castings only.")
