"""Inference for the saved pixel baseline and ONNX ResNet model."""
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
    def __init__(self, model_path="docs/model.onnx", metadata_path="artifacts/metadata.json", baseline_path="artifacts/baseline.npz"):
        self.metadata = json.loads(Path(metadata_path).read_text())
        path = Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise RuntimeError("Model checksum does not match metadata")
        baseline_file = Path(baseline_path)
        if hashlib.sha256(baseline_file.read_bytes()).hexdigest() != self.metadata["baseline_sha256"]:
            raise RuntimeError("Baseline checksum does not match metadata")
        with np.load(baseline_file, allow_pickle=False) as parameters:
            self.baseline = {key: parameters[key].copy() for key in parameters.files}
        self.models = {
            "resnet18_frozen_logistic": {"name": "ResNet-18 + logistic head", "threshold": self.metadata["threshold"]},
            "pixels_logistic": {"name": "Pixel logistic baseline", "threshold": float(self.baseline["threshold"])},
        }
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()
        self.predict_tensor(np.zeros((1, 3, self.metadata["image_size"], self.metadata["image_size"]), dtype=np.float32))

    def predict_tensor(self, tensor):
        with self.lock:
            return float(self.session.run(None, {"image": tensor})[0].ravel()[0])

    def baseline_probability(self, image):
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255.0
        pixels -= self.baseline["mean"]
        pixels /= self.baseline["scale"]
        logit = float((pixels @ self.baseline["coef"].T + self.baseline["intercept"]).item())
        if logit >= 0:
            return float(1 / (1 + np.exp(-logit)))
        exponent = np.exp(logit)
        return float(exponent / (1 + exponent))

    def predict_bytes(self, data, model="resnet18_frozen_logistic"):
        if model not in self.models:
            raise ValueError("Unknown model")
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
                if model == "pixels_logistic":
                    probability = self.baseline_probability(image)
                else:
                    size = self.metadata["image_size"]
                    image = image.resize((size, size), Image.Resampling.BILINEAR)
                    value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()[None] / 255.0
                    probability = self.predict_tensor(value)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid or unsupported image") from exc
        threshold = self.models[model]["threshold"]
        defective = probability >= threshold
        return dict(label="def_front" if defective else "ok_front", defective=defective,
                    defect_probability=probability, threshold=threshold,
                    model=model,
                    note="Prototype: probability is uncalibrated; use on similar front-view castings only.")
