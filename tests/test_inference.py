import io
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from casting_ml.api import app
from casting_ml.inference import Predictor, MAX_BYTES


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as active:
        yield active


@pytest.mark.parametrize("name,target", [("ok", False), ("defective", True)])
def test_exported_examples(predictor, name, target):
    result = predictor.predict_bytes(Path(f"docs/example-{name}.png").read_bytes())
    assert result["defective"] is target
    metadata = json.loads(Path("artifacts/metadata.json").read_text())
    expected = next(r["expected_probability"] for r in metadata["examples"] if r["target"] == int(target))
    assert result["defect_probability"] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("data", [b"", b"this is not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_uploads_rejected(predictor, data):
    with pytest.raises(ValueError):
        predictor.predict_bytes(data)


def test_checksum_mismatch_rejected(tmp_path):
    model = tmp_path/"bad.onnx"
    model.write_bytes(b"corrupted model")
    with pytest.raises(RuntimeError, match="checksum"):
        Predictor(model)


def test_unsupported_format(predictor):
    stream = io.BytesIO(); Image.new("RGB", (10,10)).save(stream, "GIF")
    with pytest.raises(ValueError, match="JPEG and PNG"):
        predictor.predict_bytes(stream.getvalue())


def test_oversize(predictor, client):
    with pytest.raises(ValueError, match="10 MiB"):
        predictor.predict_bytes(b"0" * (MAX_BYTES+1))
    response = client.post("/predict", files={"file": ("large.png", b"0"*(MAX_BYTES+1), "image/png")})
    assert response.status_code == 413


def test_api_success_and_failure(client):
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/predict", files={"file": ("example.png", Path("docs/example-defective.png").read_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["defective"] is True
    assert response.json()["inference_ms"] > 0
    assert client.post("/predict", files={"file": ("bad.jpg", b"bad", "image/jpeg")}).status_code == 400
    assert client.post("/predict").status_code == 422


def test_large_dimensions_rejected(predictor):
    stream = io.BytesIO(); Image.new("L", (5000, 4001)).save(stream,"PNG")
    with pytest.raises(ValueError, match="20 megapixel"):
        predictor.predict_bytes(stream.getvalue())
