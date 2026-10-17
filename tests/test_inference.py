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


@pytest.mark.parametrize("name", ["ok", "defective"])
def test_baseline_matches_saved_predictions(predictor, name):
    import csv
    metadata=json.loads(Path("artifacts/metadata.json").read_text())
    source=next(example["source"] for example in metadata["examples"] if example["file"]==f"example-{name}.png")
    with Path("reports/test_predictions.csv").open() as stream:
        expected=next(float(row["baseline_probability"]) for row in csv.DictReader(stream) if row["path"]==source)
    result=predictor.predict_bytes(Path(f"docs/example-{name}.png").read_bytes(),"pixels_logistic")
    assert result["model"]=="pixels_logistic"
    assert result["threshold"]==0.25
    assert result["defect_probability"]==pytest.approx(expected,abs=1e-5,rel=0)
    assert result["defective"] is (expected>=0.25)


@pytest.mark.parametrize("model,threshold", [("pixels_logistic",0.25),("resnet18_frozen_logistic",0.5)])
def test_api_model_selection(client,predictor,model,threshold):
    for name in ["ok","defective"]:
        data=Path(f"docs/example-{name}.png").read_bytes()
        response=client.post("/predict",params={"model":model},files={"file":("example.png",data,"image/png")})
        assert response.status_code==200
        actual=response.json(); expected=predictor.predict_bytes(data,model)
        assert actual["model"]==model and actual["threshold"]==threshold
        assert actual["defect_probability"]==expected["defect_probability"]
        assert actual["defective"]==expected["defective"]


def test_model_listing_and_invalid_selection(client,predictor):
    models=client.get("/models").json()["models"]
    assert {row["id"]:row["threshold"] for row in models}=={"pixels_logistic":0.25,"resnet18_frozen_logistic":0.5}
    response=client.post("/predict?model=unknown",files={"file":("example.png",Path("docs/example-ok.png").read_bytes(),"image/png")})
    assert response.status_code==422
    with pytest.raises(ValueError,match="Unknown model"):
        predictor.predict_bytes(b"invalid","unknown")


def test_browser_baseline_uses_saved_weights():
    import hashlib
    spec=json.loads(Path("docs/baseline.json").read_text())
    assert spec["source_sha256"]==hashlib.sha256(Path("artifacts/baseline.npz").read_bytes()).hexdigest()
    with np.load("artifacts/baseline.npz",allow_pickle=False) as parameters:
        for key in ["mean","scale","coef"]:
            np.testing.assert_array_equal(np.asarray(spec[key]),parameters[key].ravel())
        assert spec["intercept"]==float(parameters["intercept"].ravel()[0])
        assert spec["threshold"]==float(parameters["threshold"])


def test_baseline_checksum_mismatch(tmp_path):
    path=tmp_path/"baseline.npz"; path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError,match="Baseline checksum"):
        Predictor(baseline_path=path)
