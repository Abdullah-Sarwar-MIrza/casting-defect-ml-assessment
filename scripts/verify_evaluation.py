"""Read-only evaluation replay. Does not train, tune, or overwrite model artifacts."""
import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from casting_ml.data import image_tensor, read_rgb
from casting_ml.evaluate import metrics, bootstrap_intervals
from casting_ml.inference import Predictor
from casting_ml.model import DefectNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text())
    dataset = Path(args.dataset_root or config["dataset_root"]).resolve()
    if not dataset.is_dir():
        parser.error("Dataset not found; pass --dataset-root. Evaluation requires the external dataset.")
    recorded = json.loads(Path("reports/results.json").read_text())
    metadata = json.loads(Path("artifacts/metadata.json").read_text())
    assert metadata == json.loads(Path("docs/metadata.json").read_text())
    files = ["artifacts/model.pt", "artifacts/baseline.npz", "docs/model.onnx"]
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    for name in files:
        # Check the committed artifacts too: review must never silently change a trained model.
        committed = subprocess.check_output(["git", "show", "HEAD:" + name])
        assert hashlib.sha256(committed).hexdigest() == hashes[name], name
    assert hashes["artifacts/model.pt"] == metadata["torch_sha256"]
    assert hashes["docs/model.onnx"] == metadata["model_sha256"]
    with Path("reports/manifest.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    # Recheck every input against the recorded decoded-image hashes.
    for row in manifest:
        path = dataset / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"], row["path"]
        image = read_rgb(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        assert digest == row["pixel_sha256"], row["path"]
    with Path("reports/test_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    test_labels = {r["path"]: int(r["target"]) for r in manifest if r["split"] == "test"}
    assert len(rows) == len(test_labels) == len({r["path"] for r in rows})
    assert {r["path"]: int(r["target"]) for r in rows} == test_labels
    y = np.array([int(r["target"]) for r in rows])
    checks = {}
    for name, column in [("pixels_logistic", "baseline_probability"), ("resnet18_frozen_logistic", "transfer_probability")]:
        p = np.array([float(r[column]) for r in rows])
        original = recorded["test_results"][name]
        calculated = metrics(y, p, original["threshold"])
        for key, value in calculated.items():
            if key == "confusion_matrix":
                assert value == original["metrics"][key]
            else:
                assert abs(value-original["metrics"][key]) < 1e-8, (name, key)
        intervals = bootstrap_intervals(y, p, original["threshold"], seed=config["seed"])
        for key, value in intervals.items():
            np.testing.assert_allclose(value, original["ci95_stratified_bootstrap"][key], atol=1e-12, rtol=0)
        checks[name] = dict(metrics=calculated, bootstrap_intervals=intervals, matches_recorded_metrics=True)
    torch.set_num_threads(config["cpu_threads"])
    checkpoint = torch.load("artifacts/model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["image_size"] == metadata["image_size"]
    assert checkpoint["threshold"] == metadata["threshold"] == recorded["test_results"]["resnet18_frozen_logistic"]["threshold"]
    network = DefectNetwork().eval()
    network.load_state_dict(checkpoint["state_dict"])
    predictor = Predictor()
    baseline = np.load("artifacts/baseline.npz", allow_pickle=False)
    assert float(baseline["threshold"]) == recorded["test_results"]["pixels_logistic"]["threshold"]
    torch_values, onnx_values, baseline_values = [], [], []
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        tensor = np.stack([image_tensor(dataset/r["path"], metadata["image_size"]) for r in batch])
        with torch.inference_mode():
            torch_values.extend(network(torch.from_numpy(tensor)).numpy().ravel().tolist())
        onnx_values.extend(predictor.predict_tensor(x[None]) for x in tensor)
        pixels = np.stack([np.asarray(read_rgb(dataset/r["path"]).convert("L").resize((32,32), Image.Resampling.BILINEAR), dtype=np.float32).ravel()/255.0 for r in batch])
        # Match sklearn StandardScaler float32 in-place preprocessing.
        scaled = pixels.copy()
        scaled -= baseline["mean"]
        scaled /= baseline["scale"]
        logits = scaled@baseline["coef"].T+baseline["intercept"]
        baseline_values.extend((1/(1+np.exp(-logits.ravel()))).tolist())
    saved_baseline = np.array([float(r["baseline_probability"]) for r in rows])
    saved_transfer = np.array([float(r["transfer_probability"]) for r in rows])
    differences = dict(baseline_vs_saved=float(np.max(np.abs(baseline_values-saved_baseline))),
                       torch_vs_saved=float(np.max(np.abs(torch_values-saved_transfer))),
                       onnx_vs_saved=float(np.max(np.abs(onnx_values-saved_transfer))),
                       torch_vs_onnx=float(np.max(np.abs(np.array(torch_values)-onnx_values))))
    assert differences["baseline_vs_saved"] < 1e-5
    assert max(v for k,v in differences.items() if k != "baseline_vs_saved") < 1e-4
    for values, name in [(baseline_values,"pixels_logistic"),(torch_values,"resnet18_frozen_logistic"),(onnx_values,"resnet18_frozen_logistic")]:
        original = recorded["test_results"][name]
        assert metrics(y, np.array(values), original["threshold"])["confusion_matrix"] == original["metrics"]["confusion_matrix"]
    # Original reports remain intact; only write this review's verification evidence.
    report = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_files_verified=len(manifest), test_images_replayed=len(rows),
                  artifact_sha256=hashes, artifacts_match_git_and_metadata=True,
                  recorded_metrics=checks, maximum_absolute_probability_differences=differences,
                  replay_score_tolerances=dict(baseline=1e-5,transfer=1e-4),
                  fresh_inference_confusion_matrices_match=True,
                  retraining_performed=False, clean_environment_install_verified=False,
                  note="Replayed saved artifacts and seeded bootstrap; did not rerun training. Metrics describe the original saved predictions; float rounding can affect score ties.")
    Path("reports/evaluation_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
