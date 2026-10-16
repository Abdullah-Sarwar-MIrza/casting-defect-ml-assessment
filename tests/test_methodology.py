import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pytest
from casting_ml.evaluate import choose_threshold, metrics


def test_split_integrity():
    with Path("reports/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    hashes = {s:{r["pixel_sha256"] for r in rows if r["split"]==s} for s in ["train","validation","test"]}
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["validation"] & hashes["test"]
    assert all(r["source"]=="train" for r in rows if r["split"] in {"train","validation"})
    assert sum(r["split"]=="test" for r in rows)==715
    assert all(r["source"]=="test" for r in rows if r["split"]=="test")
    audit = json.loads(Path("reports/data_audit.json").read_text())
    assert len(rows)==audit["total_images"]


def test_metrics_recomputed_from_saved_predictions():
    with Path("reports/test_predictions.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    y=np.array([int(r["target"]) for r in rows]); p=np.array([float(r["transfer_probability"]) for r in rows])
    result=json.loads(Path("reports/results.json").read_text())["test_results"]["resnet18_frozen_logistic"]
    actual=metrics(y,p,result["threshold"])
    for key, value in actual.items():
        if key == "confusion_matrix":
            assert value == result["metrics"][key]
        else:
            assert value == pytest.approx(result["metrics"][key], abs=1e-8)
    assert sum(int(a!=b) for a,b in zip(y,p>=result["threshold"]))==5


def test_validation_selection_matches_trials():
    choices=json.loads(Path("reports/validation_selection.json").read_text())
    trials=json.loads(Path("reports/tuning_trials.json").read_text())
    for name in ["pixels_logistic","resnet18_frozen_logistic"]:
        candidates=[r for r in trials if r["model"]==name]
        best=max(candidates,key=lambda r:(r["validation"]["macro_f1"],r["validation"]["defect_f2"],-r["C"]))
        chosen=choices["baseline" if name=="pixels_logistic" else "transfer"]
        assert best==chosen
    assert choices["selected_model"]=="resnet18_frozen_logistic"


def test_threshold_tuning_prefers_recall_when_f2_improves():
    y=np.array([0,0,1,1]); p=np.array([.1,.25,.4,.8])
    assert choose_threshold(y,p,[.3,.5])==.3


def test_saved_baseline_reproduces_predictions():
    from PIL import Image
    from casting_ml.data import read_rgb
    parameters=np.load("artifacts/baseline.npz",allow_pickle=False)
    config=json.loads(Path("config.json").read_text())
    dataset=Path(config["dataset_root"])
    if not dataset.exists():
        pytest.skip("Full dataset is external and is not distributed in the repository")
    with Path("reports/test_predictions.csv").open() as stream:
        rows=list(csv.DictReader(stream))[:20]
    x=np.stack([np.asarray(read_rgb(dataset/r["path"]).convert("L").resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32).ravel()/255.0 for r in rows])
    logits=((x-parameters["mean"])/parameters["scale"])@parameters["coef"].T+parameters["intercept"]
    probability=1/(1+np.exp(-logits.ravel()))
    assert np.max(np.abs(probability-np.array([float(r["baseline_probability"]) for r in rows])))<1e-6
