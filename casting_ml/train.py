"""Run the complete CPU-friendly experiment: python -m casting_ml.train."""
import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import prepare, read_rgb, image_tensor
from .model import DefectNetwork
from .evaluate import metrics, choose_threshold, bootstrap_intervals, visual_report

LOG = logging.getLogger(__name__)


def baseline_features(root, rows):
    return np.stack([np.asarray(read_rgb(root / r["path"]).convert("L").resize((32, 32), Image.Resampling.BILINEAR),
                                dtype=np.float32).ravel() / 255.0 for r in rows])


@torch.inference_mode()
def deep_features(model, root, rows, size, batch_size, augment=False, seed=42):
    rng = np.random.default_rng(seed)
    outputs = []
    for start in range(0, len(rows), batch_size):
        images = []
        for row in rows[start:start + batch_size]:
            if augment:
                im = read_rgb(root / row["path"])
                im = ImageOps.mirror(im) if rng.random() < 0.5 else ImageOps.flip(im)
                im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.9, 1.1)))
                im = im.resize((size, size), Image.Resampling.BILINEAR)
                value = np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
            else:
                value = image_tensor(root / row["path"], size)
            images.append(value)
        outputs.append(model.features(torch.from_numpy(np.stack(images))).numpy())
        if start % (batch_size * 10) == 0:
            LOG.info("Feature extraction %s %d/%d", "augmented" if augment else "original", start, len(rows))
    return np.concatenate(outputs)


def tune(name, train_x, train_y, val_x, val_y, values, grid):
    results, fitted = [], []
    for c in values:
        begin = time.perf_counter()
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42))
        classifier.fit(train_x, train_y)
        p = classifier.predict_proba(val_x)[:, 1]
        threshold = choose_threshold(val_y, p, grid)
        score = metrics(val_y, p, threshold)
        results.append(dict(model=name, C=c, threshold=threshold, validation=score,
                            fit_seconds=time.perf_counter() - begin,
                            iterations=int(classifier[-1].n_iter_[0])))
        fitted.append(classifier)
        LOG.info("%s C=%g validation macro F1=%.4f F2=%.4f threshold=%.2f", name, c, score["macro_f1"], score["defect_f2"], threshold)
    best = max(range(len(results)), key=lambda i: (results[i]["validation"]["macro_f1"], results[i]["validation"]["defect_f2"], -results[i]["C"]))
    return fitted[best], results[best], results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
    output, artifacts, cache = Path("reports"), Path("artifacts"), Path("cache")
    for folder in (output, artifacts, cache, Path("docs")):
        folder.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "training.log", encoding="utf-8")])
    started = time.perf_counter()
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    root = Path(config["dataset_root"]).resolve()
    records, audit = prepare(root, output, seed, config["validation_fraction"])
    partitions = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}
    labels = {s: np.array([r["target"] for r in rows]) for s, rows in partitions.items()}
    # Hyperparameters are selected before any test features or test predictions are made.
    baseline_x = {s: baseline_features(root, partitions[s]) for s in ("train", "validation")}
    baseline, baseline_choice, baseline_trials = tune("pixels_logistic", baseline_x["train"], labels["train"], baseline_x["validation"], labels["validation"], config["baseline_C"], config["threshold_grid"])
    np.savez(artifacts / "baseline.npz", mean=baseline[0].mean_, scale=baseline[0].scale_,
             coef=baseline[-1].coef_, intercept=baseline[-1].intercept_, threshold=baseline_choice["threshold"])
    model = DefectNetwork(pretrained=True).eval()
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    deep_x = {}
    fingerprint = hashlib.sha256((output / "manifest.csv").read_bytes() + json.dumps(config, sort_keys=True).encode() + b"resnet18-imagenet1k-v1-original-and-flip-brightness-v1").hexdigest()[:16]
    for split in ("train", "validation"):
        file = cache / f'{fingerprint}-{split}.npy'
        if file.exists():
            deep_x[split] = np.load(file)
        else:
            deep_x[split] = deep_features(model, root, partitions[split], config["image_size"], config["batch_size"])
            np.save(file, deep_x[split])
    aug_file = cache / f'{fingerprint}-augmented.npy'
    if aug_file.exists():
        augmented = np.load(aug_file)
    else:
        augmented = deep_features(model, root, partitions["train"], config["image_size"], config["batch_size"], augment=True, seed=seed)
        np.save(aug_file, augmented)
    transfer, transfer_choice, transfer_trials = tune("resnet18_frozen_logistic", np.concatenate([deep_x["train"], augmented]),
                                                    np.tile(labels["train"], 2), deep_x["validation"], labels["validation"],
                                                    config["transfer_C"], config["threshold_grid"])
    # Lock choices to disk before the first held-out test inference.
    choices = dict(baseline=baseline_choice, transfer=transfer_choice,
                   selection_rule="maximize validation macro F1; threshold tuned for validation defect F2")
    selected = max([baseline_choice, transfer_choice], key=lambda c: (c["validation"]["macro_f1"], c["validation"]["defect_f2"]))
    choices["selected_model"] = selected["model"]
    (output / "validation_selection.json").write_text(json.dumps(choices, indent=2))
    (output / "tuning_trials.json").write_text(json.dumps(baseline_trials + transfer_trials, indent=2))
    # Fuse StandardScaler and sklearn head into Torch, so deployment has one model.
    scaler, head = transfer.steps[0][1], transfer.steps[1][1]
    weight = head.coef_[0] / scaler.scale_
    bias = head.intercept_[0] - np.dot(weight, scaler.mean_)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(weight, dtype=torch.float32).view(1, -1))
        model.head.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    test_x = deep_features(model, root, partitions["test"], config["image_size"], config["batch_size"])
    p_transfer = transfer.predict_proba(test_x)[:, 1]
    p_baseline = baseline.predict_proba(baseline_features(root, partitions["test"]))[:, 1]
    with torch.inference_mode():
        fused = torch.sigmoid(model.head(torch.tensor(test_x))).numpy().ravel()
    if not np.allclose(fused, p_transfer, atol=2e-5):
        raise RuntimeError("Fused classifier parity failed")
    test_results = {}
    for choice, probability in [(baseline_choice, p_baseline), (transfer_choice, p_transfer)]:
        test_results[choice["model"]] = dict(threshold=choice["threshold"],
                                           metrics=metrics(labels["test"], probability, choice["threshold"]),
                                           ci95_stratified_bootstrap=bootstrap_intervals(labels["test"], probability, choice["threshold"], seed))
    if selected["model"] != transfer_choice["model"]:
        raise RuntimeError("Baseline won validation; adapt export for baseline before deploying. Test results do not override selection.")
    checkpoint = dict(state_dict=model.state_dict(), image_size=config["image_size"], threshold=transfer_choice["threshold"],
                      architecture="resnet18_frozen_logistic", seed=seed)
    torch.save(checkpoint, artifacts / "model.pt")
    dummy = torch.zeros(1, 3, config["image_size"], config["image_size"])
    torch.onnx.export(model, dummy, str(Path("docs") / "model.onnx"), input_names=["image"], output_names=["defect_probability"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    session = ort.InferenceSession("docs/model.onnx", providers=["CPUExecutionProvider"])
    parity_images = np.stack([image_tensor(root / r["path"], config["image_size"]) for r in partitions["test"][:20]])
    with torch.inference_mode():
        reference = model(torch.tensor(parity_images)).numpy().ravel()
    exported = np.array([session.run(None, {"image": x[None]})[0].item() for x in parity_images])
    parity_error = float(np.max(np.abs(reference-exported)))
    assert parity_error < 1e-4
    review = visual_report(root, partitions["test"], p_transfer, transfer_choice["threshold"], output)
    # Save non-error examples for public smoke tests and interactive demo.
    examples = []
    for target, label in [(0, "ok"), (1, "defective")]:
        index = next(i for i, row in enumerate(partitions["test"]) if row["target"] == target and (p_transfer[i] >= transfer_choice["threshold"]) == target)
        dest = Path("docs") / f'example-{label}.png'
        read_rgb(root / partitions["test"][index]["path"]).save(dest)
        examples.append(dict(file=dest.name, source=partitions["test"][index]["path"], target=target,
                             expected_probability=float(p_transfer[index])))
    metadata = dict(architecture="resnet18_frozen_logistic", image_size=config["image_size"],
                    threshold=transfer_choice["threshold"], labels={"0": "ok_front", "1": "def_front"},
                    normalization=dict(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],location="inside exported graph"),
                    model_sha256=hashlib.sha256(Path("docs/model.onnx").read_bytes()).hexdigest(),
                    torch_sha256=hashlib.sha256((artifacts/"model.pt").read_bytes()).hexdigest(),
                    examples=examples, onnx_max_absolute_error=parity_error)
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    Path("docs/metadata.json").write_text(json.dumps(metadata, indent=2))
    with (output / "test_predictions.csv").open("w", newline="") as stream:
        writer=csv.writer(stream); writer.writerow(["path","target","baseline_probability","transfer_probability","transfer_prediction"])
        writer.writerows((r["path"],r["target"],float(a),float(b),int(b >= transfer_choice["threshold"])) for r,a,b in zip(partitions["test"],p_baseline,p_transfer))
    summary = dict(config=config, selected_model=selected["model"], test_results=test_results,
                   training_seconds=time.perf_counter()-started,
                   environment=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                                    cpu_threads=torch.get_num_threads(), cuda_used=False),
                   pretrained_weights="torchvision ResNet18_Weights.IMAGENET1K_V1", pretrained_finetuning=False,
                   augmented_training_images=len(augmented), feature_cache_fingerprint=fingerprint,
                   onnx_max_absolute_error=parity_error, test_errors=len(review))
    (output / "results.json").write_text(json.dumps(summary, indent=2))
    LOG.info("COMPLETE: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
