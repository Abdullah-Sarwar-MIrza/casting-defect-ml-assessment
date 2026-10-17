"""Held-out metrics, uncertainty, and visual error review."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, fbeta_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             brier_score_loss)


def metrics(y, probability, threshold):
    pred = np.asarray(probability) >= threshold
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                defect_precision=float(precision_score(y, pred, zero_division=0)),
                defect_recall=float(recall_score(y, pred, zero_division=0)),
                defect_f2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
                roc_auc=float(roc_auc_score(y, probability)),
                defect_average_precision=float(average_precision_score(y, probability)),
                brier_score=float(brier_score_loss(y, probability)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def choose_threshold(y, probability, grid):
    # Validation only: F2 values missing defects four times more heavily than false alarms.
    ranked = [(fbeta_score(y, probability >= t, beta=2, zero_division=0),
               f1_score(y, probability >= t, average="macro", zero_division=0),
               -abs(t - 0.5), t) for t in grid]
    return float(max(ranked)[3])


def bootstrap_intervals(y, probability, threshold, seed=42, repeats=1000):
    rng = np.random.default_rng(seed)
    y, probability = np.asarray(y), np.asarray(probability)
    values = []
    for _ in range(repeats):
        # Stratified resampling preserves class balance in each draw.
        ids = np.concatenate([rng.choice(np.flatnonzero(y == c), size=sum(y == c), replace=True)
                              for c in (0, 1)])
        prediction = probability[ids] >= threshold
        values.append([accuracy_score(y[ids], prediction),
                       recall_score(y[ids], prediction, zero_division=0),
                       f1_score(y[ids], prediction, average="macro", zero_division=0)])
    bounds = np.percentile(values, [2.5, 97.5], axis=0)
    return {key: bounds[:, i].tolist() for i, key in enumerate(["accuracy", "defect_recall", "macro_f1"])}


def visual_report(root, rows, probabilities, threshold, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .data import read_rgb
    output = Path(output)
    y = np.array([r["target"] for r in rows])
    pred = probabilities >= threshold
    wrong = np.flatnonzero(pred != y)
    # Prioritize false negatives, then strongest mistakes.
    order = sorted(wrong, key=lambda i: (y[i] != 1, -abs(probabilities[i] - threshold)))
    review = []
    for i in order:
        im = np.asarray(read_rgb(Path(root) / rows[i]["path"]).convert("L"), dtype=float)
        review.append(dict(path=rows[i]["path"], truth=int(y[i]), prediction=int(pred[i]),
                           defect_probability=float(probabilities[i]), mean_brightness=float(im.mean()),
                           contrast_std=float(im.std()),
                           review="Folder-level labels do not identify defect locations; visual observations are unverified."))
    (output / "misclassifications.json").write_text(json.dumps(review, indent=2))
    # Include representative correct examples when no mistakes exist.
    chosen = order[:16] if order else list(range(min(8, len(rows))))
    fig, axes = plt.subplots(max(1, (len(chosen) + 3) // 4), 4, figsize=(12, max(3, ((len(chosen)+3)//4)*3)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, chosen):
        ax.imshow(read_rgb(Path(root) / rows[i]["path"]))
        ax.set_title(f'true={y[i]} pred={int(pred[i])} p(def)={probabilities[i]:.3f}', fontsize=9)
    fig.suptitle("Held-out errors (1=defective); correct examples if no errors")
    fig.tight_layout()
    fig.savefig(output / "error_gallery.png", dpi=130)
    plt.close(fig)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["OK", "Defective"],
           yticklabels=["OK", "Defective"], xlabel="Predicted", ylabel="Actual", title="Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    # Brightness and contrast strata expose aggregate failure concentrations.
    all_stats = []
    for row in rows:
        im = np.asarray(read_rgb(Path(root) / row["path"]).convert("L"), dtype=float)
        all_stats.append([im.mean(), im.std()])
    all_stats = np.asarray(all_stats)
    slices = {}
    for col, name in enumerate(["brightness", "contrast"]):
        bounds = np.quantile(all_stats[:, col], [1/3, 2/3])
        buckets = np.digitize(all_stats[:, col], bounds)
        for bucket, label in enumerate(["low", "medium", "high"]):
            mask = buckets == bucket
            slices[f'{name}_{label}'] = dict(count=int(mask.sum()), errors=int(((pred != y) & mask).sum()),
                                           false_negatives=int(((~pred) & (y == 1) & mask).sum()))
    (output / "error_slices.json").write_text(json.dumps(slices, indent=2))
    return review
