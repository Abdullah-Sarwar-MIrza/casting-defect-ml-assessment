"""Content audit and reproducible split. The supplied test partition is held out."""
import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

LOG = logging.getLogger(__name__)
LABELS = {"ok_front": 0, "def_front": 1}


def read_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def prepare(dataset_root, output, seed=42, validation_fraction=0.2):
    root, output = Path(dataset_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, corrupt = [], []
    for partition in ("train", "test"):
        for label, target in LABELS.items():
            folder = root / partition / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing expected class folder: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                try:
                    raw = path.read_bytes()
                    im = read_rgb(path)
                    digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                    records.append(dict(path=path.relative_to(root).as_posix(), source=partition,
                                        label=label, target=target, width=im.width, height=im.height,
                                        file_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=digest))
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    corrupt.append(dict(path=str(path.relative_to(root)), reason=str(exc)))
    groups = defaultdict(list)
    for row in records:
        groups[row["pixel_sha256"]].append(row)
    conflicts = [rows for rows in groups.values() if len({r["target"] for r in rows}) > 1]
    conflict_hashes = {r["pixel_sha256"] for rows in conflicts for r in rows}
    test_hashes = {r["pixel_sha256"] for r in records if r["source"] == "test"}
    removed, unique_train, seen = [], [], set()
    for row in records:
        if row["pixel_sha256"] in conflict_hashes:
            row["split"] = "excluded_label_conflict"
            removed.append(row)
        elif row["source"] == "test":
            row["split"] = "test"
        elif row["pixel_sha256"] in test_hashes:
            row["split"] = "excluded_test_overlap"
            removed.append(row)
        elif row["pixel_sha256"] in seen:
            row["split"] = "excluded_train_duplicate"
            removed.append(row)
        else:
            seen.add(row["pixel_sha256"])
            unique_train.append(row)
    if len(unique_train) < 10:
        raise ValueError("Too few usable training images")
    train, val = train_test_split(unique_train, test_size=validation_fraction,
                                 stratify=[r["target"] for r in unique_train], random_state=seed)
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "validation"
    split_hashes = {s: {r["pixel_sha256"] for r in records if r["split"] == s}
                    for s in ("train", "validation", "test")}
    assert not (split_hashes["train"] & split_hashes["validation"])
    assert not (split_hashes["train"] & split_hashes["test"])
    assert not (split_hashes["validation"] & split_hashes["test"])
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(dataset_root_recorded_as="external; see config.json", total_images=len(records),
                   original_distribution={s: dict(Counter(r["label"] for r in records if r["source"] == s))
                                          for s in ("train", "test")},
                   split_distribution={s: dict(Counter(r["label"] for r in records if r["split"] == s))
                                       for s in ("train", "validation", "test")},
                   dimensions=dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
                   corrupt_files=corrupt, excluded_count=len(removed),
                   exclusions_by_reason=dict(Counter(r["split"] for r in removed)),
                   train_test_duplicate_groups=sum(len({r["source"] for r in g}) > 1 for g in groups.values()),
                   duplicate_groups=sum(len(g) > 1 for g in groups.values()),
                   test_duplicate_groups=sum(len([r for r in g if r["source"] == "test"]) > 1 for g in groups.values()),
                   conflicting_label_groups=[[r["path"] for r in g] for g in conflicts],
                   limitations=["No original physical-part or capture-session IDs are available.",
                                "Exact decoded-pixel duplicates are checked; transformed near duplicates cannot be ruled out.",
                                "The alternate 512x512 dataset is not mixed into this experiment.",
                                "Folder labels are assumed; visual review cannot certify every label."])
    (output / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Data audit: %s", json.dumps(summary))
    return records, summary


def image_tensor(path, size=128):
    """Canonical RGB, EXIF-aware, bilinear resize, float CHW in [0,1]."""
    image = read_rgb(path).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
