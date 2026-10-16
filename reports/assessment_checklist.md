# Assessment evidence

The five-page assessment is addressed by the artifacts below. Live deployment verification is recorded separately so reachability is not inferred from source code.

| Requirement | Evidence |
|---|---|
| Public non-trivial CV dataset and rationale | README: industrial cast-part binary classification; upstream dataset link |
| Dataset size, labels, class counts | `data_audit.json`, `manifest.csv`, README tables |
| Preprocessing and augmentation | `casting_ml/data.py`, `casting_ml/train.py`, `config.json` |
| Imbalance and observed label problems | Class-weighted heads; duplicate/conflict/corruption audit; documented label uncertainty |
| Train at least two models | `tuning_trials.json`, `results.json`, `artifacts/baseline.npz`, `artifacts/model.pt` |
| Defensible transfer learning | Frozen pretrained encoder plus trained head; CPU/small-data rationale in technical report |
| Basic hyperparameter tuning | Eight fits, validation-only C and threshold selection in `validation_selection.json` |
| Compute awareness | 128-pixel frozen features, four CPU threads, 253.8-second full pipeline, cached features |
| Train/validation/test methodology | Disjoint decoded hashes, seeded stratification, preserved 715-image test, tests in `tests/test_methodology.py` |
| Appropriate metrics | Accuracy, balanced accuracy, macro F1, defect precision/recall/F2, ROC AUC, AP, Brier, bootstrap intervals |
| Quantitative and qualitative error analysis | `test_predictions.csv`, `misclassifications.json`, `error_slices.json`, `error_gallery.png`, `technical_report.md` |
| Live publicly accessible inference | Public GitHub Pages image-upload interface using the actual trained ONNX model; `deployment.md` and live browser verification |
| Deployment URL in README | README opening links |
| Organized engineering | Separate data/training/evaluation/inference/API modules; config, logging, validation, pinned dependencies |
| GitHub repository and explanation | Separate repository, README, technical report, reproducibility commands |
| Optional Docker inference service | Runtime-only non-root `Dockerfile`, health and predict routes |
| Optional walkthrough | `walkthrough.md` and live example controls |
| Additional compute/data ideas | Technical report and README future-work sections |

Limitations include missing physical-part IDs, possible transformed near duplicates, uncalibrated scores, and no external-domain test. The project does not claim full-network fine-tuning or production certification.
