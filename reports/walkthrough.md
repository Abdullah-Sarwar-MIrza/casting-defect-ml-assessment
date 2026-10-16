# Walkthrough

1. Open the public inference link in README. Select **Acceptable casting**, then **Defective casting**. The first run downloads the model and WebAssembly runtime; subsequent runs reuse the initialized session.
2. Upload a JPEG or PNG casting image. Observe the predicted class, defect score, decision threshold, and runtime. Images stay on the device.
3. Read the README results table: the baseline achieves 85.73% accuracy, and frozen ResNet-18 plus its trained head achieves 99.30%, with 2 missed defects and 3 false alarms.
4. Open the five-error gallery and technical report. The confidently missed defect illustrates why a high aggregate score does not eliminate inspection risk.
5. Inspect the split manifest, validation trials, and held-out predictions to trace the results. Run `python -m pytest -q` to check methodology and the actual API.
6. For REST inference, run the Docker commands in README and send an image to `/predict`; `/health` and `/docs` expose service health and API documentation.

The project folder and repository are separate from the older assessment. Only the external dataset and preinstalled Python packages were read for local execution; no old project code or trained artifacts were reused.
