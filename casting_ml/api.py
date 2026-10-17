"""REST endpoints for model discovery and image classification."""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from typing import Literal
from .inference import Predictor, MAX_BYTES

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor(os.getenv("MODEL_PATH", "docs/model.onnx"),
                                    os.getenv("METADATA_PATH", "artifacts/metadata.json"),
                                    os.getenv("BASELINE_PATH", "artifacts/baseline.npz"))
    LOG.info("Model loaded and warm-up succeeded")
    yield


app = FastAPI(title="Casting defect inference", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.predictor.metadata["architecture"]}


@app.get("/models")
def models():
    return {"models": [{"id": key, **value} for key, value in app.state.predictor.models.items()]}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  model: Literal["resnet18_frozen_logistic", "pixels_logistic"] = Query("resnet18_frozen_logistic")):

    try:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB limit")
        started = time.perf_counter()
        result = await asyncio.to_thread(app.state.predictor.predict_bytes, data, model)
        result["inference_ms"] = round((time.perf_counter() - started) * 1000, 2)
        LOG.info("Prediction label=%s latency_ms=%s", result["label"], result["inference_ms"])
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
