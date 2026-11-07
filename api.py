"""FastAPI service for the static ESP reading check."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.prediction import ModelNotFoundError, PredictionError, load_model, predict_esp


WEB_DIR = Path(__file__).resolve().parent / "web"
DESIGN_SYSTEM_DIR = Path(__file__).resolve().parent / "design-system"
LOGGER = logging.getLogger(__name__)

# Browser field names stay plain-language; this map adapts them to the model's
# canonical feature names.
UI_FEATURE_MAP = {
    "intake_pressure": "intake_pressure",
    "discharge_pressure": "discharge_pressure",
    "motor_temperature": "motor_temperature",
    "intake_temperature": "intake_temperature",
    "motor_current": "motor_amps",
    "motor_voltage": "motor_volt",
    "vibration": "vibration",
    "drive_frequency": "frequency",
    "production_rate": "production",
}


class ReadingRequest(BaseModel):
    intake_pressure: float
    discharge_pressure: float
    motor_temperature: float
    intake_temperature: float
    motor_current: float
    motor_voltage: float
    vibration: float
    drive_frequency: float
    production_rate: float

    class Config:
        extra = "forbid"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _training_ranges(bundle: dict[str, Any]) -> dict[str, dict[str, float]]:
    stats = bundle.get("training_stats", {}) or {}
    model_features = set(bundle.get("features", []))
    output: dict[str, dict[str, float]] = {}
    for ui_key, feature in UI_FEATURE_MAP.items():
        if feature not in model_features:
            continue
        feature_stats = stats.get(feature, {}) or {}
        minimum = _finite_number(feature_stats.get("min"))
        maximum = _finite_number(feature_stats.get("max"))
        if minimum is not None and maximum is not None:
            output[ui_key] = {"min": minimum, "max": maximum}
    return output


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = load_model()
        features = bundle.get("features")
        if (
            not isinstance(features, (list, tuple))
            or not features
            or any(not isinstance(feature, str) for feature in features)
        ):
            raise PredictionError("The saved model does not contain a feature list.")
        unsupported = sorted(set(features) - set(UI_FEATURE_MAP.values()))
        if unsupported:
            raise PredictionError(
                "The saved model uses features that the reading form does not provide."
            )
        app.state.bundle = bundle
        app.state.model_message = ""
    except (ModelNotFoundError, PredictionError) as exc:
        app.state.bundle = None
        app.state.model_message = (
            "The saved model is unavailable. Train and save a model, then restart this service."
        )
        LOGGER.warning("ESP reading model is unavailable: %s", exc)
    yield
    app.state.bundle = None


app = FastAPI(title="ESP PredictAI reading check", lifespan=lifespan)
app.state.bundle = None
app.state.model_message = "The saved model is unavailable. Start the ESP API service."


@app.get("/api/config")
def get_config(request: Request) -> dict[str, Any]:
    bundle = request.app.state.bundle
    if bundle is None:
        return {"available": False, "message": request.app.state.model_message}
    return {
        "available": True,
        "model_label": bundle.get("model_label", "Saved model"),
        "trained_at": bundle.get("trained_at"),
        "training_ranges": _training_ranges(bundle),
    }


@app.post("/api/predict")
def create_prediction(readings: ReadingRequest, request: Request) -> dict[str, Any]:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_message)

    submitted = readings.dict()
    model_features = set(bundle.get("features", []))
    model_inputs = {
        feature: submitted[ui_key]
        for ui_key, feature in UI_FEATURE_MAP.items()
        if feature in model_features
    }
    try:
        prediction = predict_esp(bundle, model_inputs)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    out_of_range = []
    model_to_ui = {feature: ui_key for ui_key, feature in UI_FEATURE_MAP.items()}
    for feature, values in prediction["out_of_range"].items():
        ui_key = model_to_ui.get(feature)
        if ui_key is None:
            continue
        out_of_range.append(
            {
                "key": ui_key,
                "value": values["value"],
                "min": values["min"],
                "max": values["max"],
            }
        )

    band = "Low" if prediction["risk_level"] == "Normal" else prediction["risk_level"]
    return {
        "band": band,
        "score": prediction["probability"] * 100,
        "out_of_range": out_of_range,
    }


# Keep API and shared design-system routes ahead of the catch-all web mount.
app.mount(
    "/design-system",
    StaticFiles(directory=DESIGN_SYSTEM_DIR),
    name="design-system",
)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
