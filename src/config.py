"""Central configuration for ESP PredictAI.

Keeping constants in one place avoids drift between the data, modelling,
and presentation layers.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ESP PredictAI"
APP_SUBTITLE = "Intelligent Artificial Lift Monitoring & Failure Prediction"
APP_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_CSV_PATH = DATA_DIR / "esp_downhole_data.csv"
MODEL_PATH = MODELS_DIR / "esp_predictai_model.joblib"

GITHUB_REPO_URL = "https://github.com/kings-2025/Electric-Submersible-Pumps"
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/kings-2025/Electric-Submersible-Pumps/"
    "main/esp_downhole_data.csv"
)
SOURCE_DATASET_NAME = "Electric-Submersible-Pumps / esp_downhole_data.csv"

TARGET_RAW = "esp_status"
TARGET = "esp_failure"
WELL_COL = "well_id"
DATE_COL = "date"

# Columns that must never be used as model inputs.  downtime_hours and
# production_loss are deterministic functions of the target in this dataset.
LEAKAGE_COLUMNS = ["downtime_hours", "production_loss"]
TARGET_COLUMNS = [TARGET_RAW, TARGET]
PROTECTED_COLUMNS = [TARGET_RAW, TARGET, WELL_COL, DATE_COL]

FEATURE_ORDER = [
    "intake_pressure",
    "discharge_pressure",
    "motor_temperature",
    "intake_temperature",
    "motor_amps",
    "motor_volt",
    "vibration",
    "frequency",
    "production",
]

FEATURE_LABELS = {
    "intake_pressure": "Intake pressure",
    "discharge_pressure": "Discharge pressure",
    "motor_temperature": "Motor temperature",
    "intake_temperature": "Intake temperature",
    "motor_amps": "Motor current",
    "motor_volt": "Motor voltage",
    "vibration": "Vibration",
    "frequency": "Frequency",
    "production": "Production rate",
}

FEATURE_UNITS = {
    "intake_pressure": "psi",
    "discharge_pressure": "psi",
    "motor_temperature": "°F",
    "intake_temperature": "°F",
    "motor_amps": "A",
    "motor_volt": "V",
    "vibration": "g",
    "frequency": "Hz",
    "production": "bbl/d",
}

FEATURE_ROLES = {
    "well_id": "Identifier — temporal / grouped validation only",
    "date": "Timestamp — temporal validation and trend analysis only",
    "downtime_hours": "Outcome / leakage — excluded from model inputs",
    "intake_pressure": "Operational feature",
    "discharge_pressure": "Operational feature",
    "motor_temperature": "Operational feature",
    "intake_temperature": "Operational feature",
    "motor_amps": "Operational feature",
    "motor_volt": "Operational feature",
    "vibration": "Operational feature",
    "frequency": "Operational feature",
    "production": "Operational feature",
    "production_loss": "Outcome / leakage — excluded from model inputs",
    "esp_status": "Raw target label (running / failed)",
    "esp_failure": "Binary model target (0 = running, 1 = failed)",
}

# Palette — deep navy industrial theme.
COLORS = {
    "bg": "#050d18",
    "panel": "#0b1a2d",
    "panel_2": "#0e2138",
    "border": "#17324f",
    "text": "#e6eef8",
    "muted": "#8aa4bf",
    "blue": "#2d7dd2",
    "cyan": "#22d3ee",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "purple": "#8b5cf6",
}

STATUS_COLORS = {"running": COLORS["green"], "failed": COLORS["red"]}
TARGET_COLORS = {0: COLORS["green"], 1: COLORS["red"]}
RISK_COLORS = {
    "Normal": COLORS["green"],
    "Elevated": COLORS["amber"],
    "High": COLORS["red"],
}

MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

SPLIT_LABELS = {
    "random": "Random stratified (baseline)",
    "time": "Time-based (last period held out)",
    "well": "Well-based (unseen wells held out)",
}

DATA_DICTIONARY = {
    "well_id": ("Identifier", "—", "Unique well / ESP identifier"),
    "date": ("Timestamp", "—", "Daily measurement date"),
    "downtime_hours": ("Outcome", "hours", "Downtime associated with a failure event"),
    "intake_pressure": ("Feature", "psi", "Pump intake / suction pressure"),
    "discharge_pressure": ("Feature", "psi", "Pump discharge pressure"),
    "motor_temperature": ("Feature", "°F", "ESP motor winding temperature"),
    "intake_temperature": ("Feature", "°F", "Well-fluid temperature at pump intake"),
    "motor_amps": ("Feature", "A", "ESP motor current"),
    "motor_volt": ("Feature", "V", "ESP motor voltage"),
    "vibration": ("Feature", "g", "Vibration reading"),
    "frequency": ("Feature", "Hz", "Operating frequency (VSD output)"),
    "production": ("Feature", "bbl/d", "Daily production"),
    "production_loss": ("Outcome", "bbl/d", "Calculated production loss during downtime"),
    "esp_status": ("Target (raw)", "—", "running / failed on the measurement day"),
    "esp_failure": ("Target (binary)", "—", "0 = running, 1 = failed"),
}

DISCLAIMER = (
    "ESP PredictAI is an educational and research prototype. It is not a certified "
    "industrial safety, diagnostic, or maintenance system. Model scores are not "
    "calibrated field failure probabilities and must not be used as the sole basis "
    "for operational decisions."
)
