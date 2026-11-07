# Static reading check and model API

The static pages in `web/` now send one-day readings to the saved model through `api.py`. Start the service from the project root:

```powershell
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000`. The service loads `models/esp_predictai_model.joblib` once at startup, serves the static pages, and exposes two same-origin endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | Reports whether the saved model loaded, its label, and the per-feature minimum and maximum values from its training data. |
| `POST /api/predict` | Validates the nine submitted readings, runs the existing prediction pipeline, and returns the 0–100 model score, display band, and any training-range notes. |

The browser does not calculate scores or hold fallback ranges. It disables submission if the service or saved model is unavailable. Replacing the saved model requires restarting the API service. The range notes describe training data only and are not equipment limits. The displayed score is not a calibrated probability, future-failure forecast, or operating recommendation.

The API maps the plain-language form fields to the model’s canonical feature names. It does not persist readings or log request bodies. The Streamlit application continues to run separately with `streamlit run app.py`.

The browser was not visually or interactively reviewed for this milestone. Static code changes do not establish rendered layout, screen-reader behavior, or keyboard interaction quality.
