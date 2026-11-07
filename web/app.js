"use strict";

const form = document.querySelector("#reading-form");
const resultCard = document.querySelector("#result-card");
const errorSummary = document.querySelector("#error-summary");
const modelStatus = document.querySelector("#model-status");
const announcer = document.querySelector("#announcements");
const submitButton = form?.querySelector('button[type="submit"]');

const READING_FIELDS = [
  { key: "intake_pressure", id: "intake-pressure", label: "Intake pressure", unit: "psi", rule: "positive" },
  { key: "discharge_pressure", id: "discharge-pressure", label: "Discharge pressure", unit: "psi", rule: "positive" },
  { key: "motor_temperature", id: "motor-temperature", label: "Motor temperature", unit: "°F", rule: "temperature" },
  { key: "intake_temperature", id: "intake-temperature", label: "Intake temperature", unit: "°F", rule: "temperature" },
  { key: "motor_current", id: "motor-current", label: "Motor current", unit: "A", rule: "positive" },
  { key: "motor_voltage", id: "motor-voltage", label: "Motor voltage", unit: "V", rule: "positive" },
  { key: "vibration", id: "vibration", label: "Vibration", unit: "g", rule: "nonnegative" },
  { key: "drive_frequency", id: "drive-frequency", label: "Drive frequency", unit: "Hz", rule: "positive" },
  { key: "production_rate", id: "production-rate", label: "Production rate", unit: "bbl/d", rule: "nonnegative" },
];

const BAND_EXPLANATIONS = Object.freeze({
  low: "The score is below the elevated band cutoff. This does not mean the pump is healthy.",
  elevated: "The model score falls in the elevated band.",
  high: "The model score falls in the high band.",
});

let modelReady = false;
let pendingRequest = null;

function setupExpertMode() {
  const urlSetting = document.querySelector('meta[name="expert-mode-url"]');
  const configuredUrl = urlSetting?.content.trim() || "";
  if (!configuredUrl) return;

  let safeUrl;
  try {
    safeUrl = new URL(configuredUrl, window.location.href);
  } catch {
    return;
  }
  if (!new Set(["http:", "https:", "file:"]).has(safeUrl.protocol)) return;

  document.querySelectorAll(".expert-mode-link").forEach((link) => {
    link.href = safeUrl.href;
    link.hidden = false;
  });
}

function inputFor(field) {
  return document.getElementById(field.id);
}

function fieldErrorFor(field) {
  return document.getElementById(`${field.id}-error`);
}

function errorFor(field, value) {
  if (value.trim() === "") return "Enter a value.";

  const number = Number(value);
  if (!Number.isFinite(number)) return "Enter a finite number.";
  if (field.rule === "positive" && number <= 0) return "Enter a number greater than 0.";
  if (field.rule === "nonnegative" && number < 0) return "Enter 0 or more.";
  return "";
}

function validateField(field) {
  const input = inputFor(field);
  const message = errorFor(field, input.value);
  const error = fieldErrorFor(field);

  input.setAttribute("aria-invalid", message ? "true" : "false");
  input.setAttribute("aria-describedby", message
    ? `${field.id}-hint ${field.id}-error`
    : `${field.id}-hint`);
  input.closest(".field").classList.toggle("field--invalid", Boolean(message));
  error.textContent = message;
  error.hidden = !message;
  return message;
}

function clearErrorSummary() {
  errorSummary.replaceChildren();
  errorSummary.hidden = true;
  delete errorSummary.dataset.signature;
}

function announce(message) {
  if (!announcer) return;
  announcer.textContent = "";
  window.requestAnimationFrame(() => {
    announcer.textContent = message;
  });
}

function showErrorSummary(invalidFields) {
  const signature = invalidFields
    .map((field) => `${field.key}:${errorFor(field, inputFor(field).value)}`)
    .join("|");
  if (!errorSummary.hidden && signature === errorSummary.dataset.signature) return;

  const message = document.createElement("p");
  message.textContent = `${invalidFields.length} ${invalidFields.length === 1 ? "reading needs" : "readings need"} attention. Fix the marked fields.`;

  const links = document.createElement("ul");
  for (const field of invalidFields) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#${field.id}`;
    link.textContent = `${field.label}: ${errorFor(field, inputFor(field).value)}`;
    item.append(link);
    links.append(item);
  }

  errorSummary.replaceChildren(message, links);
  errorSummary.dataset.signature = signature;
  errorSummary.hidden = false;
}

function iconMarkup(kind) {
  const icons = {
    neutral: '<circle cx="12" cy="12" r="9"></circle>',
    elevated: '<circle cx="12" cy="12" r="9"></circle><path d="M12 7v6"></path><path d="M12 16h.01"></path>',
    high: '<path d="M12 3 22 20H2L12 3Z"></path><path d="M12 9v5"></path><path d="M12 17h.01"></path>',
    error: '<circle cx="12" cy="12" r="9"></circle><path d="M12 7v6"></path><path d="M12 16h.01"></path>',
    loading: '<circle cx="12" cy="12" r="9" stroke-dasharray="42 14"></circle><path d="M21 12a9 9 0 0 0-9-9"></path>',
  };
  const icon = kind === "low" ? "neutral" : kind;
  const content = icons[icon] || icons.neutral;
  const className = `icon-${icon}`;
  return `<svg class="result-icon ${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" focusable="false">${content}</svg>`;
}

function renderResult({ state, label, copy, score, detail }) {
  const resultClass = state ? ` result-band-${state}` : "";
  resultCard.className = `result-card${resultClass}`;
  resultCard.dataset.state = state || "empty";
  resultCard.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  resultCard.removeAttribute("aria-live");
  resultCard.removeAttribute("aria-atomic");

  const row = document.createElement("div");
  row.className = "result-heading";
  const iconHolder = document.createElement("span");
  iconHolder.innerHTML = iconMarkup(state || "neutral");
  row.append(iconHolder.firstElementChild);
  if (label) {
    const heading = document.createElement("h3");
    heading.textContent = label;
    row.append(heading);
  }
  resultCard.replaceChildren(row);

  if (score !== undefined && score !== null) {
    const scoreLine = document.createElement("p");
    scoreLine.className = "result-score result-score--small";
    scoreLine.textContent = `Model score ${score} of 100. This is not a probability.`;
    resultCard.append(scoreLine);
  }
  if (copy) {
    const copyLine = document.createElement("p");
    copyLine.className = "result-copy";
    copyLine.textContent = copy;
    resultCard.append(copyLine);
  }
  if (detail) {
    const detailLine = document.createElement("p");
    detailLine.className = "result-detail";
    detailLine.textContent = detail;
    resultCard.append(detailLine);
  }
}

function showSuccessfulResult(result, announcement) {
  renderResult(result);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  resultCard.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "nearest" });
  announce(announcement);
}

function readValues() {
  return Object.fromEntries(READING_FIELDS.map((field) => [field.key, Number(inputFor(field).value)]));
}

function modelErrorMessage(payload) {
  if (typeof payload?.detail === "string") return payload.detail;
  return "The model could not check these readings. Try again.";
}

function formatRangeValue(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
}

function displayPrediction(result) {
  const band = String(result.band || "").toLowerCase();
  const labels = { low: "Low", elevated: "Elevated", high: "High" };
  if (!labels[band] || !Number.isFinite(Number(result.score))) {
    throw new Error("The model returned an invalid result. Check the saved model and restart the service.");
  }

  const label = labels[band];
  const formattedScore = new Intl.NumberFormat().format(Math.round(Number(result.score)));
  const details = (result.out_of_range || []).flatMap((item) => {
    const field = READING_FIELDS.find((candidate) => candidate.key === item.key);
    if (!field || !Number.isFinite(Number(item.min)) || !Number.isFinite(Number(item.max))) return [];
    const unit = field.unit ? ` ${field.unit}` : "";
    return [`${field.label} is outside the model training data range (${formatRangeValue(item.min)}–${formatRangeValue(item.max)}${unit}). This is not an equipment limit.`];
  });
  const detail = details.length ? details.join(" ") : "";
  const announcement = `Reading check complete. ${label}. Model score ${formattedScore} of 100.${details.length ? ` ${details.join(" ")}` : ""}`;

  showSuccessfulResult({
    state: band,
    label,
    score: formattedScore,
    copy: BAND_EXPLANATIONS[band],
    detail,
  }, announcement);
}

async function setupModelService() {
  if (!form || !modelStatus || !submitButton) return;

  submitButton.disabled = true;
  modelStatus.textContent = "Connecting to the saved model…";
  try {
    const response = await fetch("/api/config", { headers: { Accept: "application/json" }, cache: "no-store" });
    const config = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(config.message || "The model service returned an error.");

    if (!config.available) {
      modelStatus.textContent = config.message || "The saved model is unavailable. Train and save a model, then restart this service.";
    } else {
      const rangeCount = Object.keys(config.training_ranges || {}).length;
      const rangeStatus = rangeCount
        ? `Training range notes are available for ${rangeCount} readings; they are not equipment limits.`
        : "Training range notes are unavailable for this model.";
      modelStatus.textContent = `Using the saved ${config.model_label || "ESP"} model. ${rangeStatus}`;
      modelReady = true;
    }
  } catch (error) {
    modelReady = false;
    modelStatus.textContent = error instanceof TypeError
      ? "The saved model service is unavailable. Start it with python -m uvicorn api:app --reload, then open http://127.0.0.1:8000."
      : error.message || "The saved model is unavailable. Train and save a model, then restart this service.";
  }
  submitButton.disabled = !modelReady;
}

let submitAttempted = false;

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!modelReady) return;
    submitAttempted = true;
    pendingRequest?.abort();
    clearErrorSummary();

    const invalidFields = READING_FIELDS.filter((field) => validateField(field));
    if (invalidFields.length) {
      showErrorSummary(invalidFields);
      renderResult({ state: "error", label: "Can’t check these readings", copy: "Fix the marked fields, then select Check readings." });
      announce(`${invalidFields.length} readings need attention. Fix the marked fields.`);
      inputFor(invalidFields[0]).focus();
      return;
    }

    const readings = readValues();
    const controller = new AbortController();
    pendingRequest = controller;
    submitButton.disabled = true;
    renderResult({ state: "loading", label: "Checking these readings…" });
    announce("Checking these readings…");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(readings),
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 503) {
          modelReady = false;
          modelStatus.textContent = modelErrorMessage(payload);
        }
        throw new Error(modelErrorMessage(payload));
      }
      displayPrediction(payload);
    } catch (error) {
      if (error.name === "AbortError") return;
      const copy = error instanceof TypeError
        ? "The model service could not be reached. Restart it and refresh this page."
        : error.message || "The model service is unavailable. Try again.";
      if (error instanceof TypeError) {
        modelReady = false;
        modelStatus.textContent = "The saved model service is unavailable. Start it with python -m uvicorn api:app --reload, then refresh this page.";
      }
      renderResult({ state: "error", label: "Could not check these readings", copy });
      announce(`Reading check failed. ${copy}`);
    } finally {
      if (pendingRequest === controller) {
        pendingRequest = null;
        submitButton.disabled = !modelReady;
      }
    }
  });

  form.addEventListener("blur", (event) => {
    const field = READING_FIELDS.find((candidate) => inputFor(candidate) === event.target);
    if (!field) return;
    if (submitAttempted || event.target.value.trim() !== "") {
      validateField(field);
      if (!submitAttempted) {
        const invalidFields = READING_FIELDS.filter((candidate) => {
          const value = inputFor(candidate).value;
          return (submitAttempted || value.trim() !== "") && Boolean(errorFor(candidate, value));
        });
        if (invalidFields.length) showErrorSummary(invalidFields);
        else clearErrorSummary();
      }
    }
  }, true);

  form.addEventListener("input", (event) => {
    const field = READING_FIELDS.find((candidate) => inputFor(candidate) === event.target);
    if (!field) return;

    if (resultCard.dataset.state === "loading") {
      pendingRequest?.abort();
      pendingRequest = null;
      submitButton.disabled = !modelReady;
      renderResult({ state: "empty", label: "No result yet", copy: "Readings changed. Check them again to refresh the result." });
    }
    if (["low", "elevated", "high"].includes(resultCard.dataset.state)) {
      renderResult({ state: "empty", label: "No result yet", copy: "Readings changed. Check them again to refresh the result." });
    }
    if (!submitAttempted) return;

    validateField(field);
    const invalidFields = READING_FIELDS.filter((candidate) => errorFor(candidate, inputFor(candidate).value));
    if (invalidFields.length) {
      showErrorSummary(invalidFields);
    } else {
      clearErrorSummary();
      if (resultCard.dataset.state === "error") {
        renderResult({ state: "empty", label: "No result yet", copy: "Readings updated. Check them again to refresh the result." });
      }
    }
  });
}

setupExpertMode();
setupModelService();
