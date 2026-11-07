"use strict";

const form = document.querySelector("#reading-form");
const resultCard = document.querySelector("#result-card");
const errorSummary = document.querySelector("#error-summary");
const previewBanner = document.querySelector("#preview-banner");
const announcer = document.querySelector("#announcements");

// MOCK - replaced in Milestone 4
if (previewBanner) previewBanner.textContent = "Preview: scores here are placeholders.";

const READING_FIELDS = [
  { key: "intake_pressure", id: "intake-pressure", label: "Intake pressure", rule: "positive" },
  { key: "discharge_pressure", id: "discharge-pressure", label: "Discharge pressure", rule: "positive" },
  { key: "motor_temperature", id: "motor-temperature", label: "Motor temperature", rule: "temperature" },
  { key: "intake_temperature", id: "intake-temperature", label: "Intake temperature", rule: "temperature" },
  { key: "motor_current", id: "motor-current", label: "Motor current", rule: "positive" },
  { key: "motor_voltage", id: "motor-voltage", label: "Motor voltage", rule: "positive" },
  { key: "vibration", id: "vibration", label: "Vibration", rule: "nonnegative" },
  { key: "drive_frequency", id: "drive-frequency", label: "Drive frequency", rule: "positive" },
  { key: "production_rate", id: "production-rate", label: "Production rate", rule: "nonnegative" },
];

// MOCK - replaced in Milestone 4
const MOCK_TRAINING_RANGES = Object.freeze({
  intake_pressure: { min: 1, max: 3000 },
  discharge_pressure: { min: 1, max: 5000 },
  motor_temperature: { min: -40, max: 350 },
  intake_temperature: { min: -40, max: 350 },
  motor_current: { min: 0.1, max: 150 },
  motor_voltage: { min: 1, max: 1000 },
  vibration: { min: 0, max: 20 },
  drive_frequency: { min: 1, max: 120 },
  production_rate: { min: 0, max: 30000 },
});

// MOCK - replaced in Milestone 4
const MOCK_BAND_CUTOFFS = Object.freeze({ elevated: 0.4, high: 0.7 });

const BAND_EXPLANATIONS = Object.freeze({
  low: "These readings resemble the model's running examples more than its failed examples. This does not mean the pump is healthy.",
  elevated: "These readings sit between the model's running and failed examples.",
  high: "These readings resemble the model's failed examples more than its running examples.",
});

// MOCK - replaced in Milestone 4
function mockScore(readings) {
  const normalized = READING_FIELDS.map(({ key }) => {
    const { min, max } = MOCK_TRAINING_RANGES[key];
    return Math.min(1, Math.max(0, (readings[key] - min) / (max - min)));
  });
  const average = normalized.reduce((sum, value) => sum + value, 0) / normalized.length;
  return Math.min(1, Math.max(0, average));
}

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
    range: '<path d="M4 8h11M12 5l3 3-3 3M20 16H9m3-3-3 3 3 3"></path>',
  };
  const icon = kind === "low" ? "neutral" : kind === "out-of-range" ? "range" : kind;
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

function findOutOfRange(readings) {
  return READING_FIELDS.filter(({ key }) => {
    const { min, max } = MOCK_TRAINING_RANGES[key];
    return readings[key] < min || readings[key] > max;
  });
}

function bandFor(score) {
  if (score >= MOCK_BAND_CUTOFFS.high) return "high";
  if (score >= MOCK_BAND_CUTOFFS.elevated) return "elevated";
  return "low";
}

function displayBand(readings) {
  const score = mockScore(readings);
  const band = bandFor(score);
  const formattedScore = new Intl.NumberFormat().format(Math.round(score * 100));
  const label = band === "low" ? "Low" : band === "elevated" ? "Elevated" : "High";
  showSuccessfulResult({ state: band, label, score: formattedScore, copy: BAND_EXPLANATIONS[band] },
    `Reading check complete. ${label}. Model score ${formattedScore} of 100.`);
}

let submitAttempted = false;
let pendingCheck = 0;

if (form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitAttempted = true;
    window.clearTimeout(pendingCheck);
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
    const outOfRange = findOutOfRange(readings);
    renderResult({ state: "loading", label: "Checking these readings…" });
    announce("Checking these readings…");

    pendingCheck = window.setTimeout(() => {
      if (outOfRange.length) {
        const names = outOfRange.map((field) => field.label);
        const note = names.length === 1
          ? `${names[0]} is outside the preview’s mock training data range. This is not an equipment limit.`
          : `${names.join(", ")} are outside the preview’s mock training data range. This is not an equipment limit.`;
        showSuccessfulResult({ state: "out-of-range", label: "Outside training data range", copy: note },
          `Reading check complete. ${note}`);
        return;
      }
      displayBand(readings);
    }, 450);
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
      window.clearTimeout(pendingCheck);
      renderResult({ state: "empty", label: "No result yet", copy: "Readings changed. Check them again to refresh the result." });
    }
    if (["low", "elevated", "high", "out-of-range"].includes(resultCard.dataset.state)) {
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
