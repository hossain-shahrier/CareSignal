const form = document.getElementById("predict-form");
const resultPanel = document.getElementById("result-panel");
const resultEmpty = document.getElementById("result-empty");
const riskScoreEl = document.getElementById("risk-score");
const riskLabelEl = document.getElementById("risk-label");
const riskBarEl = document.getElementById("risk-bar");
const riskMeterEl = document.getElementById("risk-meter");
const riskBadgeEl = document.getElementById("risk-badge");
const thresholdNoteEl = document.getElementById("threshold-note");
const resultFactsEl = document.getElementById("result-facts");
const statusPill = document.getElementById("status-pill");
const modelVersionEl = document.getElementById("model-version");
const submitBtn = document.getElementById("submit-btn");
const modelMetricsEl = document.getElementById("model-metrics");
const featureListEl = document.getElementById("feature-list");

const FIELD_LABELS = {
  age: "Age at discharge",
  sex: "Sex",
  length_of_stay_days: "Length of stay",
  prior_admissions: "Prior hospital stays",
  condition_count: "Health problems",
  procedure_count: "Procedures this stay",
  days_since_last_discharge: "Days since last discharge",
};

const PRESETS = {
  // Tuned to patterns the Synthea-trained model scores high/low (short LOS, utilization).
  high: {
    age: 78,
    sex: "male",
    length_of_stay_days: 1,
    prior_admissions: 3,
    condition_count: 25,
    procedure_count: 3,
    days_since_last_discharge: 14,
  },
  low: {
    age: 32,
    sex: "female",
    length_of_stay_days: 1,
    prior_admissions: 0,
    condition_count: 2,
    procedure_count: 0,
    days_since_last_discharge: 0,
  },
};

let modelInfo = null;

function switchView(viewId) {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewId);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${viewId}`);
    panel.hidden = panel.id !== `view-${viewId}`;
  });

  const titles = {
    assess: ["Risk assessment", "Estimate 30-day inpatient readmission probability at discharge."],
    model: ["Model & performance", "Hold-out metrics and feature contract for the loaded bundle."],
    guide: ["How it works", "Pipeline overview, label definition, and limitations."],
  };
  const [title, subtitle] = titles[viewId] || titles.assess;
  document.getElementById("view-title").textContent = title;
  document.getElementById("view-subtitle").textContent = subtitle;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusPill.textContent = `Online · v${data.model_version}`;
    statusPill.classList.add("ok");
    modelVersionEl.textContent = `CareSignal v${data.model_version}`;
    return data;
  } catch {
    statusPill.textContent = "API offline";
    statusPill.classList.add("error");
    return null;
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch("/info");
    if (!res.ok) return;
    modelInfo = await res.json();
    renderModelMetrics();
    renderFeatureList();
    updateThresholdNote();
  } catch {
    modelMetricsEl.innerHTML =
      '<div class="metric-card skeleton">Could not load /info — restart the API server.</div>';
  }
}

function renderModelMetrics() {
  if (!modelInfo) return;
  const m = modelInfo.metrics || {};
  const fmt = (v, pct = false) => {
    if (v == null) return "—";
    return pct ? `${(v * 100).toFixed(1)}%` : Number(v).toFixed(3);
  };

  const items = [
    ["Hold-out ROC-AUC", fmt(m.roc_auc)],
    ["Hold-out F1", fmt(m.f1)],
    ["Cohort readmit rate", fmt(m.positive_rate, true)],
    ["Test patients", m.test_patients ?? "—"],
    ["Decision threshold", fmt(modelInfo.threshold)],
    ["Calibrated", m.calibrated ? "Yes" : "No"],
  ];

  modelMetricsEl.innerHTML = items
    .map(
      ([label, value]) =>
        `<div class="metric-card"><dt>${label}</dt><dd>${value}</dd></div>`
    )
    .join("");
}

function renderFeatureList() {
  if (!modelInfo?.features) return;
  featureListEl.innerHTML = modelInfo.features
    .map((f) => `<li><code>${f}</code> — ${FIELD_LABELS[f] || f}</li>`)
    .join("");
}

function readForm() {
  const data = new FormData(form);
  return {
    age: Number(data.get("age")),
    sex: data.get("sex"),
    length_of_stay_days: Number(data.get("length_of_stay_days")),
    prior_admissions: Number(data.get("prior_admissions")),
    condition_count: Number(data.get("condition_count")),
    procedure_count: Number(data.get("procedure_count")),
    days_since_last_discharge: Number(data.get("days_since_last_discharge")),
  };
}

function applyPreset(preset) {
  const values = PRESETS[preset];
  for (const [key, value] of Object.entries(values)) {
    const field = form.elements.namedItem(key);
    if (field) field.value = value;
  }
  switchView("assess");
}

function updateThresholdNote(extra = "") {
  const t = modelInfo?.threshold;
  const thresholdText =
    t != null
      ? `Decision threshold: <strong>${(t * 100).toFixed(0)}%</strong> (validation-tuned).`
      : "Decision threshold from trained manifest.";
  thresholdNoteEl.innerHTML = extra || `${thresholdText} Scores at or above this value are flagged as elevated readmission risk.`;
}

function showResult(payload, inputs) {
  const { risk_score, predicted_readmission } = payload;
  const pct = Math.round(risk_score * 1000) / 10;

  resultEmpty.hidden = true;
  resultPanel.hidden = false;

  riskScoreEl.textContent = `${pct.toFixed(1)}%`;
  riskBarEl.style.width = `${Math.min(100, pct)}%`;
  riskBarEl.classList.toggle("elevated", predicted_readmission);
  riskMeterEl.setAttribute("aria-valuenow", String(pct));

  riskBadgeEl.textContent = predicted_readmission ? "Elevated risk" : "Lower risk";
  riskBadgeEl.className = `risk-badge ${predicted_readmission ? "elevated" : "low"}`;

  riskLabelEl.textContent = predicted_readmission
    ? "Flagged for possible 30-day readmission"
    : "Below elevated-risk threshold";
  riskLabelEl.className = `risk-label ${predicted_readmission ? "elevated" : "low"}`;

  const threshold = modelInfo?.threshold ?? 0.5;
  const gap = (risk_score - threshold) * 100;

  resultFactsEl.innerHTML = `
    <li><span>vs threshold</span><span>${gap >= 0 ? "+" : ""}${gap.toFixed(1)} pp</span></li>
    <li><span>Age</span><span>${inputs.age} yrs</span></li>
    <li><span>Prior admissions</span><span>${inputs.prior_admissions}</span></li>
    <li><span>Days since last DC</span><span>${inputs.days_since_last_discharge}</span></li>
  `;

  updateThresholdNote();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const inputs = readForm();
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const msg = Array.isArray(err.detail)
        ? err.detail.map((d) => d.msg).join("; ")
        : err.detail
          ? JSON.stringify(err.detail)
          : `HTTP ${res.status}`;
      throw new Error(msg);
    }
    showResult(await res.json(), inputs);
  } catch (error) {
    resultEmpty.hidden = true;
    resultPanel.hidden = false;
    riskBadgeEl.textContent = "Error";
    riskBadgeEl.className = "risk-badge elevated";
    riskLabelEl.textContent = "Could not complete assessment";
    riskLabelEl.className = "risk-label elevated";
    thresholdNoteEl.textContent = String(error.message || error);
    resultFactsEl.innerHTML = "";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run readmission risk score";
  }
});

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

document.getElementById("preset-high").addEventListener("click", () => applyPreset("high"));
document.getElementById("preset-low").addEventListener("click", () => applyPreset("low"));

checkHealth().then(() => loadModelInfo());
