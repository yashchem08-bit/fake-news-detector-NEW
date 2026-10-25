/**
 * Member D — Frontend Developer
 * ===============================
 * Handles: tab switching (text vs URL), talking to the backend
 * (Member C)'s /predict and /predict-url endpoints, and animating
 * the verdict stamp + confidence gauge.
 */

// Point this at your deployed backend when you host it.
const API_BASE_URL = "http://localhost:8000";

const tabs = document.querySelectorAll(".tab");
const textInput = document.getElementById("text-input");
const urlInput = document.getElementById("url-input");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const charCount = document.getElementById("char-count");
const errorMsg = document.getElementById("error-msg");

const resultCard = document.getElementById("result");
const stampEl = document.getElementById("stamp");
const needleEl = document.getElementById("needle");
const tickerConfidence = document.getElementById("ticker-confidence");
const tickerMeta = document.getElementById("ticker-meta");

let mode = "text"; // "text" | "url"

// ── Tab switching ──────────────────────────────────────────

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    mode = tab.dataset.mode;
    const isText = mode === "text";
    textInput.style.display = isText ? "block" : "none";
    urlInput.style.display = isText ? "none" : "block";
    hideError();
  });
});

textInput.addEventListener("input", () => {
  charCount.textContent = `${textInput.value.length} characters`;
});

// ── Submit ─────────────────────────────────────────────────

submitBtn.addEventListener("click", async () => {
  hideError();

  const isText = mode === "text";
  const value = (isText ? textInput.value : urlInput.value).trim();

  if (!value) {
    showError(isText ? "Paste some article text first." : "Paste a URL first.");
    return;
  }

  if (!isText && !isValidUrl(value)) {
    showError("That doesn't look like a valid URL.");
    return;
  }

  setLoading(true);

  try {
    const endpoint = isText ? "/predict" : "/predict-url";
    const body = isText ? { text: value } : { url: value };

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Something went wrong verifying this.");
    }

    showResult(data);
  } catch (err) {
    showError(
      err.message === "Failed to fetch"
        ? "Couldn't reach the verification backend. Is it running?"
        : err.message
    );
  } finally {
    setLoading(false);
  }
});

// ── Helpers ────────────────────────────────────────────────

function isValidUrl(str) {
  try {
    new URL(str);
    return true;
  } catch {
    return false;
  }
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "VERIFYING…" : "RUN VERIFICATION";
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
  resultCard.hidden = true;
}

function hideError() {
  errorMsg.hidden = true;
}

function showResult(data) {
  const { label, confidence, elapsed_seconds: elapsed } = data;
  const isReal = label === "REAL";

  // Re-trigger the stamp animation even if the same label appears twice in a row
  resultCard.hidden = false;
  stampEl.className = "stamp"; // reset
  void stampEl.offsetWidth; // force reflow so the animation restarts
  stampEl.classList.add(isReal ? "real" : "fake");
  stampEl.textContent = isReal ? "VERIFIED" : "FLAGGED";

  // score = how "real" the result is, from 0 (certainly fake) to 1 (certainly real)
  const score = isReal ? confidence : 1 - confidence;
  const angle = (score - 0.5) * 160; // -80deg (fake) … 0 (uncertain) … +80deg (real)
  needleEl.setAttribute("transform", `rotate(${angle} 100 100)`);

  tickerConfidence.textContent = `CONFIDENCE — ${(confidence * 100).toFixed(1)}%`;
  tickerMeta.textContent = `MODEL — baseline · PROCESSED IN ${elapsed?.toFixed(2) ?? "0.00"}s`;

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
