document.addEventListener("DOMContentLoaded", () => {
  const modelOptions = document.querySelectorAll("#model-toggle .model-option");
  let selectedModel = document.querySelector("#model-toggle .model-option.selected")?.dataset.modelKey;

  modelOptions.forEach((opt) => {
    opt.addEventListener("click", () => {
      modelOptions.forEach((o) => o.classList.remove("selected"));
      opt.classList.add("selected");
      selectedModel = opt.dataset.modelKey;
    });
  });

  const searchBtn = document.getElementById("search-btn");
  const conditionInput = document.getElementById("condition-input");
  const searchStatus = document.getElementById("search-status");
  const searchResults = document.getElementById("search-results");
  const judgeSection = document.getElementById("judge-section");
  const judgeStatus = document.getElementById("judge-status");
  const judgeResult = document.getElementById("judge-result");
  const patientTextEl = document.getElementById("patient-text");

  searchBtn.addEventListener("click", async () => {
    const condition = conditionInput.value.trim();
    if (!condition) {
      searchStatus.textContent = "Enter a condition to search.";
      return;
    }
    searchStatus.textContent = "Searching ClinicalTrials.gov…";
    searchResults.innerHTML = "";
    searchBtn.disabled = true;

    try {
      const resp = await fetch("/api/live_search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Search failed");

      if (!data.results.length) {
        searchStatus.textContent = `No currently-recruiting trials found for "${condition}".`;
        return;
      }
      searchStatus.textContent = `${data.results.length} currently-recruiting trials found.`;

      data.results.forEach((trial) => {
        const card = document.createElement("div");
        card.className = "verdict-card";
        card.innerHTML = `
          <div class="verdict-card-head">
            <span class="verdict-card-name">${escapeHtml(trial.title)}</span>
          </div>
          <div class="verdict-citation">${escapeHtml(trial.nct_id)} — ${escapeHtml((trial.conditions || []).join(", "))}</div>
          <button class="btn btn-ghost" style="margin-top:10px;">Judge eligibility against this trial</button>
        `;
        card.querySelector("button").addEventListener("click", () => runJudge(trial));
        searchResults.appendChild(card);
      });
    } catch (err) {
      searchStatus.textContent = `Error: ${err.message}`;
    } finally {
      searchBtn.disabled = false;
    }
  });

  async function runJudge(trial) {
    const patientText = patientTextEl.value.trim();
    if (!patientText) {
      alert("Describe the patient first.");
      return;
    }
    judgeSection.style.display = "";
    judgeStatus.textContent = `Judging with ${selectedModel.replace("_", " ")}…`;
    judgeResult.innerHTML = "";

    try {
      const resp = await fetch("/api/live_judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_text: patientText,
          trial_title: trial.title,
          inclusion_criteria: trial.inclusion_criteria,
          exclusion_criteria: trial.exclusion_criteria,
          model_config: selectedModel,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Judge call failed");

      judgeStatus.textContent = "";
      judgeResult.innerHTML = `
        <div class="verdict-card">
          <div class="verdict-card-head">
            <span class="verdict-card-name">${escapeHtml(trial.title)}</span>
            <span class="badge badge-${data.label}">${data.label}</span>
          </div>
          <div class="verdict-citation">“${escapeHtml(data.cited_criterion)}”</div>
          <div class="verdict-rationale">${escapeHtml(data.rationale)}</div>
        </div>
      `;
    } catch (err) {
      judgeStatus.textContent = `Error: ${err.message}`;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }
});
