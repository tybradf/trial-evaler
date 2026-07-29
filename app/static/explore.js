document.addEventListener("DOMContentLoaded", () => {
  const PAIRS = JSON.parse(document.getElementById("explorer-data").textContent);
  const MODELS = JSON.parse(document.getElementById("models-data").textContent);

  const SEGMENTS = [
    { key: "accurate_excluded", title: "Accurate — excluded", severity: "good",
      desc: "Judge correctly excluded a patient the physician excluded." },
    { key: "accurate_eligible", title: "Accurate — eligible", severity: "good",
      desc: "Judge correctly confirmed eligibility the physician confirmed." },
    { key: "missed_exclusion", title: "Missed exclusion", severity: "danger",
      desc: "Judge said eligible; physician said excluded. The dangerous error." },
    { key: "over_exclusion", title: "Over-exclusion", severity: "mild",
      desc: "Judge said excluded; physician said eligible. Wrong, but the safe direction." },
    { key: "abstained_excluded", title: "Abstained — was excluded", severity: "uncertain",
      desc: "Judge declined to answer; physician had excluded this patient." },
    { key: "abstained_eligible", title: "Abstained — was eligible", severity: "uncertain",
      desc: "Judge declined to answer; physician had confirmed eligibility." },
  ];

  function computeBucket(groundTruth, label) {
    if (label === "insufficient_information") {
      return groundTruth === "excluded" ? "abstained_excluded" : "abstained_eligible";
    }
    if (groundTruth === "excluded" && label === "excluded") return "accurate_excluded";
    if (groundTruth === "eligible" && label === "eligible") return "accurate_eligible";
    if (groundTruth === "excluded" && label === "eligible") return "missed_exclusion";
    if (groundTruth === "eligible" && label === "excluded") return "over_exclusion";
    return null;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  let selectedModel = document.querySelector("#explore-model-toggle .model-option.selected")?.dataset.modelKey;
  let selectedSegment = null;

  function bucketedPairs(model) {
    const buckets = {};
    SEGMENTS.forEach((s) => { buckets[s.key] = []; });
    PAIRS.forEach((pair) => {
      const verdict = pair.verdicts[model];
      if (!verdict) return; // this combo wasn't run for this pair -- skip, don't crash
      const bucket = computeBucket(pair.ground_truth, verdict.label);
      if (bucket) buckets[bucket].push(pair);
    });
    return buckets;
  }

  function renderSegments() {
    const buckets = bucketedPairs(selectedModel);
    const total = PAIRS.filter((p) => p.verdicts[selectedModel]).length;

    if (!selectedSegment || !buckets[selectedSegment]) {
      let maxKey = SEGMENTS[0].key;
      SEGMENTS.forEach((s) => {
        if (buckets[s.key].length > buckets[maxKey].length) maxKey = s.key;
      });
      selectedSegment = maxKey;
    }

    const grid = document.getElementById("segment-grid");
    grid.innerHTML = "";
    SEGMENTS.forEach((seg) => {
      const count = buckets[seg.key].length;
      const pct = total ? Math.round((count / total) * 100) : 0;
      const tile = document.createElement("button");
      tile.className = "segment-tile" + (seg.key === selectedSegment ? " selected" : "");
      tile.dataset.segmentKey = seg.key;
      tile.dataset.severity = seg.severity;
      tile.innerHTML = `
        <div class="segment-title">${seg.title}</div>
        <div><span class="segment-count">${count}</span> <span class="segment-count-suffix">/ ${total} (${pct}%)</span></div>
        <div class="segment-desc">${seg.desc}</div>
      `;
      tile.addEventListener("click", () => {
        selectedSegment = seg.key;
        renderSegments();
        renderCaseList();
      });
      grid.appendChild(tile);
    });

    renderCaseList();
  }

  function renderCaseList() {
    const buckets = bucketedPairs(selectedModel);
    const segMeta = SEGMENTS.find((s) => s.key === selectedSegment);
    const cases = buckets[selectedSegment] || [];

    document.getElementById("case-list-title").textContent = segMeta ? segMeta.title : "";
    document.getElementById("case-list-count").textContent = `${cases.length} case${cases.length === 1 ? "" : "s"}`;

    const listEl = document.getElementById("case-list");
    listEl.innerHTML = "";

    if (!cases.length) {
      listEl.innerHTML = `<div class="empty-state">No cases in this segment for the selected model.</div>`;
      return;
    }

    cases.forEach((pair) => {
      const verdict = pair.verdicts[selectedModel];
      const row = document.createElement("div");
      row.className = "case-row";

      const tagsHtml = (pair.tags || [])
        .map((t) => `<span class="tag-chip" title="${escapeHtml(t.note)}">${escapeHtml(t.tag.split(":")[0])}</span>`)
        .join("");

      row.innerHTML = `
        <div class="case-row-summary">
          <span class="case-row-chevron">▸</span>
          <span class="case-row-snippet">${escapeHtml(pair.patient_text.slice(0, 110))}…</span>
          <span class="case-row-trial">${escapeHtml(pair.trial_title)}</span>
          <div class="case-row-badges">
            ${tagsHtml}
            <span class="badge badge-${pair.ground_truth}">${pair.ground_truth}</span>
            <span class="badge badge-${verdict.label}">${verdict.label}</span>
          </div>
        </div>
        <div class="case-row-detail"></div>
      `;

      const summary = row.querySelector(".case-row-summary");
      const detail = row.querySelector(".case-row-detail");
      let built = false;

      summary.addEventListener("click", () => {
        row.classList.toggle("expanded");
        if (row.classList.contains("expanded") && !built) {
          detail.innerHTML = buildDetailHtml(pair);
          built = true;
        }
      });

      listEl.appendChild(row);
    });
  }

  function buildDetailHtml(pair) {
    const inclusionHtml = pair.inclusion_criteria.map((c) => `<li>${escapeHtml(c)}</li>`).join("");
    const exclusionHtml = pair.exclusion_criteria.map((c) => `<li>${escapeHtml(c)}</li>`).join("");

    const tagsBlock = (pair.tags || []).length
      ? `<div class="case-card-label" style="margin-top:16px;">Flagged in error taxonomy</div>
         <div style="display:flex; flex-direction:column; gap:6px;">
           ${pair.tags.map((t) => `<div><span class="tag-chip">${escapeHtml(t.tag)}</span> <span style="font-size:0.82rem; color:var(--muted);">${escapeHtml(t.note)}</span></div>`).join("")}
         </div>`
      : "";

    const verdictCards = Object.entries(pair.verdicts).map(([comboKey, v]) => {
      const cfg = MODELS[comboKey];
      const name = cfg ? cfg.display_name : comboKey;
      const matchClass = v.label === pair.ground_truth ? "match" : "mismatch";
      return `
        <div class="verdict-card">
          <div class="verdict-card-head">
            <span class="verdict-card-name">${escapeHtml(name)}</span>
            <span class="badge badge-${matchClass}">${escapeHtml(v.label)}</span>
          </div>
          <div class="verdict-citation">"${escapeHtml(v.cited_criterion)}"</div>
          <div class="verdict-rationale">${escapeHtml(v.rationale)}</div>
        </div>
      `;
    }).join("");

    return `
      <div class="case-card-label" style="margin-top:16px;">Patient vignette — TREC topic ${pair.query_id}</div>
      <div class="patient-text">${escapeHtml(pair.patient_text)}</div>

      <div class="case-card-label" style="margin-top:16px;">Trial — ${escapeHtml(pair.doc_id)}</div>
      <h3 style="margin: 0 0 8px;">${escapeHtml(pair.trial_title)}</h3>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
          <div class="case-card-label">Inclusion</div>
          <ul class="criteria-list">${inclusionHtml}</ul>
        </div>
        <div>
          <div class="case-card-label">Exclusion</div>
          <ul class="criteria-list">${exclusionHtml}</ul>
        </div>
      </div>

      ${tagsBlock}

      <div class="case-card-label" style="margin-top:16px;">All judge verdicts for this case</div>
      <div class="verdict-grid">${verdictCards}</div>
    `;
  }

  document.querySelectorAll("#explore-model-toggle .model-option").forEach((opt) => {
    opt.addEventListener("click", () => {
      document.querySelectorAll("#explore-model-toggle .model-option").forEach((o) => o.classList.remove("selected"));
      opt.classList.add("selected");
      selectedModel = opt.dataset.modelKey;
      renderSegments();
    });
  });

  renderSegments();
});
