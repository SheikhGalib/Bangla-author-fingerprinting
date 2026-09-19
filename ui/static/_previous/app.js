// Bangla Authorship Attribution Frontend Application Logic
document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const passageInput = document.getElementById("passage-input");
  const liveTokenCount = document.getElementById("live-token-count");
  const liveCharCount = document.getElementById("live-char-count");
  const liveSentCount = document.getElementById("live-sent-count");
  const liveShortWarning = document.getElementById("live-short-warning");

  const btnAttribute = document.getElementById("btn-attribute");
  const btnText = document.getElementById("btn-text");
  const spinner = document.getElementById("spinner");

  const demoButtons = document.getElementById("demo-buttons");
  const btnExampleShort = document.getElementById("btn-example-short");
  const btnClear = document.getElementById("btn-clear");

  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");
  const resultsWrapper = document.getElementById("results-wrapper");
  const modelStatusPill = document.getElementById("model-status-pill");

  // Headline Result Elements
  const resAuthorAvatar = document.getElementById("res-author-avatar");
  const resAuthorBn = document.getElementById("res-author-bn");
  const resAuthorEn = document.getElementById("res-author-en");
  const resAuthorYears = document.getElementById("res-author-years");
  const resPassageMeta = document.getElementById("res-passage-meta");
  const badgeAgreement = document.getElementById("badge-agreement");
  const badgeShortWarning = document.getElementById("badge-short-warning");

  const bannerDisagreement = document.getElementById("banner-disagreement");
  const disagreementDetails = document.getElementById("disagreement-details");

  // Register Elements
  const registerRatioLabel = document.getElementById("register-ratio-label");
  const registerIndicatorBar = document.getElementById("register-indicator-bar");
  const chalitRateLabel = document.getElementById("chalit-rate-label");
  const sadhuRateLabel = document.getElementById("sadhu-rate-label");

  // Discriminative Elements
  const svmMarginVal = document.getElementById("svm-margin-val");
  const svmPredictedName = document.getElementById("svm-predicted-name");
  const svmRunnerupName = document.getElementById("svm-runnerup-name");
  const svmScoresBars = document.getElementById("svm-scores-bars");
  const readableForList = document.getElementById("readable-for-list");
  const readableAgainstList = document.getElementById("readable-against-list");
  const allForList = document.getElementById("all-for-list");
  const allAgainstList = document.getElementById("all-against-list");

  // Generative Elements
  const perplexityBarsContainer = document.getElementById("perplexity-bars-container");
  const genEvidenceBadges = document.getElementById("gen-evidence-badges");

  // Accordion Toggles
  const btnToggleAllFeatures = document.getElementById("btn-toggle-all-features");
  const allFeaturesContent = document.getElementById("all-features-content");
  const arrowAllFeatures = document.getElementById("arrow-all-features");

  const btnToggleProfile = document.getElementById("btn-toggle-profile");
  const profileContent = document.getElementById("profile-content");
  const arrowProfile = document.getElementById("arrow-profile");
  const profileGrid = document.getElementById("profile-grid");

  let demosData = null;

  // 1. Check Server & Model Health
  async function checkHealth() {
    try {
      const resp = await fetch("/api/health");
      const data = await resp.json();
      if (!data.models_loaded) {
        modelStatusPill.className = "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200";
        modelStatusPill.innerHTML = '<span class="w-2 h-2 rounded-full bg-rose-500"></span> Models Not Loaded';
        showError(data.error || "Trained models not found. Please run notebook 04.");
      }
    } catch (err) {
      console.warn("Health check failed:", err);
    }
  }
  checkHealth();

  // 2. Fetch Demos
  async function loadDemos() {
    try {
      const resp = await fetch("/api/demo");
      if (resp.ok) {
        demosData = await resp.json();
        renderDemoButtons();
      }
    } catch (err) {
      console.warn("Failed to load demo passages:", err);
    }
  }
  loadDemos();

  // 3. Live Token & Sentence Counter
  function updateLiveCounts() {
    const text = passageInput.value;
    const trimmed = text.trim();
    const chars = text.length;
    const tokens = trimmed ? trimmed.split(/\s+/).length : 0;
    
    // Sentence split on dari (।), ?, !, and newline/period
    const sents = trimmed ? trimmed.split(/[।?!]|\.\s+/).filter(s => s.trim().length > 0).length : 0;

    liveTokenCount.textContent = `${tokens} ${tokens === 1 ? 'token' : 'tokens'}`;
    liveCharCount.textContent = `${chars} chars`;
    liveSentCount.textContent = `${sents} ${sents === 1 ? 'sentence' : 'sentences'}`;

    if (tokens > 0 && tokens < 80) {
      liveShortWarning.classList.remove("hidden");
    } else {
      liveShortWarning.classList.add("hidden");
    }
  }

  passageInput.addEventListener("input", updateLiveCounts);

  // 4. Sample buttons, built from whatever /api/demo returns.
  //
  // The server reads these straight out of corpus/unseen/, so every sample is
  // from a book no model was trained on — the same thing a reader does by hand.
  // Nothing about the roster is hard-coded here; change the corpus and the
  // buttons change with it.
  function renderDemoButtons() {
    if (!demoButtons || !demosData) return;
    demoButtons.innerHTML = "";
    Object.entries(demosData).forEach(([key, demo]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "px-2.5 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 " +
        "rounded-md font-medium transition flex items-center gap-1";
      btn.title = demo.title || key;

      const en = document.createElement("span");
      en.textContent = (demo.author_hint || key).split(" (")[0];
      const bn = document.createElement("span");
      bn.className = "font-bengali text-slate-500";
      bn.textContent = demo.book ? `(${demo.book})` : "";

      btn.append(en, bn);
      btn.addEventListener("click", () => {
        passageInput.value = demo.text;
        updateLiveCounts();
        hideError();
      });
      demoButtons.appendChild(btn);
    });
  }

  btnExampleShort.addEventListener("click", () => {
    // Demonstrates the low-confidence path: one sentence off a held-out book.
    const first = demosData ? Object.values(demosData)[0] : null;
    if (first) {
      const sentences = first.text.split("।");
      passageInput.value = sentences[0].trim() + "।";
    }
    updateLiveCounts();
    hideError();
  });

  btnClear.addEventListener("click", () => {
    passageInput.value = "";
    updateLiveCounts();
    resultsWrapper.classList.add("hidden");
    hideError();
  });

  // 5. Accordion Toggles
  btnToggleAllFeatures.addEventListener("click", () => {
    const isHidden = allFeaturesContent.classList.contains("hidden");
    if (isHidden) {
      allFeaturesContent.classList.remove("hidden");
      arrowAllFeatures.classList.add("rotate-180");
    } else {
      allFeaturesContent.classList.add("hidden");
      arrowAllFeatures.classList.remove("rotate-180");
    }
  });

  btnToggleProfile.addEventListener("click", () => {
    const isHidden = profileContent.classList.contains("hidden");
    if (isHidden) {
      profileContent.classList.remove("hidden");
      arrowProfile.classList.add("rotate-180");
    } else {
      profileContent.classList.add("hidden");
      arrowProfile.classList.remove("rotate-180");
    }
  });

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  // 6. Attribute Action
  btnAttribute.addEventListener("click", async () => {
    const text = passageInput.value.trim();
    if (!text) {
      showError("Please paste or type a Bangla passage first.");
      return;
    }

    hideError();
    btnAttribute.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Analyzing Style...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, top_k: 8 })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error occurred during prediction.");
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      showError(err.message);
      resultsWrapper.classList.add("hidden");
    } finally {
      btnAttribute.disabled = false;
      spinner.classList.add("hidden");
      btnText.textContent = "Attribute Author & Explain";
    }
  });

  // 7. Render Prediction Results
  function renderResults(res) {
    const authors = res.authors || {};
    const predictedKey = res.predicted;
    const authorMeta = authors[predictedKey] || {
      en: predictedKey,
      bn: predictedKey,
      short: predictedKey,
      years: ""
    };

    // Headline
    resAuthorBn.textContent = authorMeta.bn;
    resAuthorEn.textContent = authorMeta.en;
    resAuthorYears.textContent = authorMeta.years ? `(${authorMeta.years})` : "";
    resAuthorAvatar.textContent = authorMeta.bn.charAt(0) || "বা";
    resPassageMeta.textContent = `${res.n_tokens} tokens · ${res.n_sentences} sentences`;

    // Agreement Badge & Banner
    if (res.agree) {
      badgeAgreement.className = "px-3 py-1 rounded-full font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1.5 shadow-sm";
      badgeAgreement.innerHTML = `
        <svg class="w-3.5 h-3.5 text-emerald-600" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
        <span>Both Paradigms Agree</span>
      `;
      bannerDisagreement.classList.add("hidden");
    } else {
      badgeAgreement.className = "px-3 py-1 rounded-full font-semibold bg-amber-50 text-amber-800 border border-amber-300 flex items-center gap-1.5 shadow-sm";
      badgeAgreement.innerHTML = `
        <svg class="w-3.5 h-3.5 text-amber-600" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
        <span>Models Disagree (Low Confidence)</span>
      `;
      
      const dAuthor = authors[res.discriminative?.predicted]?.en || res.discriminative?.predicted;
      const gAuthor = authors[res.generative?.predicted]?.en || res.generative?.predicted;
      disagreementDetails.innerHTML = `
        The discriminative SVM predicts <strong>${dAuthor}</strong>, while the generative language models favor <strong>${gAuthor}</strong>. Because the two paradigms point in opposite directions, treat this attribution with lower confidence.
      `;
      bannerDisagreement.classList.remove("hidden");
    }

    // Short Passage Warning Badge
    if (res.short_passage_warning) {
      badgeShortWarning.classList.remove("hidden");
    } else {
      badgeShortWarning.classList.add("hidden");
    }

    // Register Meter
    const reg = res.register || { sadhu_rate: 0, chalit_rate: 0, sadhu_chalit_ratio: 0 };
    sadhuRateLabel.textContent = `Sadhu marker rate: ${(reg.sadhu_rate * 100).toFixed(2)}%`;
    chalitRateLabel.textContent = `Chalit marker rate: ${(reg.chalit_rate * 100).toFixed(2)}%`;
    registerRatioLabel.textContent = `Sadhu-to-Chalit Ratio: ${reg.sadhu_chalit_ratio.toFixed(2)}`;

    // Calculate meter percentage: ratio of sadhu vs (sadhu + chalit)
    let meterPercent = 50;
    const totalMarkers = reg.sadhu_rate + reg.chalit_rate;
    if (totalMarkers > 0) {
      meterPercent = Math.min(95, Math.max(5, (reg.sadhu_rate / totalMarkers) * 100));
    }
    registerIndicatorBar.style.width = `${meterPercent}%`;

    // Discriminative (SVM) Render
    const disc = res.discriminative;
    if (disc) {
      svmMarginVal.textContent = disc.margin >= 0 ? `+${disc.margin.toFixed(3)}` : disc.margin.toFixed(3);
      
      const discPredMeta = authors[disc.predicted] || { bn: disc.predicted, en: disc.predicted };
      const discRunnerMeta = authors[disc.runner_up] || { bn: disc.runner_up, en: disc.runner_up };
      
      svmPredictedName.textContent = `${discPredMeta.bn} (${discPredMeta.en})`;
      svmRunnerupName.textContent = `${discRunnerMeta.bn} (${discRunnerMeta.en})`;

      // Render Scores
      if (disc.scores) {
        svmScoresBars.innerHTML = "";
        const scoreEntries = Object.entries(disc.scores).sort((a, b) => b[1] - a[1]);
        const maxScore = Math.max(...scoreEntries.map(e => Math.abs(e[1])), 1.0);
        
        scoreEntries.forEach(([key, val]) => {
          const aMeta = authors[key] || { short: key, bn: key };
          const isWinner = (key === disc.predicted);
          const pct = Math.max(8, Math.min(100, (val / maxScore) * 100));
          
          const row = document.createElement("div");
          row.className = "flex items-center justify-between text-[11px] py-0.5";
          row.innerHTML = `
            <span class="w-32 font-medium ${isWinner ? 'text-indigo-700 font-bold' : 'text-slate-600'} truncate">
              ${aMeta.short} <span class="font-bengali text-slate-400">(${aMeta.bn})</span>
            </span>
            <div class="flex-1 mx-2 bg-slate-200 h-2 rounded-full overflow-hidden">
              <div class="${isWinner ? 'bg-indigo-600' : 'bg-slate-400'} h-full rounded-full" style="width: ${pct}%"></div>
            </div>
            <span class="w-12 text-right font-mono ${isWinner ? 'font-bold text-indigo-700' : 'text-slate-500'}">
              ${val.toFixed(2)}
            </span>
          `;
          svmScoresBars.appendChild(row);
        });
      }

      // Supporting human-checkable evidence
      renderEvidenceList(readableForList, disc.readable_for, true);
      // Opposing human-checkable evidence
      renderEvidenceList(readableAgainstList, disc.readable_against, false);

      // Raw Technical View
      renderRawList(allForList, disc.evidence_for, true);
      renderRawList(allAgainstList, disc.evidence_against, false);
    }

    // Generative (Language Models) Render
    const gen = res.generative;
    if (gen && gen.perplexity) {
      perplexityBarsContainer.innerHTML = "";
      const pplEntries = Object.entries(gen.perplexity).sort((a, b) => a[1] - b[1]); // ascending: lower is better!
      const minPpl = pplEntries[0][1];
      const maxPpl = pplEntries[pplEntries.length - 1][1];

      pplEntries.forEach(([key, ppl], idx) => {
        const aMeta = authors[key] || { en: key, bn: key, short: key };
        const isBest = (idx === 0);
        
        // Scale bar so best fit has lowest width or relative indicator
        // Visually: lower perplexity = tighter/better fit
        const relPct = Math.min(100, Math.max(15, (ppl / maxPpl) * 100));

        const item = document.createElement("div");
        item.className = "p-2 rounded-lg border transition " + (isBest ? 'bg-emerald-50/70 border-emerald-300' : 'bg-slate-50 border-slate-200');
        item.innerHTML = `
          <div class="flex justify-between items-center mb-1">
            <span class="font-semibold ${isBest ? 'text-emerald-900 font-bold' : 'text-slate-700'}">
              ${aMeta.en} <span class="font-bengali text-slate-500 text-[11px]">(${aMeta.bn})</span>
            </span>
            <div class="flex items-center gap-1.5">
              <span class="font-mono font-bold ${isBest ? 'text-emerald-700 text-sm' : 'text-slate-600'}">
                ${ppl.toFixed(2)}
              </span>
              ${isBest ? '<span class="text-[10px] uppercase font-bold bg-emerald-600 text-white px-1.5 py-0.5 rounded">◀ Best Fit</span>' : ''}
            </div>
          </div>
          <div class="ppl-bar-track">
            <div class="${isBest ? 'ppl-bar-fill' : 'ppl-bar-fill-secondary'}" style="width: ${relPct}%"></div>
          </div>
        `;
        perplexityBarsContainer.appendChild(item);
      });

      // Generative log-odds character evidence
      genEvidenceBadges.innerHTML = "";
      if (gen.evidence_for && gen.evidence_for.length > 0) {
        gen.evidence_for.forEach(item => {
          const badge = document.createElement("span");
          badge.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded bg-emerald-100/70 text-emerald-900 border border-emerald-300 font-medium";
          badge.innerHTML = `<span class="font-bengali font-bold text-sm">“${item.symbol}”</span> <span class="text-[10px] text-emerald-700 font-mono">+${item.log_odds.toFixed(2)}</span>`;
          genEvidenceBadges.appendChild(badge);
        });
      } else {
        genEvidenceBadges.innerHTML = '<span class="text-slate-400 italic">No specific high-margin character highlights.</span>';
      }
    }

    // Profile Render (29 Features)
    if (res.profile && Array.isArray(res.profile)) {
      profileGrid.innerHTML = "";
      res.profile.forEach(feat => {
        const card = document.createElement("div");
        card.className = "bg-white p-2.5 rounded border border-slate-200 flex justify-between items-center";
        card.innerHTML = `
          <span class="text-slate-600 truncate pr-2 font-medium" title="${feat.label}">${feat.label}</span>
          <span class="font-mono font-bold text-slate-800 text-right shrink-0">${feat.value}</span>
        `;
        profileGrid.appendChild(card);
      });
    }

    // Show Results & Smooth Scroll
    resultsWrapper.classList.remove("hidden");
    resultsWrapper.classList.add("fade-in");
    resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderEvidenceList(container, items, isPositive) {
    container.innerHTML = "";
    if (!items || items.length === 0) {
      container.innerHTML = '<div class="text-slate-400 italic py-1">None observed in this passage.</div>';
      return;
    }

    const maxContrib = Math.max(...items.map(it => Math.abs(it.contribution)), 0.01);

    items.forEach(it => {
      const absVal = Math.abs(it.contribution);
      const widthPct = Math.min(100, Math.max(10, (absVal / maxContrib) * 100));
      const sign = it.contribution >= 0 ? "+" : "";

      const row = document.createElement("div");
      row.className = "p-2 bg-slate-50 rounded border border-slate-200/80 space-y-1";
      row.innerHTML = `
        <div class="flex justify-between items-start gap-2">
          <span class="font-medium text-slate-800 leading-tight">
            ${escapeHtml(it.gloss)}
          </span>
          <span class="font-mono font-bold shrink-0 ${isPositive ? 'text-emerald-700' : 'text-rose-700'}">
            ${sign}${it.contribution.toFixed(4)}
          </span>
        </div>
        <div class="evidence-bar-track">
          <div class="${isPositive ? 'evidence-bar-fill-pos' : 'evidence-bar-fill-neg'}" style="width: ${widthPct}%"></div>
        </div>
        <div class="flex justify-between text-[10px] text-slate-400">
          <span>Observed value in text: ${it.value.toFixed(3)}</span>
        </div>
      `;
      container.appendChild(row);
    });
  }

  function renderRawList(container, items, isPositive) {
    container.innerHTML = "";
    if (!items || items.length === 0) {
      container.innerHTML = '<span class="text-slate-400 italic">None</span>';
      return;
    }

    items.forEach(it => {
      const sign = it.contribution >= 0 ? "+" : "";
      const div = document.createElement("div");
      div.className = "flex justify-between py-0.5 border-b border-slate-100 last:border-0";
      div.innerHTML = `
        <span class="${isPositive ? 'text-emerald-800' : 'text-rose-800'} truncate mr-2">${escapeHtml(it.gloss)}</span>
        <span class="font-mono ${isPositive ? 'text-emerald-700' : 'text-rose-700'} shrink-0">${sign}${it.contribution.toFixed(4)}</span>
      `;
      container.appendChild(div);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

