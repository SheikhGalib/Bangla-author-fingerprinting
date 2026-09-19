// Bangla Authorship Attribution — demonstration frontend.
//
// Four things on one page, in the order they are explained: the corpus, the
// training pipeline, an input box, and the three models' answers side by side.
//
// Nothing about the roster, the books or the sample passages is written here.
// It all arrives from the API, which reads corpus/ at request time — so the
// page cannot drift from the data the models were actually trained on.

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);

  const passageInput = $("passage-input");
  const corpusGrid = $("corpus-grid");
  const pipelineBox = $("pipeline");
  const demoButtons = $("demo-buttons");
  const verdictGrid = $("verdict-grid");
  const resultsSection = $("results-section");
  const consensusLine = $("consensus-line");
  const errorBox = $("error-box");
  const btnCompare = $("btn-compare");
  const btnClear = $("btn-clear");

  let authors = {};
  let demos = {};

  // The stages mirror Figure 1 of the report exactly, so the same eight boxes
  // can be pointed at on screen and on paper without re-explaining them.
  const STAGES = [
    ["Corpus", "6 training books", "The 3 held-out books are set aside here and are not seen again until stage 8."],
    ["Passages", "~120 words each", "Cut on sentence boundaries, never across a chapter."],
    ["WordPiece tokenizer", "[CLS] + subwords + [SEP]", "Splits into subwords, so an unfamiliar inflected form still breaks into known pieces."],
    ["BanglaBERT encoder", "12 transformer layers", "Already pretrained on a large Bangla corpus. Fine-tuning only adapts it."],
    ["Classification head", "768 → 3 scores", "The [CLS] vector summarises the passage; one score comes out per author."],
    ["Weighted cross-entropy", "AdamW optimiser", "Training data is uneven, so each author is weighted by inverse frequency."],
    ["Keep the best epoch", "chosen on validation", "Stops us reporting whichever epoch happened to run last."],
    ["Test", "537 passages, 3 held-out books", "The only point at which the held-out books are touched."],
  ];

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }
  function hideError() {
    errorBox.classList.add("hidden");
  }

  // ---------------------------------------------------------------- counts
  function updateCounts() {
    const t = passageInput.value.trim();
    const words = t ? t.split(/\s+/).length : 0;
    const sents = t ? (t.match(/[।?!]/g) || []).length || 1 : 0;
    $("count-tokens").textContent = words;
    $("count-sents").textContent = sents;
    $("short-warning").classList.toggle("hidden", words === 0 || words >= 40);
  }
  passageInput.addEventListener("input", updateCounts);

  // --------------------------------------------------------------- section 2
  function renderPipeline() {
    pipelineBox.innerHTML = STAGES.map(([name, sub, note], i) => `
      <div class="flex items-stretch gap-3">
        <div class="flex flex-col items-center shrink-0">
          <div class="w-8 h-8 rounded-full ${i === 7 ? "bg-amber-600" : "bg-emerald-700"}
                      text-white text-xs font-bold flex items-center justify-center">${i + 1}</div>
          ${i < STAGES.length - 1 ? '<div class="w-px flex-1 bg-slate-300 my-1"></div>' : ""}
        </div>
        <div class="flex-1 pb-3 grid grid-cols-1 sm:grid-cols-[minmax(0,15rem)_1fr] gap-x-4 gap-y-1">
          <div class="bg-white border ${i === 7 ? "border-amber-300" : "border-slate-200"} rounded-lg px-3 py-2">
            <p class="text-sm font-semibold text-slate-900">${esc(name)}</p>
            <p class="text-xs text-slate-500 font-mono">${esc(sub)}</p>
          </div>
          <p class="text-xs text-slate-600 self-center">${esc(note)}</p>
        </div>
      </div>`).join("");
  }

  // --------------------------------------------------------------- section 1
  function renderCorpus(data) {
    authors = data.authors || {};
    const byAuthor = {};
    (data.books || []).forEach((b) => (byAuthor[b.author] ||= []).push(b));

    corpusGrid.innerHTML = Object.keys(authors).map((key) => {
      const meta = authors[key];
      const books = byAuthor[key] || [];
      return `
        <div class="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <p class="font-bengali text-base font-semibold text-slate-900">${esc(meta.bn)}</p>
          <p class="text-xs text-slate-500 mb-3">${esc(meta.en)} · ${esc(meta.years)}</p>
          <div class="space-y-2">
            ${books.map((b) => {
              const held = b.role === "unseen";
              return `
              <div>
                <button class="book-toggle w-full text-left rounded-lg border px-2.5 py-2 transition
                        ${held ? "border-amber-300 bg-amber-50 hover:bg-amber-100"
                               : "border-slate-200 bg-slate-50 hover:bg-slate-100"}"
                        data-book="${esc(b.book)}">
                  <div class="flex items-start justify-between gap-2">
                    <span class="font-bengali text-sm text-slate-900">${esc(b.book)}</span>
                    <span class="shrink-0 text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded
                          ${held ? "bg-amber-600 text-white" : "bg-slate-300 text-slate-700"}">
                      ${held ? "held out" : "train"}
                    </span>
                  </div>
                  <p class="text-[11px] text-slate-500 mt-0.5">
                    ${b.chapters} chapters · ${b.chars.toLocaleString()} characters
                  </p>
                </button>
                <div class="book-glimpse hidden mt-1 p-2.5 bg-white border border-slate-200 rounded-lg">
                  <p class="text-[10px] uppercase tracking-wide text-slate-400 mb-1">A paragraph from this book</p>
                  <p class="font-bengali text-[13px] leading-relaxed text-slate-700">${esc(b.glimpse)}</p>
                </div>
              </div>`;
            }).join("")}
          </div>
        </div>`;
    }).join("");

    corpusGrid.querySelectorAll(".book-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.parentElement.querySelector(".book-glimpse").classList.toggle("hidden");
      });
    });
  }

  // --------------------------------------------------------------- section 3
  function renderDemoButtons() {
    demoButtons.innerHTML = "";
    Object.entries(demos).forEach(([key, demo]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "px-2.5 py-1 bg-amber-50 border border-amber-200 hover:bg-amber-100 " +
                    "text-amber-900 rounded-md font-medium transition";
      b.title = demo.title;
      b.textContent = (authors[key]?.short) || key;
      b.addEventListener("click", () => {
        passageInput.value = demo.text;
        updateCounts();
        hideError();
      });
      demoButtons.appendChild(b);
    });
  }

  // --------------------------------------------------------------- section 4
  function renderVerdicts(data) {
    const names = data.authors || authors;

    verdictGrid.innerHTML = (data.verdicts || []).map((v, i) => {
      if (!v.available) {
        return `
          <div class="bg-white rounded-xl border border-slate-200 p-4 opacity-60">
            <p class="text-sm font-bold text-slate-700">${esc(v.name)}</p>
            <p class="text-xs text-slate-400 mt-2">Not loaded</p>
            <p class="text-[11px] text-slate-400 mt-1">${esc(v.error || "")}</p>
          </div>`;
      }
      const ranked = Object.entries(v.scores).sort((a, b) => b[1] - a[1]);
      const winner = names[v.predicted] || { bn: v.predicted, en: v.predicted };
      return `
        <div class="bg-white rounded-xl border-2 ${i === 2 ? "border-emerald-500" : "border-slate-200"} p-4 shadow-sm">
          <div class="flex items-center justify-between gap-2 mb-1">
            <p class="text-sm font-bold text-slate-900">${esc(v.name)}</p>
            <span class="text-[10px] bg-slate-100 text-slate-600 rounded px-1.5 py-0.5">${i + 1}</span>
          </div>
          <p class="text-[11px] text-slate-500 mb-3">${esc(v.stage)}</p>

          <p class="font-bengali text-lg font-bold text-emerald-800 leading-tight">${esc(winner.bn)}</p>
          <p class="text-xs text-slate-500 mb-3">${esc(winner.en)}</p>

          <div class="space-y-1.5">
            ${ranked.map(([a, s]) => `
              <div>
                <div class="flex justify-between text-[11px] text-slate-600">
                  <span class="font-bengali">${esc((names[a] || {}).bn || a)}</span>
                  <span class="font-mono">${(s * 100).toFixed(1)}%</span>
                </div>
                <div class="h-1.5 bg-slate-100 rounded overflow-hidden">
                  <div class="h-full ${a === v.predicted ? "bg-emerald-600" : "bg-slate-300"}"
                       style="width:${Math.max(2, s * 100).toFixed(1)}%"></div>
                </div>
              </div>`).join("")}
          </div>
        </div>`;
    }).join("");

    if (data.unanimous) {
      const w = names[data.consensus] || { bn: data.consensus };
      consensusLine.innerHTML =
        `<span class="inline-block bg-emerald-50 border border-emerald-300 text-emerald-900 rounded-lg px-3 py-1.5">
           All ${data.n_models} models agree: <strong class="font-bengali">${esc(w.bn)}</strong>
         </span>`;
    } else {
      consensusLine.innerHTML =
        `<span class="inline-block bg-amber-50 border border-amber-300 text-amber-900 rounded-lg px-3 py-1.5">
           The models <strong>disagree</strong> — which is itself informative. Compare the bars below.
         </span>`;
    }
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------ wiring
  btnCompare.addEventListener("click", async () => {
    const text = passageInput.value.trim();
    if (!text) return showError("Paste a Bangla passage first.");
    hideError();
    btnCompare.disabled = true;
    btnCompare.textContent = "Running three models…";
    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "request failed");
      renderVerdicts(data);
    } catch (err) {
      showError(err.message);
    } finally {
      btnCompare.disabled = false;
      btnCompare.textContent = "Compare all three models";
    }
  });

  btnClear.addEventListener("click", () => {
    passageInput.value = "";
    updateCounts();
    hideError();
    resultsSection.classList.add("hidden");
  });

  // ------------------------------------------------------------------- boot
  (async () => {
    renderPipeline();
    try {
      const health = await (await fetch("/api/health")).json();
      const m = health.models || {};
      $("model-status").innerHTML = Object.entries({
        "TF-IDF": m.tfidf_svm, "BiLSTM": m.bilstm, "BanglaBERT": m.banglabert,
      }).map(([label, ok]) => `
        <span class="px-1.5 py-0.5 rounded border ${ok
          ? "bg-emerald-50 border-emerald-200 text-emerald-800"
          : "bg-rose-50 border-rose-200 text-rose-700"}">${label}</span>`).join("");
    } catch { /* the page is still usable without the status pills */ }

    try {
      renderCorpus(await (await fetch("/api/corpus")).json());
    } catch (err) {
      corpusGrid.innerHTML =
        `<p class="text-sm text-rose-600">Could not load the corpus: ${esc(err.message)}</p>`;
    }

    try {
      demos = await (await fetch("/api/demo")).json();
      renderDemoButtons();
    } catch { /* the textarea still works without samples */ }

    updateCounts();
  })();
});
