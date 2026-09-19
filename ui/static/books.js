// The Books page — one profile per book, with overall summary, chapter story
// synopses, and stylistic analytics computed directly from the corpus.

document.addEventListener("DOMContentLoaded", async () => {
  const root = document.getElementById("author-sections");

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const num = (v, d = 0) =>
    v === undefined || v === null ? "—" : Number(v).toLocaleString(undefined, {
      minimumFractionDigits: d, maximumFractionDigits: d,
    });

  let notes = {};
  try {
    notes = await (await fetch("/book_notes.json")).json();
  } catch { /* notes are optional; analytics carry the page */ }

  let data;
  try {
    data = await (await fetch("/api/analytics/books")).json();
  } catch (err) {
    root.innerHTML = `<p class="text-sm text-rose-600">Could not load analytics: ${esc(err.message)}</p>`;
    return;
  }

  const authors = data.authors || {};
  const byAuthor = {};
  (data.books || []).forEach((b) => (byAuthor[b.author] ||= []).push(b));
  // Training books first, held-out last, so the page reads in pipeline order.
  Object.values(byAuthor).forEach((list) =>
    list.sort((a, b) => (a.role === b.role ? 0 : a.role === "train" ? -1 : 1)));

  const chips = (items, cls, label) => {
    if (!items || !items.length) {
      return `<p class="text-xs text-slate-400 italic">none — ${esc(label)}</p>`;
    }
    return `<div class="flex flex-wrap gap-1.5">` + items.map((d) => `
      <span class="inline-flex items-baseline gap-1 px-2 py-0.5 rounded ${cls}">
        <span class="font-bengali text-[13px]">${esc(d.word)}</span>
        <span class="text-[10px] opacity-70">${d.count}</span>
      </span>`).join("") + `</div>`;
  };

  const statCell = (label, value, hint) => `
    <div class="bg-slate-50 rounded-lg px-2.5 py-2">
      <p class="text-[10px] uppercase tracking-wide text-slate-400">${esc(label)}</p>
      <p class="text-sm font-semibold text-slate-800">${value}</p>
      ${hint ? `<p class="text-[10px] text-slate-400">${esc(hint)}</p>` : ""}
    </div>`;

  root.innerHTML = Object.keys(authors).map((key) => {
    const meta = authors[key];
    const books = byAuthor[key] || [];
    return `
      <section>
        <div class="flex items-baseline gap-3 mb-4 pb-2 border-b-2 border-slate-200">
          <h2 class="font-bengali text-xl font-bold text-slate-900">${esc(meta.bn)}</h2>
          <p class="text-sm text-slate-500">${esc(meta.en)} · ${esc(meta.years)}</p>
        </div>

        <div class="space-y-6">
        ${books.map((b) => {
          const held = b.role === "unseen";
          const note = notes[b.book] || {};
          const nameHint = b.n_chapters > 20
            ? "a collection of separate stories, so no single cast recurs"
            : "no name met the frequency and spread thresholds";

          return `
          <article class="bg-white rounded-xl border-2 ${held ? "border-amber-300" : "border-slate-200"} shadow-sm overflow-hidden">
            <!-- Header bar -->
            <div class="${held ? "bg-amber-50" : "bg-slate-50"} px-4 py-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-b ${held ? "border-amber-200" : "border-slate-200"}">
              <h3 class="font-bengali text-lg font-bold text-slate-900">${esc(b.book)}</h3>
              ${note.form ? `<span class="text-xs px-2 py-0.5 rounded bg-white/80 border border-slate-200 text-slate-600 font-medium">${esc(note.form)}</span>` : ""}
              <span class="ml-auto text-[10px] font-bold uppercase tracking-wide px-2 py-1 rounded
                    ${held ? "bg-amber-600 text-white" : "bg-slate-300 text-slate-700"}">
                ${held ? "held out — never trained on" : "training book"}
              </span>
            </div>

            <div class="p-5 space-y-4">
              <!-- Book & Stories Summaries -->
              <div class="space-y-3">
                ${note.summary ? `
                <div class="bg-slate-50 border border-slate-200/90 rounded-xl p-3.5 space-y-1.5">
                  <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 shrink-0"></span>
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-800">Overall Book Summary</h4>
                  </div>
                  <p class="text-sm text-slate-700 leading-relaxed">${esc(note.summary)}</p>
                </div>` : ""}

                ${note.stories_summary ? `
                <div class="bg-sky-50/70 border border-sky-200/80 rounded-xl p-3.5 space-y-1.5">
                  <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-sky-500 shrink-0"></span>
                    <h4 class="text-xs font-bold uppercase tracking-wider text-sky-950">Stories &amp; Chapters Narrative Arc</h4>
                  </div>
                  <p class="text-xs text-slate-700 leading-relaxed">${esc(note.stories_summary)}</p>
                </div>` : ""}

                ${!note.summary && note.note ? `<p class="text-sm text-slate-600">${esc(note.note)}</p>` : ""}
              </div>

              <!-- Statistical summary grid -->
              <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                ${statCell("Chapters", num(b.n_chapters))}
                ${statCell("Words", num(b.n_tokens))}
                ${statCell("Unique words", num(b.n_unique))}
                ${statCell("Type/token", num(b.type_token_ratio, 3), "vocabulary richness")}
                ${statCell("Sentence len", num(b.mean_sentence_len, 1), "words")}
                ${statCell("Dialogue", num(b.dialogue_rate, 2), "quote marks /100 words")}
                ${statCell("Sadhu:chalit", num(b.sadhu_chalit_ratio, 2), "register")}
              </div>

              <!-- Vocabulary signals -->
              <div class="grid md:grid-cols-2 gap-4">
                <div>
                  <p class="text-xs font-bold text-slate-700 mb-1.5">Recurring names &amp; places
                    <span class="font-normal text-slate-400">— who and where this book is about</span></p>
                  ${chips(b.recurring_names, "bg-emerald-50 text-emerald-900 border border-emerald-200", nameHint)}
                </div>
                <div>
                  <p class="text-xs font-bold text-slate-700 mb-1.5">Distinctive vocabulary
                    <span class="font-normal text-slate-400">— highest TF-IDF vs the other 8 books</span></p>
                  ${chips(b.distinctive, "bg-sky-50 text-sky-900 border border-sky-200", "")}
                </div>
              </div>

              <!-- Chapter & Stories Directory (Accordion) -->
              <details class="group bg-slate-50/70 border border-slate-200 rounded-xl p-3.5 transition-all">
                <summary class="cursor-pointer text-xs font-bold text-slate-700 hover:text-slate-900 select-none flex items-center justify-between">
                  <span class="flex items-center gap-2">
                    <span class="text-slate-400 group-open:rotate-90 inline-block transition-transform duration-150">▶</span>
                    <span>Chapter Stories &amp; Content Directory (${b.n_chapters})</span>
                  </span>
                  <span class="text-[11px] font-normal text-slate-400">Click to view chapter-by-chapter story details</span>
                </summary>
                <div class="mt-3.5 divide-y divide-slate-200/80 max-h-[500px] overflow-y-auto pr-1">
                  ${(b.chapters || []).map((c, idx) => {
                    const chStory = (note.chapter_stories && note.chapter_stories[c]) || "";
                    return `
                    <div class="py-2 flex flex-col sm:flex-row sm:items-start gap-1 sm:gap-3 text-xs">
                      <div class="flex items-baseline gap-1.5 shrink-0 sm:w-56">
                        <span class="text-slate-400 font-mono text-[11px]">${idx + 1}.</span>
                        <span class="font-bengali font-semibold text-[13px] text-slate-800">${esc(c)}</span>
                      </div>
                      <p class="text-slate-600 leading-relaxed font-sans text-xs flex-1">
                        ${chStory ? esc(chStory) : `<span class="text-slate-400 italic">Chapter text processed in corpus</span>`}
                      </p>
                    </div>`;
                  }).join("")}
                </div>
              </details>

              <!-- Structural measurements -->
              <details>
                <summary class="cursor-pointer text-xs font-semibold text-slate-600 hover:text-slate-900 select-none">
                  All 29 structural measurements ▾
                </summary>
                <div class="mt-2 grid sm:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-0.5">
                  ${Object.entries(b.structural || {}).map(([k, v]) => `
                    <div class="flex justify-between text-[11px] border-b border-slate-100 py-0.5">
                      <span class="text-slate-500 font-mono">${esc(k)}</span>
                      <span class="text-slate-800 font-semibold">${num(v, 3)}</span>
                    </div>`).join("")}
                </div>
              </details>
            </div>
          </article>`;
        }).join("")}
        </div>
      </section>`;
  }).join("");
});
