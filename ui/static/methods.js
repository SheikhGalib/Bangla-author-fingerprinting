// Methods & Code page.
//
// Each step is explanation → code → the output that code actually printed.
// The outputs are transcribed from the seeded pipeline runs, so what is shown
// here is what the scripts produce, not an idealised version of it.

document.addEventListener("DOMContentLoaded", () => {
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------------------------------------------------------------- steps
  const STEPS = [
    {
      n: 1,
      title: "Collect the corpus, and hold one book back",
      why: "The decision that makes every later number meaningful. For each author, two books go to training and one is held out — chosen before anything was downloaded. The held-out book is the test set, so it cannot be contaminated by accident.",
      file: "scripts/01_build_corpus.py",
      code: `BOOKS = {
  "tagore": [
    Book("galpaguchchha", "গল্পগুচ্ছ",        url, "train"),
    Book("sadhana",       "সাধনা",            url, "train"),
    Book("rahasya",       "রহস্য সমগ্র",      url, "unseen"),   # <- test set
  ],
  ...
}

# Each book is an index page linking to chapter pages, in reading order.
index  = client.get(book.url)
for title, url in chapter_links(index):
    text = chapter_text(client.get(url))   # keep Bangla prose paragraphs only`,
      out: `train    1,406,173 chars over 100 chapters
unseen     565,190 chars over  31 chapters`,
    },
    {
      n: 2,
      title: "Catch the anthology that would have ruined it",
      why: "Bengali publishers reprint stories in later compilations. Tagore's held-out book turned out to reprint six stories from one of his training books. Comparing titles misses this (punctuation differs); comparing text catches it.",
      file: "src/banglastylo/overlap.py",
      code: `def containment(a, b):
    """Fraction of a's 10-word shingles that also occur in b."""
    return len(a & b) / len(a) if a else 0.0

# Stride MUST be 1. At stride 5 the two printings fall out of phase —
# one extra word shifts every later window — and identical stories
# scored only 0.10-0.32, slipping under the threshold.
SHINGLE_N, SHINGLE_STRIDE = 10, 1`,
      out: `6 training chapter(s) reprinted in a held-out book:
  গল্পগুচ্ছ / অপরিচিতা               -> রহস্য সমগ্র / অপরিচিতা        (0.99)
  গল্পগুচ্ছ / কঙ্কাল                  -> রহস্য সমগ্র / কঙ্কাল           (0.87)
  গল্পগুচ্ছ / পুত্রযজ্ঞ                -> রহস্য সমগ্র / পুত্রযজ্ঞ         (0.81)
  গল্পগুচ্ছ / খোকাবাবুর প্রত্যাবর্তন  -> রহস্য সমগ্র / ...              (0.79)
  গল্পগুচ্ছ / সম্পত্তি-সমর্পণ         -> রহস্য সমগ্র / সম্পত্তি সমর্পণ  (0.72)
  গল্পগুচ্ছ / ক্ষুধিত পাষাণ           -> রহস্য সমগ্র / ক্ষুধিত পাষাণ    (0.67)

every unrelated pair scored exactly 0.000
-> removed from the TRAINING side; held-out books are never edited`,
    },
    {
      n: 3,
      title: "Cut into passages and fix the split",
      why: "Passages of ~120 words, cut on sentence boundaries, never crossing a chapter. Validation is held out by whole chapter so early stopping is never tuned on text the model effectively saw.",
      file: "scripts/02_make_splits.py",
      code: `split = splits.split_by_role(passages)   # test = the 3 unseen books
split.test = balance_test(split.test)   # equal per author -> chance = 1/3

rep = splits.leakage_report(split)
if not rep["clean"]:
    raise SystemExit("LEAKAGE DETECTED - refusing to write the split")`,
      out: `author       train    val   test   test book
tagore         866    169    179   রহস্য সমগ্র
nazrul         324     70    179   ব্যথার দান
humayun        193     37    179   হিমুর হাতে কয়েকটি নীলপদ্ম
TOTAL         1383    276    537

leakage check: clean
chance accuracy on the balanced test set: 0.333`,
    },
    {
      n: 4,
      title: "Model 1 — Naive Bayes over word counts",
      why: "The classic Bayes' rule classifier, written out in NumPy rather than imported. It asks: under which author's word-frequency model is this passage most likely? Add-α smoothing stops a single unseen word from vetoing an author outright.",
      file: "src/banglastylo/classifiers.py",
      code: `# log P(c) - the prior, straight from class frequencies
self.log_prior_ = np.log(counts / counts.sum())

# log P(w|c) - summed word counts per class, then add-alpha smoothing
feat += self.alpha
self.log_likelihood_ = np.log(feat / feat.sum(axis=1, keepdims=True))

# Everything in log space: a few hundred probabilities multiplied
# together underflow float64 long before the comparison matters.
def joint_log_likelihood(self, X):
    return X @ self.log_likelihood_.T + self.log_prior_`,
      out: `Naive Bayes (word BoW)       acc=0.9665  macroF1=0.9661   20000 word features
Naive Bayes (char n-gram)    acc=0.9441  macroF1=0.9440    3000 char features`,
    },
    {
      n: 5,
      title: "Model 2 — TF-IDF unigrams + linear SVM",
      why: "Same words, but weighted: a word common in one author and rare elsewhere counts for more. Also the project's topic control — word unigrams are the most topic-sensitive representation there is, so this score estimates how much of the task is solvable by vocabulary alone.",
      file: "scripts/03_train_local.py",
      code: `bow = make_pipeline(
    TfidfVectorizer(max_features=20000, token_pattern=r"\\S+"),
    LinearSVC(random_state=SEED, dual="auto", max_iter=5000))
bow.fit(xtr, ytr)`,
      out: `TF-IDF words + SVM           acc=0.9683  macroF1=0.9682
SVM (all 4 stylometric families)  acc=0.9702
  ├─ char n-grams only            acc=0.9683   <- this family carries it
  ├─ structural (29) only         acc=0.8994
  ├─ function words only          acc=0.8920
  └─ POS n-grams only             acc=0.7374`,
    },
    {
      n: 6,
      title: "Model 3 — BiLSTM, trained from scratch",
      why: "The first model that can see word order. Padding is masked out of both the recurrence and the pooling, otherwise short passages get their signal diluted. It starts from random weights and has only six books, which is exactly why it underperforms.",
      file: "src/banglastylo/neural.py",
      code: `def forward(self, x):
    mask = (x != PAD).unsqueeze(-1)
    h, _ = self.lstm(self.emb(x))          # [B, T, 2H]
    h = h.masked_fill(~mask, 0.0)
    mean = h.sum(1) / mask.sum(1).clamp(min=1)   # padding excluded
    mx   = h.masked_fill(~mask, -1e9).max(1).values
    return self.head(torch.cat([mean, mx], dim=1))`,
      out: `epoch  25/25 loss=0.0001  val_acc=0.9710
restored best checkpoint (val_acc=0.9710)
bilstm test acc = 0.8436

note: val_acc 0.97 but test 0.84 — validation comes from the training
BOOKS, so it measures generalising across chapters, not across books.`,
    },
    {
      n: 7,
      title: "Model 4 — BanglaBERT, fine-tuned on a GPU",
      why: "Same task, but starting from an encoder already pretrained on a large Bangla corpus. Class weights are needed because the training side is deliberately unbalanced. Trained as a Kaggle kernel on a Tesla T4.",
      file: "kaggle/train_gpu.py",
      code: `model = AutoModelForSequenceClassification.from_pretrained(
    "csebuetnlp/banglabert", num_labels=len(classes))   # 3 authors

# Training is unbalanced on purpose, so reweight the loss
# rather than throw away Tagore's extra material.
weights = counts.sum() / (len(classes) * counts)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)`,
      out: `torch 2.10.0+cu128  cuda=True  Tesla T4
class weights: {'humayun': 2.388, 'nazrul': 1.423, 'tagore': 0.532}
  epoch 1/4 loss=0.6543 val_acc=0.9819
  epoch 2/4 loss=0.1288 val_acc=0.9964
  epoch 3/4 loss=0.0355 val_acc=1.0000
  epoch 4/4 loss=0.0143 val_acc=1.0000
  BanglaBERT test acc = 0.9926`,
    },
    {
      n: 8,
      title: "The experiment that matters most",
      why: "Not a model — a protocol comparison. The same classifier is scored the honest way (whole held-out books) and the naive way (shuffle all passages, split randomly). The gap is the part of an accuracy score that is topic rather than style.",
      file: "scripts/05_report_artifacts.py",
      code: `# honest: train on training books, test on the held-out books
work_acc = accuracy_score(yte, pipe.fit(xtr, ytr).predict(xte))

# naive: pool everything, shuffle, split by passage
rng.shuffle(idx)
rand_acc = accuracy_score(y_test_random, pipe2.predict(x_test_random))`,
      out: `book-disjoint (honest) : 0.9683
random passage split   : 1.0000   <- a PERFECT score
inflation from leakage : +0.0317

A perfect score is the symptom, not the achievement: passages from the
same book land on both sides, so the model identifies the novel by its
character names and then names that novel's author.`,
    },
  ];

  document.getElementById("steps").innerHTML = STEPS.map((s) => `
    <section class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div class="px-5 py-3 bg-slate-50 border-b border-slate-200 flex items-baseline gap-3">
        <span class="w-6 h-6 shrink-0 rounded-full bg-slate-800 text-white text-xs font-bold flex items-center justify-center">${s.n}</span>
        <h3 class="font-bold text-slate-900">${esc(s.title)}</h3>
        <code class="ml-auto text-[11px] text-slate-500">${esc(s.file)}</code>
      </div>
      <div class="p-5 space-y-3">
        <p class="text-sm text-slate-600">${esc(s.why)}</p>
        <div>
          <p class="text-[10px] uppercase tracking-wide text-slate-400 mb-1">Code</p>
          <pre class="code bg-slate-900 text-slate-100 rounded-lg p-3 overflow-x-auto">${esc(s.code)}</pre>
        </div>
        <div>
          <p class="text-[10px] uppercase tracking-wide text-slate-400 mb-1">What it printed</p>
          <pre class="out bg-emerald-50 border border-emerald-200 text-emerald-950 rounded-lg p-3 overflow-x-auto">${esc(s.out)}</pre>
        </div>
      </div>
    </section>`).join("");

  // ------------------------------------------------------------- families
  (async () => {
    try {
      const { families } = await (await fetch("/api/analytics/features")).json();
      document.getElementById("families").innerHTML = families.map((f) => `
        <div class="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div class="flex items-baseline gap-2 mb-1">
            <h3 class="font-bold text-slate-900">${esc(f.name)}</h3>
            <span class="text-xs font-mono bg-slate-100 text-slate-700 rounded px-1.5 py-0.5">${f.count} features</span>
          </div>
          <p class="text-sm text-slate-700">${esc(f.what)}</p>
          <p class="text-xs text-slate-500 mt-1"><strong>Why it is here:</strong> ${esc(f.why)}</p>
          ${f.names ? `
            <details class="mt-2">
              <summary class="cursor-pointer text-xs font-semibold text-slate-600">All ${f.names.length} names ▾</summary>
              <div class="mt-1.5 flex flex-wrap gap-1">
                ${f.names.map((n) => `<code class="text-[10px] bg-slate-100 rounded px-1.5 py-0.5">${esc(n)}</code>`).join("")}
              </div>
            </details>` : ""}
        </div>`).join("");
    } catch { /* page still useful without it */ }
  })();

  // -------------------------------------------------------------- results
  (async () => {
    try {
      const { rows, chance } = await (await fetch("/api/results")).json();
      const HEADLINE = ["Naive Bayes (word BoW)", "TF-IDF words + SVM",
                        "BiLSTM (from scratch)", "BanglaBERT (fine-tuned)"];
      document.getElementById("results").innerHTML = `
        <table class="w-full text-sm">
          <thead class="bg-slate-50 text-slate-600 text-xs uppercase tracking-wide">
            <tr><th class="text-left px-4 py-2">Model</th>
                <th class="text-left px-4 py-2">Family</th>
                <th class="text-right px-4 py-2">Accuracy</th>
                <th class="text-right px-4 py-2">Macro F1</th></tr>
          </thead>
          <tbody>
          ${rows.map((r) => {
            const star = HEADLINE.includes(r.model);
            return `<tr class="border-t border-slate-100 ${star ? "bg-emerald-50/60 font-semibold" : ""}">
              <td class="px-4 py-1.5">${star ? "★ " : ""}${esc(r.model)}</td>
              <td class="px-4 py-1.5 text-xs text-slate-500">${esc(r.family)}</td>
              <td class="px-4 py-1.5 text-right font-mono">${r.accuracy.toFixed(4)}</td>
              <td class="px-4 py-1.5 text-right font-mono text-slate-500">${r.macro_f1.toFixed(4)}</td>
            </tr>`;
          }).join("")}
          <tr class="border-t-2 border-slate-300 text-slate-500">
            <td class="px-4 py-1.5 italic" colspan="2">chance (3 balanced classes)</td>
            <td class="px-4 py-1.5 text-right font-mono">${chance.toFixed(4)}</td><td></td>
          </tr>
          </tbody>
        </table>
        <p class="text-xs text-slate-400 px-4 py-2 border-t border-slate-100">★ = the four models presented in the report.</p>`;
    } catch (err) {
      document.getElementById("results").innerHTML =
        `<p class="text-sm text-slate-400 p-4">Results table not available: ${esc(err.message)}</p>`;
    }
  })();

  // ------------------------------------------------- live representations
  const repOut = document.getElementById("rep-out");
  const table = (title, note, items) => `
    <div>
      <p class="text-xs font-bold text-slate-700">${esc(title)}
        <span class="font-normal text-slate-400">— ${esc(note)}</span></p>
      <div class="mt-1 flex flex-wrap gap-1">
        ${items.map((d) => `
          <span class="inline-flex items-baseline gap-1 bg-slate-100 rounded px-1.5 py-0.5">
            <span class="font-bengali text-[13px]">${esc(d.item)}</span>
            <span class="text-[10px] text-slate-500">${d.count}</span>
          </span>`).join("")}
      </div>
    </div>`;

  async function runRep() {
    const text = document.getElementById("rep-input").value.trim();
    if (!text) return;
    repOut.innerHTML = `<p class="text-sm text-slate-400">Working…</p>`;
    try {
      const resp = await fetch("/api/analytics/represent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const d = await resp.json();
      if (!resp.ok) throw new Error(d.detail || "failed");
      repOut.innerHTML = `
        <p class="text-xs text-slate-500 mb-3">${d.n_tokens} words, ${d.n_unique} unique</p>
        <div class="space-y-3">
          ${table("Bag of Words", "each word and how often it occurs; order thrown away", d.bag_of_words)}
          ${table("Word bigrams", "pairs of adjacent words — a little order", d.word_bigrams)}
          ${table("Word trigrams", "triples; sparser, more specific", d.word_trigrams)}
          ${table("Character trigrams", "␣ marks a word boundary; these see inside words", d.char_trigrams)}
          <details>
            <summary class="cursor-pointer text-xs font-semibold text-slate-600">The 29 structural measurements for this sentence ▾</summary>
            <div class="mt-1.5 grid sm:grid-cols-3 gap-x-4">
              ${Object.entries(d.structural).map(([k, v]) => `
                <div class="flex justify-between text-[11px] border-b border-slate-100 py-0.5">
                  <span class="text-slate-500 font-mono">${esc(k)}</span>
                  <span class="font-semibold">${Number(v).toFixed(3)}</span>
                </div>`).join("")}
            </div>
          </details>
        </div>`;
    } catch (err) {
      repOut.innerHTML = `<p class="text-sm text-rose-600">${esc(err.message)}</p>`;
    }
  }
  document.getElementById("rep-run").addEventListener("click", runRep);
  runRep();
});
