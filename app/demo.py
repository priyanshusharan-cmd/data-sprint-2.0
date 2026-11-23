"""
Live demo: Hindi / Hinglish voice-query intent router (55 intents + class 55 = out of scope).

    .venv\\Scripts\\python.exe app\\demo.py            # http://127.0.0.1:7860
    .venv\\Scripts\\python.exe app\\demo.py --knn      # with the nearest-neighbour vote (v006 pipeline)

The model runs offline on CPU from local files; the Anek fonts load from Google Fonts when online (system fonts
otherwise). The dataset has no intent names, so each intent is shown with its most distinctive words in the official
training data, computed automatically at start-up.

Design: the text is the interface. The query is typed large on a plain page and the answer is set beneath it as
type (a large intent number and the words it stands for). One type family made for Indian scripts (Anek Latin +
Anek Devanagari), black on white, one green accent reserved for the chosen intent; no cards, boxes or shadows.
"""
import argparse
import html
import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import IntentPipeline, OOS  # noqa: E402
from normalize import phonetic_key  # noqa: E402

DEV = re.compile("[ऀ-ॿ]")
LAT = re.compile("[A-Za-z]")

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anek+Devanagari:wdth,wght@75..125,300..700&family=Anek+Latin:wdth,wght@75..125,300..700&display=swap" rel="stylesheet">
<script>
  if (!new URL(location).searchParams.has("__theme")) {
    const u = new URL(location); u.searchParams.set("__theme", "light"); location.replace(u);
  }
</script>
"""

CSS = """
:root, .dark {
  --ink: #000000; --soft: #5E5E5E; --faint: #9A9A9A; --rule: #DADADA; --paper: #FFFFFF; --green: #00704A;
  --body-background-fill: var(--paper); --background-fill-primary: var(--paper); --block-background-fill: transparent;
  --block-border-width: 0px; --block-shadow: none; --layout-gap: 0px; --body-text-color: var(--ink);
}
html, body, gradio-app { overflow-x: hidden; background: var(--paper) !important; }
body, .gradio-container { color: var(--ink); font-family: "Anek Latin", "Anek Devanagari", system-ui, sans-serif !important; }
.gradio-container { max-width: 1240px !important; min-width: 0 !important; width: 100% !important; box-sizing: border-box;
  margin: 0 auto !important; padding: 0 clamp(20px, 4vw, 56px) 64px !important; background: var(--paper) !important; }
.gradio-container .main.fillable { padding-left: 0 !important; padding-right: 0 !important; }
footer { display: none !important; }
.html-container, .prose { padding: 0 !important; }
.gradio-container .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }

.top { display: flex; justify-content: space-between; align-items: baseline; gap: 24px; padding: 28px 0 0;
  font-size: 17px; color: var(--soft); }
.top b { color: var(--ink); font-weight: 600; font-variation-settings: "wdth" 112; font-size: 19px; }

#panes { gap: clamp(32px, 6vw, 96px) !important; align-items: flex-start !important; flex-wrap: nowrap !important; margin-top: 56px; }
#left, #right { min-width: 0 !important; }

/* the query: typed big, a single underline, nothing else */
#query, #query label, #query .input-container { background: transparent !important; padding: 0 !important;
  border: 0 !important; box-shadow: none !important; }
#query textarea { font-family: "Anek Latin", "Anek Devanagari", sans-serif !important; font-size: clamp(30px, 3.6vw, 46px) !important;
  font-weight: 500 !important; line-height: 1.2 !important; color: var(--ink) !important; background: transparent !important;
  border: 0 !important; border-bottom: 2px solid var(--ink) !important; border-radius: 0 !important; box-shadow: none !important;
  padding: 0 0 12px !important; resize: none; }
#query textarea::placeholder { color: var(--faint); }
#query textarea:focus { outline: none; border-bottom-color: var(--green) !important; }
.hint-enter { font-size: 16px; color: var(--soft); margin-top: 12px; }
#route { display: none !important; }

/* the answer, set as type */
.ans { margin-top: 48px; }
.ans .line { display: flex; align-items: flex-end; gap: clamp(16px, 2.4vw, 32px); }
.ans .num { font-size: clamp(110px, 13vw, 176px); line-height: .78; font-weight: 600; font-variation-settings: "wdth" 80;
  color: var(--green); letter-spacing: -2px; animation: in .35s ease-out; }
.ans .words { font-size: clamp(26px, 2.8vw, 38px); line-height: 1.15; font-weight: 500; padding-bottom: 6px; }
.ans .sure { font-size: 19px; color: var(--soft); margin-top: 20px; }
.ans .sure strong { color: var(--ink); font-weight: 600; }
.ans.oos .num { color: var(--faint); }
.ans.oos .words { color: var(--ink); }
.ans .also { margin-top: 28px; font-size: 17px; color: var(--soft); line-height: 1.7; }
.ans .also span { color: var(--ink); }
.ans .also em { font-style: normal; color: var(--faint); }
.ans.empty { font-size: 20px; color: var(--faint); }
@keyframes in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .ans .num { animation: none; } }

/* right column: plain list of example queries */
#right .lead p { font-size: 17px; color: var(--soft); margin: 4px 0 14px; }
#examples { flex-direction: column !important; gap: 0 !important; align-items: flex-start !important; }
#examples button { all: unset; cursor: pointer; display: block; width: 100%; box-sizing: border-box; padding: 9px 0;
  font-family: "Anek Latin", "Anek Devanagari", sans-serif !important; font-size: 19px; line-height: 1.3; color: var(--ink);
  border-bottom: 1px solid var(--rule); text-align: left; }
#examples button:hover { color: var(--green); }
#examples button:focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }
.facts { margin-top: 40px; font-size: 16px; line-height: 1.6; color: var(--soft); max-width: 38ch; }
.facts p { margin: 0 0 10px; }
.facts strong { color: var(--ink); font-weight: 600; }

@media (max-width: 900px) {
  #panes { flex-direction: column !important; flex-wrap: wrap !important; margin-top: 36px; }
  #left, #right { width: 100% !important; }
  .top { flex-direction: column; gap: 4px; }
}
"""


def class_hints(tr, n=4):
    """Most distinctive words per class (lift over corpus frequency) in the training data."""
    keys = tr.query_text.map(phonetic_key)
    cv = CountVectorizer(min_df=3, binary=True)
    X = cv.fit_transform(keys)
    voc = np.array(cv.get_feature_names_out())
    df = np.asarray(X.mean(0)).ravel()
    y = tr.target.values
    hints = {}
    for c in range(56):
        p = np.asarray(X[y == c].mean(0)).ravel()
        hints[c] = ", ".join(voc[np.argsort(-(p * np.log((p + 1e-9) / (df + 1e-9))))[:n]])
    hints[OOS] = "out of scope"
    return hints


def demo_examples(tr, seed=7):
    """Real queries from the official training data: both scripts for a few intents, plus out-of-scope ones."""
    rng = np.random.default_rng(seed)
    dev = tr.query_text.str.contains(DEV.pattern)
    rows = []
    for c in rng.choice(list(range(55)), size=4, replace=False):
        for script in (True, False):
            pool = tr[(tr.target == c) & (dev == script)]
            if len(pool):
                rows.append(pool.query_text.iloc[rng.integers(len(pool))])
    rows += tr[tr.target == OOS].query_text.sample(2, random_state=seed).tolist()
    return rows


def script_of(text):
    d, l = bool(DEV.search(text)), bool(LAT.search(text))
    return "Devanagari and Roman mixed" if d and l else "Devanagari" if d else "Roman Hinglish" if l else "other script"


def answer(r, text, hints):
    e = html.escape
    c, p = r["label"], dict(r["top3"]).get(r["label"], r["top3"][0][1])
    if r["out_of_scope"]:
        words = "No matching action. The assistant says it can't help instead of guessing."
    else:
        words = e(hints[c])
    others = [f"<span>{k}</span> {e(hints[k]) if k != OOS else 'out of scope'} <em>{q:.0%}</em>"
              for k, q in r["top3"] if k != c][:2]
    ms = r["ms"]
    return (f"<div class='ans{' oos' if r['out_of_scope'] else ''}' role='status' aria-live='polite'>"
            f"<div class='line'><div class='num'>{c}</div><div class='words'>{words}</div></div>"
            f"<div class='sure'><strong>{p:.0%}</strong> confident, answered in <strong>{ms:.1f} ms</strong> "
            f"on this laptop's CPU{' (over the 50 ms limit)' if ms >= 50 else ''}. Read as {script_of(text)}.</div>"
            f"<div class='also'>Next closest: {' and '.join(others)}</div></div>")


EMPTY = "<div class='ans empty'>Press Enter to route the query.</div>"


def build(pipe, hints, examples):
    def route(text):
        text = (text or "").strip()
        return answer(pipe.predict(text), text, hints) if text else EMPTY

    p = pipe.params
    with gr.Blocks(title="Intent router") as ui:
        gr.HTML(f"<div class='top'><b>Intent router for Hindi and Hinglish</b>"
                f"<span>{p['total'] / 1e6:.1f}M of 300M parameters</span></div>")
        with gr.Row(elem_id="panes"):
            with gr.Column(scale=8, elem_id="left", min_width=0):
                box = gr.Textbox(elem_id="query", show_label=False, lines=1, max_lines=3, autofocus=True,
                                 placeholder="bhai kal subah ka alarm laga do")
                btn = gr.Button("Route", elem_id="route")
                out = gr.HTML(EMPTY)
            with gr.Column(scale=4, elem_id="right", min_width=0):
                gr.Markdown("Real queries from the training data", elem_classes="lead")
                with gr.Row(elem_id="examples"):
                    chips = [gr.Button(ex) for ex in examples]
                gr.HTML(
                    "<div class='facts'>"
                    f"<p><strong>Model.</strong> MuRIL ({p['encoder']:,} parameters) and a character TF-IDF classifier "
                    f"({p['tfidf_svm']:,}) vote on each query; total {p['total']:,}.</p>"
                    "<p><strong>Accuracy.</strong> Macro-F1 0.915 in 5-fold cross-validation, 0.84464 on the public "
                    "leaderboard.</p>"
                    "<p><strong>Fallback.</strong> When no action is confident enough, the answer is class 55.</p></div>")
        reveal = ("() => { if (window.innerWidth < 900) document.querySelector('.ans')"
                  ".scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'}); }")
        for b, ex in zip(chips, examples):
            b.click(lambda t=ex: (t, route(t)), None, [box, out]).then(None, None, None, js=reveal)
        btn.click(route, box, out)
        box.submit(route, box, out).then(None, None, None, js=reveal)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    print("loading models…", flush=True)
    pipe = IntentPipeline(knn=a.knn)
    print(f"ready: {pipe.params}", flush=True)
    ui = build(pipe, class_hints(tr), demo_examples(tr))
    ui.launch(server_name="127.0.0.1", server_port=a.port, css=CSS, head=HEAD, theme=gr.themes.Base())


if __name__ == "__main__":
    main()
