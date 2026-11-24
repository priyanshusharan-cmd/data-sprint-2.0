# Defense prep: live demo and judge questions

## The pipeline in one minute (say this first)
1. **The data problem.** Training queries are mostly Hindi script; the test is mostly Hinglish typed in English
   letters, more casual and English-heavy. A model trained naively on train fails on the test.
2. **Fix 1, data.** We wrote rules that convert each training sentence into test-style Hinglish (English-letter
   spelling, fillers like *bhai*, digits, typos). Rules only: no AI-generated data.
3. **Fix 2, two models that make different mistakes.** A spelling-pattern model (character TF-IDF + linear SVM,
   9.3M parameters) and a language model pretrained on Hinglish (HingRoBERTa, 278M) fine-tuned on our data.
   We average their probabilities 50/50.
4. **Fix 3, the fallback.** Per-class adjustments tuned on cross-validation; class 55 (out of scope) is chosen when
   nothing else fits.
5. **Result.** 287.4M parameters, public macro-F1 0.86786, ~19 ms per query on this laptop, and a notebook that
   reproduces the exact submission.

## 5-minute live demo script
**Start a fresh app 10 minutes early** (close any demo left running from earlier: a process that sat idle for hours,
or was started on battery, answered in 70–1,100 ms in our test): `.venv\Scripts\python.exe app\demo.py`, open
http://127.0.0.1:7860, then **click 10 suggestions to warm it up**. The first few answers after a start take
40–60 ms; after warm-up they settle at ~25–33 ms in the app (~18 ms in `src/infer.py --check`). Laptop plugged in,
best-performance power mode, other apps and browser tabs closed.

| Minute | Show | Say |
|---|---|---|
| 0:00 | The page | "Type any voice-assistant query in Hindi, Hinglish or a mix. It returns one of 55 intents or 'out of scope', with confidence and the time taken." |
| 0:30 | Click a **Devanagari** suggestion, then a **Roman Hinglish** suggestion | "Same model, both scripts. The script line shows how it read the input. The number is the intent; the words next to it are that intent's most typical words in the training data." |
| 1:30 | Click an **out-of-scope** suggestion (e.g. the podcast one) | "Grey 55 means fallback: the assistant says it can't help instead of guessing a wrong action." |
| 2:15 | Let the **judges type** their own queries | Read out the latency each time: "Under 50 ms on CPU." |
| 3:30 | Right column | "287,378,288 parameters, under the 300M limit, checked with your assertion. 0.917 cross-validated, 0.868 public." |
| 4:15 | Mention the notebook | "Our Kaggle notebook retrains everything from train.csv in 20 minutes and gives our submission 100% identically." |

## Likely questions and short answers
**Why HingRoBERTa and not MuRIL or XLM-R?**
It was pretrained on code-mixed Hinglish in both scripts, which is what the test looks like. Over 5 folds it beat
MuRIL on every fold (0.893 vs 0.886) and especially on test-like rows (0.904 vs 0.888). Public: 0.868 vs 0.845.

**Why keep a TF-IDF model next to a transformer?**
They make different mistakes. TF-IDF is strong on exact spellings and typos; the transformer on meaning and new
phrasing. The blend beat each alone (0.917 vs 0.896 and 0.886 cross-validated).

**How did you stay under 300M parameters?**
One encoder (278,086,712) + one SVM (9,291,576) = 287,378,288. We never averaged fold models in a submission
(that would be 1.39B). The notebook prints the organisers' assertion.

**How do you count TF-IDF parameters?**
The SVM's learned weights: 165,920 features × 56 classes + 56 biases. The TF-IDF vocabulary is a lookup table, not
learned weights.

**Did you use any external data or LLMs?**
No. Only train.csv and test.csv, a public open-source pretrained checkpoint, and rule-based augmentation written as
code. No LLM APIs for inference, labels or data; no MASSIVE/SLURP.

**How do you handle out-of-scope queries?**
Class 55 has its own training examples, and the decision rule falls back to 55 when no intent is confident. On
cross-validation the best threshold was 0, so the learned class 55 does most of the work.

**I typed "set an alarm" and it said out of scope. Is that a bug?**
No. In this dataset alarm-*setting* requests are labelled class 55 (it looks like a few intents were merged into
out-of-scope). Our model follows the official labels. Stopping or managing an existing alarm is its own intent.

**How do you know it generalises and isn't overfitted to the public leaderboard?**
All choices were made on 5-fold cross-validation over 16,936 rows, plus a test-like subset selected by a
train-vs-test classifier. Our first four uploads each beat the previous one and the test-like subset ranked them correctly.
Later candidates that looked slightly better locally did not beat v007 on public, so we kept v007 rather than
chasing the public board.

**What did not work?**
Pseudo-labelling (its apparent gain was a leak; leak-free it was zero), a nearest-neighbour vote (too few test
queries have close matches), three-model blends with MuRIL (public 0.856 and 0.858), training 14 epochs (worse on
one of two folds), and int8 quantisation (changed 15% of answers). On the last day: domain-adaptive pretraining
(public 0.864) and two HingRoBERTas together (0.862). All won on our validation, none beat v007 on public.

**Did you try pretraining on your own data?**
Yes. HingRoBERTa continued its fill-in-the-blank training on our train and test query text (no labels) before
fine-tuning. It learned faster (first-epoch score 0.75 vs 0.68) but its gain on a validation fold was small (+0.002), and it
scored 0.864 public vs 0.868, so we kept it only as our second final pick.

**How did you fit two 278M models under 300M?**
Vocabulary trimming: most of HingRoBERTa's parameters are a 250,002-row word table, and our data uses about
10,000 rows. We removed rows for tokens our data never uses (mostly other scripts: Cyrillic, Chinese, Arabic) and
checked that the test outputs stayed bit-identical. Two trimmed encoders + TF-IDF = 299,498,920 parameters.

**Why didn't your later ideas help if they scored higher on validation?**
Near the top, gains of 0.003–0.02 on our checks are smaller than the noise of the 974-query public board, and
our test-like subset is built from training data. Four local wins in a row lost on public, so we stopped and kept
the simplest model, which is also the one that runs in 20 ms and is fully reproducible.

**How is latency under 50 ms on a CPU?**
The trained model runs on ONNX Runtime (same weights, identical answers) with six threads on the laptop's
performance cores. Median ~19 ms for the whole pipeline. A rare query can exceed 50 ms when Windows gives the CPU to
another program, which is why we close other apps for the demo.

**Can you reproduce your submission?**
Yes. `kaggle/reproduce_v007.ipynb` retrains TF-IDF and HingRoBERTa from train.csv with a fixed seed. On a Kaggle T4
it produced a submission 100% identical to our v007 file in 20 minutes.

**What would you do with more time?**
Distil a HingRoBERTa + MuRIL ensemble into one model under the cap; translate-train with an open MT model; better
out-of-scope detection for genuinely novel topics.

## If something goes wrong live
- Page not loading: check the terminal says `ready:`; restart with the command above (takes ~10 s).
- A query shows over 50 ms: say it is a Windows scheduling spike, run it again (median ~19 ms).
- Offline with no internet: the model works; only the web fonts fall back to the system font.
