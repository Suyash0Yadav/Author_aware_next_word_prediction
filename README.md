# Author-aware next-word prediction: Shakespeare keyboard autocomplete

University NLP project. A keyboard-style next-word predictor and word completer trained on Shakespeare's complete
works (Project Gutenberg #100), comparing **n-gram language models** (Kneser-Ney, add-one, unigram) with an
**RNN and an LSTM**, plus an interpolated ensemble, and a **modern-English baseline** (WikiText-2) to show how badly
autocomplete trained on the wrong domain does. Includes the postprocessing pipeline between raw model output and what
the user sees, and an offline Flask website ("The Bard's Keyboard") with a results dashboard for every phase and the keyboard demo.

**Data:** 44 works, 963,240 words → 68,379 sentences, vocabulary of 14,713 tokens (train only, `min_freq = 2`),
split *by work* 36 / 4 / 4 (train / val / test).

## Headline results (test set, sentence-only context)

| model | perplexity | top-1 | top-3 | keystroke savings | archaic top-1 |
|---|---:|---:|---:|---:|---:|
| KN bigram | 147.4 | 10.8 % | 19.9 % | 46.7 % | 4.5 % |
| KN trigram | 134.7 | 12.6 % | 22.5 % | 47.7 % | 6.5 % |
| Vanilla RNN | 112.9 | 12.8 % | 22.9 % | 48.8 % | 8.4 % |
| **LSTM** | **97.6** | 13.9 % | 24.8 % | 49.8 % | 9.9 % |
| Ensemble (0.8 LSTM + 0.2 KN3) | 92.8 | 14.3 % | 25.5 % | 50.2 % | 9.5 % |

Seed variation, bootstrap intervals, the domain-transfer experiment and the reranking experiment are in notebooks
05, 06 and 08. **Every number that appears in more than one place, with its source file, is in
[`docs/key_numbers.md`](docs/key_numbers.md)** (generated and cross-checked by `docs/make_key_numbers.py`).

## Folder structure

```
get_shakespeare.py        step 1: download Gutenberg #100, strip boilerplate, split into 44 works, metadata, validation
data_stats.py             early helper: quick corpus statistics
01_eda.ipynb              exploratory data analysis, the train/val/test split (writes data/split.csv)
notebooks/
  02_preprocessing.ipynb  what each preprocessing stage does (before/after samples, stage table, OOV, case map)
  03_ngram.ipynb          unigram / Laplace / Kneser-Ney, discount tuning, test (once); 03b = follow-ups
  04_rnn.ipynb            vanilla RNN, LSTM (grid on validation), ensemble, comparison figures, test (once)
  05_robustness.ipynb     3 seeds, paired bootstrap, McNemar, stateful latency
  06_baseline.ipynb       WikiText-2 baseline and cross-domain evaluation
  07_postprocessing.ipynb the postprocessing stages and their evidence
  08_reranking.ipynb      optional keystroke-aware reranking (stage 6)
src/
  preprocess.py           raw works -> sentences/tokens/vocabulary (8 stages, each saved in data/interim/)
  wikitext.py             WikiText-2 with the same rules, subsampled to the Shakespeare train size
  data.py                 loaders (vocabulary, splits, case map) shared by every model
  ngram.py                n-gram models from scratch (numpy): unigram, Laplace, interpolated Kneser-Ney
  rnn.py                  vanilla RNN / LSTM (PyTorch), training, evaluator interface, ensemble, stateful session
  evaluate.py             the single evaluator: perplexity, top-k, MRR, KSR, archaic subset, latency, logging
  stats.py                paired bootstrap, McNemar          generate.py   sampling
  postprocess.py          stages 1-6 between model output and the screen
  train_rnn.py / train_seeds.py / train_baseline.py   training entry points (checkpoints in models/)
app/                      the website: results dashboard + keyboard demo (app.py, site_data.py, pages.py, templates/, static/, demo_prefixes.txt, README.md)
tests/                    143 pytest tests (+ smoke_ui.py: browser test of the website, run by hand)
docs/                     kn_design.md (two deliberate differences from nltk), key_numbers.md (+ generator)
tools/run_notebook.py     execute a notebook from the command line
tables/  figures/         every table (CSV) and figure (PNG, dpi 200) cited in the report
results/metrics.csv       one row per evaluation: model, n, split, hyper-parameters, all metrics, params, size
data/  models/            generated (git-ignored): data/{raw,interim,works,processed,baseline}, trained models
```

## Reproduce everything, in order

Hardware for the times below: Windows 11, 16 CPU threads, NVIDIA RTX 4050 Laptop GPU (6 GB). Python 3.13.

```bash
pip install -r requirements.txt            # pinned versions; torch with CUDA 12.4 (CPU-only: pip install torch==2.6.0)

python get_shakespeare.py                  # ~20 s (download) -> data/raw, data/works, data/metadata.csv
python tools/run_notebook.py 01_eda.ipynb  # ~1 min (+ ~1 min first time: downloads WikiText-2) -> data/split.csv, tables/figures
python src/preprocess.py                   # ~10 s  -> data/interim/stage*, data/processed/{train,val,test}.txt, vocab.json, case_map.json
python -m src.wikitext                     # ~10 s  -> data/processed/wikitext/ (subsampled train, own vocabulary, case map)
python tools/run_notebook.py notebooks/02_preprocessing.ipynb   # ~1 min
python tools/run_notebook.py notebooks/03_ngram.ipynb           # ~4 min  (evaluates TEST once: models/kn*.pkl)
python tools/run_notebook.py notebooks/03b_ngram_followups.ipynb  # ~1 min

python -m src.train_rnn                    # ~50 min GPU: 8 LSTM configurations + 2 vanilla RNNs (models/*.pt)
python tools/run_notebook.py notebooks/04_rnn.ipynb             # ~7 min  (evaluates TEST once per model)
python -m src.train_seeds                  # ~30 min GPU: seeds 1 and 2 of the selected LSTM and RNN
python tools/run_notebook.py notebooks/05_robustness.ipynb      # ~5 min
python -m src.train_baseline               # ~15 min GPU: WikiText LSTM (needs `python -m src.wikitext` first)
python tools/run_notebook.py notebooks/06_baseline.ipynb        # ~7 min
python tools/run_notebook.py notebooks/07_postprocessing.ipynb  # ~4 min (CPU)
python tools/run_notebook.py notebooks/08_reranking.ipynb       # ~2 min

python app/make_demo_prefixes.py           # <1 min -> app/demo_prefixes.txt (needs the models of 04 and 06)
python app/app.py                          # the website (dashboard + keyboard demo) at http://127.0.0.1:5000 (works offline)
python app/benchmark_latency.py            # ~2 min -> tables/app_latency.csv
python docs/make_key_numbers.py            # refreshes docs/key_numbers.md and fails if a consistency check breaks
python -m pytest tests -q                  # ~1 min, 143 tests
```

Notes: total about 2 hours on this machine, of which ~95 minutes is GPU training. Notebooks 03 and 04 evaluate the test
set (deterministically, so a re-run reproduces the same numbers and replaces its own rows in `results/metrics.csv`).
The notebooks can equally be run interactively in Jupyter/VS Code with the `python3` kernel.

## Protocol and disclosures

* **Selection on validation only.** Kneser-Ney discount per order, the LSTM grid (hidden {256, 512} x layers {1, 2} x
  dropout {0.3, 0.5}), the vanilla RNN's dropout, and the ensemble weight (steps of 0.1) were all chosen on validation.
  The test set was scored once per reported model (03, 04); notebook 05 scores *new* models (seeds 1 and 2) and re-scores
  the seed-42 models only to obtain per-sentence statistics (asserted identical to the logged numbers); notebook 06
  scores the cross-domain models; the reranking experiment (08) was **not** evaluated on test because it did not
  improve validation KSR.
* **Same conditions for every model:** same vocabulary, loader and evaluator; context = the current sentence only,
  padded with `<s>`; perplexity over all tokens including `</s>`, `<unk>`, `<num>` and punctuation; accuracy, MRR and
  KSR over word targets only (suggestions are words only: never special tokens, `<num>` or punctuation).
* **WikiText LSTM batch size.** The WikiText LSTM was trained with a smaller batch (**2,048 tokens**) than the
  Shakespeare LSTM (**4,096**): WikiText's 25k-word vocabulary made the logits too large for the 6 GB GPU at 4,096 tokens.
  Model, optimiser, schedule, dropout and early stopping are identical.
* **WikiText comparison is scored on unmapped target words** (an unknown target counts as a miss), so the in-domain
  Shakespeare numbers in notebook 06 are slightly lower than in the main table (which excludes `<unk>` targets).
* **Latencies come in two kinds and are always labelled:** "model call" (one `next_word_probs` call) vs "end-to-end in
  the app" (HTTP round trip incl. postprocessing); see `docs/key_numbers.md`.
* **Known limitations:** single run per model apart from the 3-seed study; the best LSTM sits at the edge of the grid
  (largest hidden size and dropout); about 0.03 % of the training text is residual stage directions (quantified in
  `tables/preprocessing_residue.csv`); `docs/kn_design.md` documents a design choice (raw counts for `<s>`-initial
  n-grams) that is slightly worse (0.4 % perplexity) than the nltk convention in this implementation.
