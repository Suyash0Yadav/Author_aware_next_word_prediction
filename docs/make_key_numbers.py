"""Build docs/key_numbers.md: every number that appears in more than one place, with its single source of truth.

    python docs/make_key_numbers.py

Values are READ from the source files (tables/*.csv, data/processed/*, results/metrics.csv), the report-critical
identities are CHECKED (and the script fails loudly if one is violated), and the notebooks / README / docs are
searched for the formatted value (column "also quoted in") and for known stale values.
"""
import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = lambda name: pd.read_csv(ROOT / "tables" / name)
P = ROOT / "data" / "processed"

# ------------------------------------------------------------------------------------------ read sources
vocab = json.loads((P / "vocab.json").read_text(encoding="utf-8"))
wt_info = json.loads((P / "wikitext" / "info.json").read_text(encoding="utf-8"))
wt_vocab = json.loads((P / "wikitext" / "vocab.json").read_text(encoding="utf-8"))
stages = T("preprocessing_stages.csv").set_index("stage")
split = T("split_summary.csv").set_index("split")
idx = pd.read_csv(P / "works_index.csv")
overview = T("corpus_overview.csv").set_index("metric")["value"]
oov = T("preprocessing_oov.csv").set_index("split")
final = T("rnn_test_results.csv").set_index("model")
ngram_test = T("ngram_test_results.csv").set_index("model")
seeds = T("robust_seeds_summary.csv").set_index("model")
lat_rob = T("robust_latency.csv").set_index("prefix length")
lat_app = T("app_latency.csv").set_index("model")
cross = T("baseline_cross_eval.csv")
resid = T("preprocessing_residue.csv")
val = T("ngram_val_results.csv").set_index("model")
metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
files = {k: T(k + ".csv") for k in ()}

n = lambda x: f"{int(round(x)):,}"
f2 = lambda x: f"{x:.2f}"
pc = lambda x, d=1: f"{100 * x:.{d}f} %"
sents = idx.groupby("split")["n_sentences"].sum()
tokens = {sp: sum(len(l.split()) for l in (P / f"{sp}.txt").read_text(encoding="utf-8").splitlines()) for sp in ("train", "val", "test")}

# ------------------------------------------------------------------------------------------ checks
checks = []


def check(name, cond):
    checks.append((name, bool(cond)))


check("vocab.json size == stage-table final vocabulary", vocab["size"] == int(stages.loc[9, "vocab_size"]) == len(vocab["itos"]))
check("sentences: works_index == stage-table (stage 6 = 9) == lines of train/val/test.txt",
      int(sents.sum()) == int(stages.loc[6, "n_sentences"]) == int(stages.loc[9, "n_sentences"]) ==
      sum(len((P / f"{s}.txt").read_text(encoding="utf-8").splitlines()) for s in ("train", "val", "test")))
check("tokens: train+val+test == stage-table stage 9", sum(tokens.values()) == int(stages.loc[9, "n_tokens"]))
check("split words add up to the corpus total", int(split["num_words"].sum()) == int(overview["total words (whitespace split)"]))
check("raw words: stage 0 == corpus_overview", int(stages.loc[0, "n_tokens"]) == int(overview["total words (whitespace split)"]))
check("test positions = test tokens + test sentences (</s>)", tokens["test"] + int(sents["test"]) == 108_336)
check("WikiText train subsample == Shakespeare train tokens", wt_info["wikitext_train_tokens"] == tokens["train"] == wt_info["shakespeare_train_tokens"])
check("WikiText vocabulary size == info.json", wt_vocab["size"] == wt_info["vocab_size"])
check("metrics.csv test perplexity of the LSTM == tables/rnn_test_results.csv",
      abs(float(metrics[(metrics.model == "lstm_h512_l1_d0.5") & (metrics.split == "test")]["ppl"].iloc[0]) - float(final.loc["LSTM", "ppl"])) < 1e-6)
check("KN trigram test perplexity identical in ngram_test_results and rnn_test_results",
      abs(float(ngram_test.loc["kn3", "ppl"]) - float(final.loc["KN trigram", "ppl"])) < 1e-9)

# ------------------------------------------------------------------------------------------ the table
rows = []


def add(group, quantity, value, source, note=""):
    rows.append({"group": group, "quantity": quantity, "value": value, "source": source, "note": note})


add("data", "raw corpus size (whitespace words)", n(overview["total words (whitespace split)"]), "tables/corpus_overview.csv", "stage 0 of tables/preprocessing_stages.csv")
add("data", "works", n(overview["number of works"]), "tables/corpus_overview.csv")
add("data", "split by work (train / val / test)", "36 / 4 / 4 works", "data/split.csv")
add("data", "split by words (train / val / test)", " / ".join(n(split.loc[s, "num_words"]) for s in ("train", "val", "test")), "tables/split_summary.csv")
add("data", "sentences (total)", n(sents.sum()), "data/processed/works_index.csv", "= stage 6 and stage 9 of tables/preprocessing_stages.csv")
add("data", "sentences (train / val / test)", " / ".join(n(sents[s]) for s in ("train", "val", "test")), "data/processed/works_index.csv")
add("data", "tokens, no <s>/</s> (train / val / test)", " / ".join(n(tokens[s]) for s in ("train", "val", "test")), "data/processed/{train,val,test}.txt", f"total {n(sum(tokens.values()))}")
add("data", "evaluated positions: val / test (tokens + </s>)", f"{n(tokens['val'] + sents['val'])} / {n(tokens['test'] + sents['test'])}", "src/evaluate.py (n_tokens in results/metrics.csv)")
add("vocabulary", "FINAL Shakespeare vocabulary (train, min_freq 2, incl. 4 special + 6 punctuation)", n(vocab["size"]), "data/processed/vocab.json", "the number to quote")
add("vocabulary", "  ... of which words", n(vocab["size"] - 10), "data/processed/vocab.json")
add("vocabulary", "EDA vocabulary (regex tokens, lowercase, whole corpus, no cut)", "26,909", "tables/vocabulary_stats.csv", "a different definition - do not mix with the final vocabulary")
add("vocabulary", "raw whitespace types (case-sensitive)", n(stages.loc[0, "vocab_size"]), "tables/preprocessing_stages.csv", "stage 0")
add("vocabulary", "<unk> rate val / test (all tokens)", f"{f2(oov.loc['val', 'unk_pct_all_tokens'])} % / {f2(oov.loc['test', 'unk_pct_all_tokens'])} %", "tables/preprocessing_oov.csv")
add("vocabulary", "<unk> rate val / test (word tokens)", f"{f2(oov.loc['val', 'unk_pct_word_tokens'])} % / {f2(oov.loc['test', 'unk_pct_word_tokens'])} %", "tables/preprocessing_oov.csv")
add("vocabulary", "WikiText-2 vocabulary (own, train subsample, min_freq 2)", n(wt_vocab["size"]), "data/processed/wikitext/vocab.json")
add("vocabulary", "WikiText-2 train / val / test tokens", f"{n(wt_info['wikitext_train_tokens'])} / {n(wt_info['val_tokens'])} / {n(wt_info['test_tokens'])}", "data/processed/wikitext/info.json")
add("preprocessing", "residue after cleaning (tokens, % of text)", f"{n(resid.iloc[-1]['direction_like_tokens'])} tokens = {resid.iloc[-1]['pct_of_tokens']:.3f} %", "tables/preprocessing_residue.csv")

for key, label in (("KN bigram", "KN bigram"), ("KN trigram", "KN trigram"), ("Vanilla RNN", "vanilla RNN"), ("LSTM", "LSTM"), ("Ensemble (LSTM+KN3)", "ensemble")):
    r = final.loc[key]
    add("test results", f"{label}: perplexity", f2(r["ppl"]), "tables/rnn_test_results.csv", "also results/metrics.csv")
add("test results", "LSTM / KN trigram top-1 | top-3 | KSR", f"{pc(final.loc['LSTM', 'top1'])} | {pc(final.loc['LSTM', 'top3'])} | {pc(final.loc['LSTM', 'ksr'])}  /  {pc(final.loc['KN trigram', 'top1'])} | {pc(final.loc['KN trigram', 'top3'])} | {pc(final.loc['KN trigram', 'ksr'])}", "tables/rnn_test_results.csv", "scored on <unk>-mapped test data, <unk> targets excluded")
add("test results", "same models scored on unmapped words (OOV = miss), top-1 / top-3 / KSR", " / ".join(
    f"{a}: {pc(r.top1)} | {pc(r.top3)} | {pc(r.ksr)}" for a, r in ((x.architecture, x) for x in cross[(cross.trained_on == 'Shakespeare') & (cross.tested_on == 'Shakespeare')].itertuples())), "tables/baseline_cross_eval.csv", "slightly LOWER than the line above, by design")
add("validation", "LSTM / vanilla RNN / KN trigram validation perplexity", f"99.85 / 116.55 / {f2(val.loc['kn3', 'ppl'])}", "tables/rnn_lstm_grid.csv, tables/rnn_vanilla_runs.csv, tables/ngram_val_results.csv")
add("robustness", "LSTM test perplexity, 3 seeds (mean ± std)", seeds.loc["LSTM", "ppl"], "tables/robust_seeds_summary.csv")
add("robustness", "vanilla RNN test perplexity, 3 seeds (mean ± std)", seeds.loc["Vanilla RNN", "ppl"], "tables/robust_seeds_summary.csv")
add("size", "parameters / size on disk: KN trigram, LSTM, RNN, ensemble",
    "; ".join(f"{k}: {final.loc[k, 'params_M']:.2f} M / {final.loc[k, 'size_mb']:.1f} MB" for k in ("KN trigram", "LSTM", "Vanilla RNN", "Ensemble (LSTM+KN3)")),
    "tables/rnn_test_results.csv", "n-gram 'parameters' = stored n-gram types")

# latencies - always labelled
for key, label in (("KN bigram", "KN bigram"), ("KN trigram", "KN trigram"), ("Vanilla RNN", "vanilla RNN"), ("LSTM", "LSTM"), ("Ensemble (LSTM+KN3)", "ensemble")):
    add("latency (MODEL CALL)", f"{label}: one next_word_probs call, CPU, contexts sampled from test sentences", f"{final.loc[key, 'latency_ms']:.3f} ms" if final.loc[key, "latency_ms"] < 1 else f"{final.loc[key, 'latency_ms']:.2f} ms",
        "tables/rnn_test_results.csv (latency_ms)", "stateless: RNN re-reads the prefix; evaluator uses 8 CPU threads")
add("latency (MODEL CALL)", "LSTM stateless vs stateful, all positions of 150 val sentences", f"{lat_rob.loc['all', 'stateless_ms']:.2f} ms vs {lat_rob.loc['all', 'stateful_ms']:.2f} ms", "tables/robust_latency.csv", "stateful = newest token only; independent of prefix length")
for m_ in lat_app.index:
    r = lat_app.loc[m_]
    add("latency (END-TO-END IN THE APP)", f"{m_}: median (p95), localhost HTTP round trip", f"{r.end_to_end_ms_median:.1f} ms ({r.end_to_end_ms_p95:.1f})", "tables/app_latency.csv",
        f"server side incl. postprocessing {r.suggest_ms_median:.1f} ms; bare model call on these short prefixes {r.model_call_ms_median:.2f} ms; 4 CPU threads, Flask dev server")

add("disclosure", "WikiText LSTM batch token budget", "2,048 tokens per batch (Shakespeare LSTM: 4,096)", "src/train_baseline.py (cfg max_tokens)", "larger vocabulary (25k) -> logits too large for 6 GB GPU at 4,096; same model, optimiser, schedule")
add("disclosure", "Ney discount vs tuned (KN trigram)", "0.854 vs 0.9", "tables/ngram_ney_discount.csv, tables/ngram_ney_vs_tuned.csv")

table = pd.DataFrame(rows)

# ------------------------------------------------------------------------------------------ where else are values quoted?
text_sources = {}
for p in list((ROOT / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(p, as_version=4)
    text_sources[f"notebooks/{p.name}"] = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
for p in [ROOT / "README.md"] + list((ROOT / "docs").glob("*.md")):
    if p.exists() and p.name != "key_numbers.md":
        text_sources[str(p.relative_to(ROOT)).replace("\\", "/")] = p.read_text(encoding="utf-8")


def quoted_in(value):
    keys = [t for t in re.findall(r"\d[\d,]*\.?\d*", value) if len(t.replace(",", "")) >= 4 or "." in t]
    pats = [re.compile(r"(?<![\d.,])" + re.escape(k) + r"(?![\d]|,\d)") for k in keys]       # whole numbers only
    hits = sorted({name for name, txt in text_sources.items() for pat in pats if pat.search(txt)})
    return ", ".join(hits)


table["also quoted in"] = [quoted_in(v) for v in table["value"]]

stale = {"14,714": "old vocabulary size", "14,716": "old vocabulary size", "68,380": "old sentence count", "68,438": "old stage-6 count",
         "1,063,822": "old token count", "842,761": "old train token count", "5,357,736": "pre-fix character count"}
stale_hits = [(k, why, name) for k, why in stale.items() for name, txt in text_sources.items() if k in txt]

# ------------------------------------------------------------------------------------------ write
out = ["# Key numbers - single source of truth", "",
       "Generated by `python docs/make_key_numbers.py` from the files named in the *source* column. "
       "Quote numbers from here; if a number changes, rerun the script and fix every place listed under *also quoted in* (that column is a literal text match, so an equal-looking number with a different meaning can appear there).", ""]
out += ["## Consistency checks", ""] + [f"- {'PASS' if ok else '**FAIL**'} - {name}" for name, ok in checks] + [""]
out += ["## Stale values found in notebooks / README / docs", ""] + ([f"- `{k}` ({why}) in {name}" for k, why, name in stale_hits] or ["- none"]) + [""]
for g, grp in table.groupby("group", sort=False):
    out += [f"## {g}", "", "| quantity | value | source | also quoted in | note |", "|---|---|---|---|---|"]
    for _, r in grp.iterrows():
        out.append(f"| {r['quantity']} | {r['value']} | `{r['source']}` | {r['also quoted in']} | {r['note']} |")
    out.append("")
(ROOT / "docs" / "key_numbers.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[: 6 + len(checks) + 4 + max(1, len(stale_hits))]))
if not all(ok for _, ok in checks):
    sys.exit("consistency check FAILED")
