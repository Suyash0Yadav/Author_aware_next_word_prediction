"""View-models of the pages: which files are read and how they are shown. No result is typed in here."""
import json

import pandas as pd
from markupsafe import Markup

from content import (FIGURES, METRIC_HELP, PIPELINE, PROBLEM, RELATED, SAMPLES, STAGE_TEXT)


def _pct(v):
    return f"{100 * float(v):.1f} %"


def _sci(v):
    return f"{float(v):.2e}"


def _signed(v):
    return f"{float(v):+.3f}"


def overview(S):
    return {"cards": S.overview_cards(), "problem": PROBLEM, "pipeline": PIPELINE}


def dataset(S):
    T = S.table_html
    ov = S.table("corpus_overview")
    vs = S.table("vocabulary_stats")
    works = S.works_split()
    arch = S.table("archaic_keywords")
    if arch is not None:
        arch = arch[arch["is_archaic_candidate"]].head(30)[["rank", "word", "count_shakespeare", "count_wikitext", "rate_ratio", "z_score"]]
    return {
        "overview_tbl": T(ov, labels={"metric": "Quantity", "value": "Value"}, sortable=False, fmt={"value": lambda v: f"{v:,.1f}" if float(v) != int(float(v)) else f"{int(float(v)):,}"}),
        "vocab_tbl": T(vs, labels={"metric": "Quantity", "value": "Value"}, sortable=False, fmt={"value": lambda v: f"{float(v):,.4g}" if float(v) < 100 else f"{int(float(v)):,}"}),
        "dtypes_tbl": T(S.table("metadata_dtypes"), sortable=False),
        "category_tbl": T(S.table("category_summary"), labels={"num_works": "Works", "total_words": "Words", "avg_words_per_work": "Avg. words / work", "pct_of_words": "% of words"},
                          fmt={"pct_of_words": lambda v: f"{float(v):.1f} %"}),
        "split_tbl": T(S.table("split_summary"), labels={"num_works": "Works", "num_words": "Words", "pct_of_words": "% of words"}, fmt={"pct_of_words": lambda v: f"{float(v):.2f} %"}, sortable=False),
        "split_cat_tbl": T(S.table("split_works_per_category"), labels={"category": "Category"}, sortable=False),
        "works": works.to_dict("records") if works is not None else None,
        "sparsity_svg": S.ngram_sparsity_svg(),
        "sparsity_tbl": T(S.table("ngram_uniqueness"), labels={"n": "n", "total_ngrams": "Total", "unique_ngrams": "Unique", "occurring_once": "Seen once", "pct_unique_occurring_once": "% of types seen once"},
                          fmt={"pct_unique_occurring_once": lambda v: f"{float(v):.2f} %"}, sortable=False),
        "buckets_tbl": T(S.table("frequency_buckets"), labels={"bucket": "Occurrences", "word_types": "Word types", "pct_of_vocabulary": "% of vocabulary", "tokens": "Tokens", "pct_of_tokens": "% of tokens"},
                         fmt={"pct_of_vocabulary": lambda v: f"{float(v):.2f} %", "pct_of_tokens": lambda v: f"{float(v):.2f} %"}, sortable=False),
        "archaic_tbl": T(arch, labels={"rank": "Rank", "word": "Word", "count_shakespeare": "Count (Shakespeare)", "count_wikitext": "Count (WikiText-2)", "rate_ratio": "Rate ratio", "z_score": "z-score"},
                         fmt={"rate_ratio": lambda v: f"{float(v):,.0f} x", "z_score": lambda v: f"{float(v):.1f}"}),
    }


def preprocessing(S):
    T = S.table_html
    st = S.stage_table()
    stages = [] if st is None else st[["stage", "name", "description", "unit", "n_tokens", "vocab_size", "n_sentences"]].to_dict("records")
    for r in stages:
        r["n_sentences"] = None if pd.isna(r["n_sentences"]) else int(r["n_sentences"])
    return {
        "stage_tbl": T(st, labels={"stage": "#", "name": "Stage", "description": "What it does", "unit": "Unit", "n_tokens": "Tokens / words", "vocab_size": "Vocabulary", "n_sentences": "Sentences"}, sortable=False),
        "stages_json": stages,
        "samples": S.sample_defs(),
        "residue_tbl": T(S.table("preprocessing_residue"), labels={"residue": "Residue", "count": "Count", "unit": "Unit", "of_which_direction_like": "Direction-like", "direction_like_tokens": "Tokens", "pct_of_tokens": "% of tokens"},
                         fmt={"pct_of_tokens": lambda v: f"{float(v):.4f} %"}, sortable=False),
        "oov_dec_tbl": T(S.table("preprocessing_oov_decomposition"), labels={"step": "Step", "test_word_tokens": "Test word tokens", "oov_pct": "Unknown words (%)"}, fmt={"oov_pct": lambda v: f"{float(v):.2f} %"}, sortable=False),
        "oov_mech_tbl": T(S.table("preprocessing_oov_mechanism"), labels={"group": "Group", "tokens": "Tokens", "pct_oov_vs_train_vocab": "Unknown vs train vocabulary (%)"}, fmt={"pct_oov_vs_train_vocab": lambda v: f"{float(v):.1f} %"}, sortable=False),
        "oov_tbl": T(S.table("preprocessing_oov"), labels={"split": "Split", "tokens": "Tokens", "word_tokens": "Word tokens", "unk_pct_all_tokens": "<unk> (all tokens)", "unk_pct_word_tokens": "<unk> (word tokens, %)",
                                                          "word_types": "Word types", "oov_pct_word_types": "OOV word types (%)"},
                     fmt={"unk_pct_word_tokens": lambda v: f"{float(v):.2f} %", "oov_pct_word_types": lambda v: f"{float(v):.2f} %", "unk_pct_all_tokens": lambda v: f"{float(v):.2f} %"}, sortable=False),
        "rules_tbl": T(S.table("preprocessing_rule_examples"), labels={"rule": "Rule", "input": "Input", "output": "Output"}, sortable=False),
    }


MAIN_COLS = ["label", "ppl", "top1", "top3", "top5", "mrr", "ksr", "archaic_top1", "archaic_top5", "latency_ms", "params", "size_mb"]


def main_table(S):
    t, notes = S.main_test_table()
    if t is None:
        return S.notice_missing("results/metrics.csv"), notes
    # archaic_top5 is in metrics.csv; params / size come from the same file
    labels = {c: METRIC_HELP[c][0] for c in METRIC_HELP}
    labels["label"] = "Model"
    arrows = {c: (" ↑" if METRIC_HELP[c][1] else " ↓") for c in METRIC_HELP}
    labels = {c: labels[c] + arrows.get(c, "") for c in labels}
    highlight = {c: ("max" if METRIC_HELP[c][1] else "min") for c in METRIC_HELP if c in t}
    fmt = {"mrr": lambda v: f"{float(v):.3f}", "params": lambda v: f"{int(v):,}", "size_mb": lambda v: f"{float(v):.1f}",
           "latency_ms": lambda v: f"{float(v):.3f}" if float(v) < 1 else f"{float(v):.2f}"}
    return S.table_html(t, columns=[c for c in MAIN_COLS if c in t], labels=labels, fmt=fmt, highlight=highlight,
                        tooltips={c: METRIC_HELP[c][2] for c in METRIC_HELP}, cls="main"), notes


def models(S):
    T = S.table_html
    tbl, notes = main_table(S)
    rows = S.table("rnn_samples")
    samples = {}
    if rows is not None:
        for (m, temp), g in rows.groupby(["model", "temperature"], sort=False):
            samples.setdefault(m, []).append({"temperature": temp, "sentences": list(g["sample"])})
    val = S.validation_table()
    qual = S.table("rnn_examples")
    if qual is not None:
        qual["prefix"] = qual["prefix"].astype(str).map(lambda s: "… " + " ".join(s.split()[-9:]) if len(s.split()) > 9 else s)
    return {
        "main_tbl": tbl, "main_notes": notes, "metric_help": METRIC_HELP,
        "seeds_tbl": T(S.table("robust_seeds_summary"), labels={"model": "Model", "val_ppl": "Val. perplexity (validation)", "ppl": "Test perplexity", "top1": "Test top-1", "top3": "Test top-3", "ksr": "Test KSR", "archaic_top1": "Test archaic top-1"}, sortable=False),
        "seeds_runs_tbl": T(S.table("robust_seeds_per_run"), labels={"model": "Model", "seed": "Seed", "run": "Run", "best_epoch": "Best epoch", "val_ppl": "Val. ppl (validation)", "ppl": "Test ppl", "top1": "Test top-1", "top3": "Test top-3", "ksr": "Test KSR", "archaic_top1": "Test archaic top-1"}),
        "boot_tbl": T(S.table("robust_bootstrap"), labels={"comparison": "Comparison", "metric": "Metric", "A": "A", "B": "B", "diff_A_minus_B": "A - B", "ci95_low": "95 % CI low", "ci95_high": "95 % CI high", "rel_diff_pct": "Relative (%)", "ci_excludes_0": "CI excludes 0"},
                      fmt={"A": lambda v: f"{float(v):.4g}", "B": lambda v: f"{float(v):.4g}", "diff_A_minus_B": _signed, "ci95_low": _signed, "ci95_high": _signed, "rel_diff_pct": lambda v: f"{float(v):+.1f} %"}, sortable=False),
        "mcnemar_tbl": T(S.table("robust_mcnemar"), labels={"targets": "Targets", "n": "n", "both_correct": "Both right", "only_LSTM_correct": "Only LSTM right", "only_KN3_correct": "Only KN-3 right", "neither": "Neither", "p_exact": "Exact p", "chi2_cc": "chi2 (cc)", "p_chi2": "p (chi2)"},
                         fmt={"p_exact": _sci, "p_chi2": _sci, "chi2_cc": lambda v: f"{float(v):.1f}"}, sortable=False),
        "latency_tbl": T(S.table("robust_latency"), labels={"prefix length": "Prefix length", "stateful_ms": "Stateful (ms)", "stateless_ms": "Stateless (ms)", "speed_up": "Speed-up"}, fmt={"speed_up": lambda v: f"{float(v):.2f} x", "stateful_ms": lambda v: f"{float(v):.2f}", "stateless_ms": lambda v: f"{float(v):.2f}"}, sortable=False),
        "qual_tbl": T(qual, columns=["kind", "play", "prefix", "actual_next", "KN trigram top-5", "KN trigram rank", "LSTM top-5", "LSTM rank", "Ensemble top-5", "Ensemble rank"], labels={"kind": "Kind", "play": "Play", "prefix": "Prefix", "actual_next": "Actual next"}, sortable=False, cls="wide"),
        "samples": samples,
        "val_tbl": T(val, labels={"model": "Run", "n": "n", "hyperparameters": "Hyper-parameters", "context": "Context", "ppl": "Perplexity (val)", "top1": "Top-1 (val)", "top3": "Top-3 (val)", "ksr": "KSR (val)", "archaic_top1": "Archaic top-1 (val)", "latency_ms": "Latency ms"}),
        "grid_tbl": T(S.table("rnn_lstm_grid"), labels={"run": "LSTM run", "hidden": "Hidden", "layers": "Layers", "dropout": "Dropout", "params": "Parameters", "epochs_run": "Epochs", "best_epoch": "Best epoch", "best_val_ppl": "Best val. perplexity (validation)", "train_ppl_eval_at_best": "Train ppl at best", "sec_per_epoch": "s / epoch"}, highlight={"best_val_ppl": "min"}),
        "rnn_tbl": T(S.table("rnn_vanilla_runs"), labels={"run": "RNN run", "hidden": "Hidden", "layers": "Layers", "dropout": "Dropout", "params": "Parameters", "epochs_run": "Epochs", "best_epoch": "Best epoch", "best_val_ppl": "Best val. perplexity (validation)", "sec_per_epoch": "s / epoch"}, highlight={"best_val_ppl": "min"}),
        "disc_tbl": T(S.table("ngram_discount_tuning"), labels={"n": "n", "d": "Discount d", "val_ppl": "Perplexity (validation)"}, fmt={"val_ppl": lambda v: f"{float(v):.2f}"}, highlight={"val_ppl": "min"}),
        "pos_tbl": T(S.table("rnn_top1_by_position"), labels={"model": "Model", "position": "Position", "n_targets": "Targets", "top1": "Top-1"}),
        "groups_tbl": T(S.table("rnn_top1_archaic_groups"), labels={"model": "Model", "group": "Target group", "n_targets": "Targets", "top1": "Top-1"}, fmt={"top1": _pct}),
    }


def crossdomain(S):
    T = S.table_html
    cx = S.table("baseline_cross_eval")
    return {
        "cross_json": None if cx is None else json.loads(cx.to_json(orient="records")),      # NaN -> null (valid JSON)
        "cross_tbl": T(cx, labels={"architecture": "Model", "trained_on": "Trained on", "tested_on": "Tested on", "n_word_targets": "Word targets", "top1": "Top-1", "top3": "Top-3", "top5": "Top-5", "mrr": "MRR", "ksr": "KSR",
                                   "oov_word_rate": "Unknown targets", "n_archaic_targets": "Archaic targets", "archaic_top1": "Archaic top-1", "archaic_top5": "Archaic top-5"}, fmt={"mrr": lambda v: f"{float(v):.3f}"}),
        "examples_tbl": T(S.table("baseline_examples"), labels={"kind": "Kind", "prefix": "Prefix", "actual_next": "Actual next", "Shakespeare LSTM top-5": "Shakespeare LSTM top-5", "rank_S": "Rank (Shakespeare)", "WikiText LSTM top-5": "WikiText LSTM top-5", "rank_W": "Rank (WikiText)", "in WikiText vocab": "In WikiText vocabulary"},
                          fmt={"rank_S": lambda v: f"{int(v):,}", "rank_W": lambda v: f"{int(v):,}", "in WikiText vocab": lambda v: "yes" if v else "no (cannot be suggested)"}, sortable=False, cls="wide"),
        "indomain_tbl": T(S.table("baseline_in_domain_vs_main"), sortable=False, fmt={"main results table": lambda v: f"{float(v) * 100:.2f} %", "this notebook (unmapped, OOV = miss)": lambda v: f"{float(v) * 100:.2f} %"}),
        "data_tbl": T(S.table("baseline_data_summary"), sortable=False),
    }


def postprocessing(S):
    T = S.table_html
    rv, rt = S.table("rerank_val_results"), S.table("rerank_test_results")
    test_has_rerank = rt is not None and rt["ordering"].str.contains("keystrokes").any()
    expl = None
    if rv is not None:
        expl = rv[["model", "ordering", "archaic_top1", "archaic_ksr"]]
    pc = lambda v: f"{float(v):.1f} %"
    return {
        "stages": STAGE_TEXT,
        "stage_tbl": T(S.table("postproc_stage_table"), sortable=False, cls="wide"),
        "special_tbl": T(S.table("postproc_raw_special_rate"), labels={"model": "Model", "positions": "Test positions"}, fmt={c: (lambda v: f"{float(v):.1f} %") for c in ("raw top-1 is special/punct (%)", "any of raw top-3 is special/punct (%)", "targets that are words (%)")}, sortable=False),
        "concentration_tbl": T(S.table("postproc_prediction_concentration"), fmt={"top-5 tokens' share (%)": pc, "top-10 share (%)": pc}, sortable=False),
        "case_tbl": T(S.table("postproc_case_accuracy"), fmt={c: (lambda v: f"{float(v):.1f} %") for c in ("stage-4 restoration (%)", "all lowercase (%)", "capitalise sentence start only (%)")}, sortable=False),
        "case_adj_tbl": T(S.table("postproc_case_accuracy_adjusted"), fmt={"stage-4 accuracy (%)": pc, "adjusted accuracy (%) - estimate": pc}, sortable=False),
        "rerank_val_tbl": T(rv, labels={"model": "Model", "ordering": "Ordering", "n_word_targets": "Targets"}, columns=[c for c in ("model", "ordering", "top1", "top3", "top5", "mrr", "ksr") if rv is not None and c in rv], fmt={"mrr": lambda v: f"{float(v):.3f}"}, sortable=False),
        "rerank_len_tbl": T(S.table("rerank_val_by_length"), labels={"model": "Model", "ordering": "Ordering", "target length": "Target length", "n": "Targets"}, sortable=False),
        "rerank_expl_tbl": T(expl, labels={"model": "Model", "ordering": "Ordering", "archaic_top1": "Archaic top-1 (val)", "archaic_ksr": "Archaic KSR (val)"}, sortable=False),
        "rerank_test_note": ("Reranked results were evaluated on test." if test_has_rerank else
                             "Not evaluated on the test set: by the pre-declared rule (test only if validation KSR improves) the reranked ordering did not qualify."),
        "rerank_test_tbl": T(rt, labels={"model": "Model", "ordering": "Ordering"}, columns=[c for c in ("model", "ordering", "top1", "top3", "ksr") if rt is not None and c in rt], sortable=False) if rt is not None else None,
        "rerank_examples_tbl": T(S.table("rerank_examples"), sortable=False, cls="wide"),
    }


def about(S):
    readme = S.read_text("README.md")
    key = S.read_text("docs/key_numbers.md")
    return {
        "related": RELATED,
        "limitations": S.md_to_html(S.md_section(readme, "Protocol and disclosures"), heading_shift=2),
        "reproduce": S.md_to_html(S.md_section(readme, "Reproduce everything, in order"), heading_shift=2),
        "key_numbers": S.md_to_html("\n".join(key.splitlines()[1:]) if key else None, heading_shift=2),
    }


BUILDERS = {"overview": overview, "dataset": dataset, "preprocessing": preprocessing, "models": models,
            "cross-domain": crossdomain, "postprocessing": postprocessing, "keyboard": lambda S: {}, "about": about}
