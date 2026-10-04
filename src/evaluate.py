"""
evaluate.py -- one evaluator for every next-word model (n-gram now, RNN/LSTM later).

A model only has to provide

    model.next_word_probs(context_ids) -> np.ndarray of shape (V,), non-negative, summing to 1

where `context_ids` is a list of vocabulary ids. By default (context="sentence") the context is
the CURRENT SENTENCE ONLY: [<s>, w1, ..., wi]. With context="cross" the (last `cross_len` tokens of
the) previous sentences of the same work are prepended, e.g. [..., </s>, <s>, w1, ..., wi].

Metrics on a split (one model call per token of the split, targets = every token after <s>,
including </s>, <unk>, <num> and punctuation):

  ppl        perplexity over ALL targets
  top1/top3/top5/mrr
             over WORD targets only. A word = any vocabulary token except <s>, </s>, <unk>,
             <num> and the six punctuation marks; targets outside that set are skipped.
             Suggestions are ranked among the same word set, so <s> </s> <unk> <num> (and, by
             default, punctuation) can never be suggested. Ties are broken by lower id.
  archaic_*  the same metrics restricted to targets listed in tables/archaic_keywords.csv with
             is_archaic_candidate == True (matched ignoring leading/trailing apostrophes,
             so the keyword "tis" matches the token "'tis")
  ksr        keystroke savings rate, simulated per word target:
               - without prediction a word costs len(word) + 1 keystrokes (letters + space)
               - before every letter (also before the first, and after the last) the top
                 `k_suggest`=3 vocabulary words starting with the typed prefix are shown; if the
                 target is among them one tap (which also inserts the space) completes the word
               - cost with prediction = (#letters typed before the tap) + 1
             KSR = 1 - keys_with / keys_without  (summed over all word targets)
  latency_ms average wall-clock time of one next_word_probs call

evaluate() returns a dict and (log=True) appends a row to results/metrics.csv.

Fast path: a model that also provides `iter_sentence_probs(sentences)` (yielding, per sentence, an
array of shape (len-1, V) with the next-word distribution at every position, computed in ONE pass)
is evaluated with it when context="sentence"; this is how the RNN/LSTM models are scored. Such a
model should also provide `latency_ms(contexts)` (time of a stateless single next_word_probs call).
Options: prefix_completion=False shows suggestions only before the first letter (no word completion),
details=True also returns per-target ranks/positions for further analysis.
"""

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import data as D

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics.csv"
ARCHAIC_CSV = ROOT / "tables" / "archaic_keywords.csv"

COLUMNS = ["timestamp", "model", "n", "split", "context", "hyperparameters",
           "n_tokens", "n_word_targets", "ppl", "top1", "top3", "top5", "mrr", "ksr", "latency_ms",
           "n_archaic_targets", "archaic_ppl", "archaic_top1", "archaic_top3", "archaic_top5",
           "archaic_mrr", "archaic_ksr", "punct_suggestions", "params", "size_mb"]

_PREFIX_CACHE = {}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def word_ids(vocab, allow_punct=False):
    """Ids a keyboard may suggest / that count as word targets (ascending)."""
    banned = set(D.SPECIALS) | (set() if allow_punct else set(D.PUNCT))
    return np.array([i for i, t in enumerate(vocab.itos) if t not in banned], dtype=np.int64)


def prefix_index(vocab, cand_ids):
    """prefix string -> ids (ascending) of candidate words starting with it ('' -> all candidates)."""
    key = (id(vocab), len(cand_ids))
    if key not in _PREFIX_CACHE:
        idx = {}
        for i in cand_ids:
            w = vocab.itos[i]
            for k in range(len(w) + 1):
                idx.setdefault(w[:k], []).append(int(i))
        _PREFIX_CACHE[key] = {p: np.array(v, dtype=np.int64) for p, v in idx.items()}
    return _PREFIX_CACHE[key]


def archaic_set(path=ARCHAIC_CSV):
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8")
    return {w.strip("'") for w in df.loc[df["is_archaic_candidate"].astype(bool), "word"]}


def _rank(p_cand, cand_ids, p_t, target):
    """1 + number of candidates that outrank the target (higher p, or equal p and lower id)."""
    return 1 + int((p_cand > p_t).sum()) + int(((p_cand == p_t) & (cand_ids < target)).sum())


class _Acc:
    """Accumulator for rank metrics + KSR over a set of word targets."""

    def __init__(self):
        self.n = self.t1 = self.t3 = self.t5 = 0
        self.rr = 0.0
        self.keys_with = self.keys_without = 0
        self.nll = 0.0

    def add(self, rank, keys_with, keys_without, logp):
        self.n += 1
        self.t1 += rank <= 1
        self.t3 += rank <= 3
        self.t5 += rank <= 5
        self.rr += 1.0 / rank
        self.keys_with += keys_with
        self.keys_without += keys_without
        self.nll -= logp

    def result(self, prefix=""):
        n = max(self.n, 1)
        return {f"{prefix}top1": self.t1 / n, f"{prefix}top3": self.t3 / n, f"{prefix}top5": self.t5 / n,
                f"{prefix}mrr": self.rr / n,
                f"{prefix}ksr": 1.0 - self.keys_with / max(self.keys_without, 1),
                f"{prefix}ppl": math.exp(self.nll / n)}


def split_sentences_with_works(split, vocab):
    """(sentences as id lists with <s>/</s>, work index per sentence) for a split."""
    sents = D.load_split(split, add_boundaries=True, vocab=vocab)
    idx = D.load_works_index()
    idx = idx[idx["split"] == split]
    work = np.zeros(len(sents), dtype=np.int64)
    for w, (_, r) in enumerate(idx.iterrows()):
        work[r.first_line: r.first_line + r.n_sentences] = w
    return sents, work


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def evaluate(model, split="val", vocab=None, *, context="sentence", cross_len=100, metrics="all",
             k_suggest=3, allow_punct_suggestions=False, sentences=None, works=None,
             archaic_path=ARCHAIC_CSV, name=None, n=None, hyper=None, log=True, metrics_path=None,
             max_sentences=None, check_sums=100, prefix_completion=True, details=False, extra=None, rerank=None):
    """Evaluate `model` on `split` (or on `sentences`, a list of id lists with <s> ... </s>).

    metrics="all" computes everything; metrics="ppl" only perplexity (fast, used for tuning).
    context="sentence" (default, the main comparison) or "cross" (previous sentences of the same
    work are prepended, at most `cross_len` tokens).
    rerank="keystroke" orders the suggestions (stage 6 of src/postprocess.py) by the expected number of
    keystrokes saved, P(w) * (len(w) - len(typed prefix)), instead of by P(w); perplexity is unaffected.
    """
    if rerank not in (None, "keystroke"):
        raise ValueError("rerank must be None or 'keystroke'")
    if context not in ("sentence", "cross"):
        raise ValueError("context must be 'sentence' or 'cross'")
    vocab = vocab or D.load_vocab()
    if sentences is None:
        sentences, works = split_sentences_with_works(split, vocab)
    if works is None:
        works = np.zeros(len(sentences), dtype=np.int64)
    if max_sentences:
        sentences, works = sentences[:max_sentences], works[:max_sentences]
    full = metrics == "all"

    cand = word_ids(vocab, allow_punct_suggestions)
    if full:
        # a *target* counts as a word target only if it is a real word (never punctuation), regardless
        # of whether punctuation may be suggested
        is_word_target = np.zeros(len(vocab), dtype=bool)
        is_word_target[word_ids(vocab, allow_punct=False)] = True
        prefixes = prefix_index(vocab, cand)
        arch = archaic_set(archaic_path)
        is_arch = np.array([t.strip("'") in arch and is_word_target[i] for i, t in enumerate(vocab.itos)])
        tok_len = np.array([len(t) for t in vocab.itos], dtype=np.float64)      # characters of every token

    acc_all, acc_arch = _Acc(), _Acc()
    nll, n_tok, n_calls, call_time = 0.0, 0, 0, 0.0
    history, prev_work = [], None
    perf = time.perf_counter
    fast = context == "sentence" and hasattr(model, "iter_sentence_probs")
    fast_iter = iter(model.iter_sentence_probs(sentences)) if fast else None
    lat_contexts = []                                           # sampled contexts for model.latency_ms
    det = ({"pos": [], "target": [], "rank": [], "sent": [], "keys_with": [], "keys_without": [], "arch": [],
            "tok_sent": [], "tok_logp": []} if (full and details) else None)

    for si, (s, w) in enumerate(zip(sentences, works)):
        if context == "cross" and w != prev_work:
            history = []
        prev_work = w
        if fast:
            t0 = perf()
            P = next(fast_iter)
            call_time += perf() - t0
        for i in range(1, len(s)):
            target = s[i]
            if fast:
                p = P[i - 1]
                if n_tok % 200 == 0 and len(lat_contexts) < 300:
                    lat_contexts.append(list(s[:i]))
            else:
                ctx = s[:i]
                if context == "cross" and history:
                    ctx = history + ctx
                t0 = perf()
                p = model.next_word_probs(ctx)
                call_time += perf() - t0
                n_calls += 1
            if n_tok < check_sums and abs(float(p.sum()) - 1.0) > 1e-6:
                raise ValueError(f"next_word_probs does not sum to 1 (sum={float(p.sum()):.8f})")
            pt = float(p[target])
            logp = math.log(pt) if pt > 0 else -745.0           # ~ log of the smallest double
            nll -= logp
            n_tok += 1
            if det is not None:
                det["tok_sent"].append(si)
                det["tok_logp"].append(logp)
            if not full or not is_word_target[target]:
                continue
            word = vocab.itos[target]
            p_c = p[cand]
            if rerank:                                           # score = P(w) * (len(w) - typed prefix length)
                rank = _rank(p_c * tok_len[cand], cand, pt * len(word), target)
            else:
                rank = _rank(p_c, cand, pt, target)
            first_k = len(word)                                  # letters typed before the tap
            if rank <= k_suggest:
                first_k = 0
            elif prefix_completion:
                for k in range(1, len(word)):
                    ids = prefixes[word[:k]]
                    if rerank:
                        r_k = _rank(p[ids] * (tok_len[ids] - k), ids, pt * (len(word) - k), target)
                    else:
                        r_k = _rank(p[ids], ids, pt, target)
                    if r_k <= k_suggest:
                        first_k = k
                        break
            keys_with, keys_without = first_k + 1, len(word) + 1
            acc_all.add(rank, keys_with, keys_without, logp)
            if is_arch[target]:
                acc_arch.add(rank, keys_with, keys_without, logp)
            if det is not None:
                det["pos"].append(i); det["target"].append(target); det["rank"].append(rank); det["sent"].append(si)
                det["keys_with"].append(keys_with); det["keys_without"].append(keys_without)
                det["arch"].append(bool(is_arch[target]))
        if context == "cross":
            history = (history + list(s))[-cross_len:]

    if fast and hasattr(model, "latency_ms"):
        latency = float(model.latency_ms(lat_contexts))
    else:
        latency = 1000.0 * call_time / max(n_calls, 1)

    res = {"model": name or getattr(model, "name", type(model).__name__),
           "n": n if n is not None else getattr(model, "n", None),
           "split": split, "context": context,
           "hyperparameters": json.dumps(hyper if hyper is not None else
                                         (model.hyperparams() if hasattr(model, "hyperparams") else {})),
           "n_tokens": n_tok, "ppl": math.exp(nll / n_tok),
           "latency_ms": latency, "punct_suggestions": allow_punct_suggestions}
    if full:
        r = acc_all.result()
        r.pop("ppl")                                             # ppl above is over ALL tokens
        res.update(r)
        res["n_word_targets"] = acc_all.n
        res["n_archaic_targets"] = acc_arch.n
        res.update(acc_arch.result("archaic_"))
    if extra:
        res.update(extra)
    if det is not None:
        res["details"] = {k: np.asarray(v) for k, v in det.items()}
    if log:
        _append_row(res, Path(metrics_path) if metrics_path else METRICS_CSV)
    return res


def _append_row(res, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: res.get(c, "") for c in COLUMNS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8") as f:
            header = next(csv.reader(f))
        if header != COLUMNS:                                   # new columns were added: migrate old rows
            import pandas as pd
            pd.read_csv(path).reindex(columns=COLUMNS).to_csv(path, index=False)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------------------
# scoring against the REAL (unmapped) target words - used for cross-domain evaluation
# --------------------------------------------------------------------------------------
def evaluate_unmapped(model, vocab, token_sentences, *, k_suggest=3, archaic=None, name=None,
                      check_sums=50, details=False):
    """Score `model` (which has its own `vocab`) on tokenised sentences (lists of strings, no <s>/</s>).

    Contexts are mapped to the model's vocabulary (unknown words -> <unk>), but the TARGETS are the real
    words. A target is a *word target* if it is not punctuation, <num> or </s>; the denominator is ALL word
    targets, so a target missing from the model's vocabulary counts as a miss (and saves no keystrokes).
    No perplexity (it is not comparable across vocabularies). Returns top1/top3/top5/mrr/ksr, the OOV rate of
    word targets, and - if `archaic` (a set of keywords, apostrophes stripped) is given - the same metrics on
    the archaic targets.
    """
    cand = word_ids(vocab)
    cand_set = set(cand.tolist())
    prefixes = prefix_index(vocab, cand)
    fast = hasattr(model, "iter_sentence_probs")
    id_sents = [[vocab.bos_id] + vocab.encode(s) + [vocab.eos_id] for s in token_sentences]
    it = iter(model.iter_sentence_probs(id_sents)) if fast else None
    special = set(D.SPECIALS) | set(D.PUNCT)
    acc, acc_arch = _Acc(), _Acc()
    n_words = n_oov = 0
    det = ({"target_text": [], "rank": [], "keys_with": [], "keys_without": [], "sent": [], "pos": []}
           if details else None)
    for si, toks in enumerate(token_sentences):
        ids = id_sents[si]
        P = next(it) if fast else None
        for i, t in enumerate(toks, start=1):
            if t in special:
                continue
            p = P[i - 1] if P is not None else model.next_word_probs(ids[:i])
            if n_words < check_sums and abs(float(p.sum()) - 1.0) > 1e-6:
                raise ValueError("next_word_probs does not sum to 1")
            n_words += 1
            tid = vocab.stoi.get(t)
            if tid is None or tid not in cand_set:                       # missing from the model vocabulary: a miss
                n_oov += 1
                rank, keys_with = 10 ** 9, len(t) + 1
            else:
                pt = float(p[tid])
                rank = _rank(p[cand], cand, pt, tid)
                first_k = len(t)
                if rank <= k_suggest:
                    first_k = 0
                else:
                    for k in range(1, len(t)):
                        sub = prefixes[t[:k]]
                        if _rank(p[sub], sub, pt, tid) <= k_suggest:
                            first_k = k
                            break
                keys_with = first_k + 1
            acc.add(rank, keys_with, len(t) + 1, 0.0)
            if archaic is not None and t.strip("'") in archaic:
                acc_arch.add(rank, keys_with, len(t) + 1, 0.0)
            if det is not None:
                det["target_text"].append(t)
                det["rank"].append(rank)
                det["keys_with"].append(keys_with)
                det["keys_without"].append(len(t) + 1)
                det["sent"].append(si)
                det["pos"].append(i)
    res = {"model": name or getattr(model, "name", "model"), "n_word_targets": acc.n,
           "oov_word_rate": n_oov / max(n_words, 1)}
    r = acc.result()
    r.pop("ppl")
    res.update(r)
    if archaic is not None:
        res["n_archaic_targets"] = acc_arch.n
        ra = acc_arch.result("archaic_")
        ra.pop("archaic_ppl")
        res.update(ra)
    if det is not None:
        res["details"] = {k: np.asarray(v) for k, v in det.items()}
    return res
