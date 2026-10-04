"""Text inspector: colour every word of a passage by the rank the selected model gave it.

The passage is normalised and tokenised with the SAME functions as the training data (src.postprocess.tokenize_spans,
which reuses the preprocessing patterns); the context of every word is the current sentence only, like in the
evaluation. A word that is not in the model's vocabulary counts as a miss, like in the unmapped cross-domain evaluation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate as E, postprocess as PP  # noqa: E402

MAX_WORDS = 300


def colour(rank):
    if rank is None:
        return "r"
    return "g" if rank <= 1 else "y" if rank <= 3 else "o" if rank <= 10 else "r"


def inspect_passage(entry, text, max_words=MAX_WORDS):
    """entry: app.Entry (model, vocab, ...). Returns {segments, shares, n_words, truncated}."""
    vocab, model = entry.vocab, entry.model
    norm, spans = PP.tokenize_spans(text)
    n_seen, cut, truncated = 0, len(norm), False
    for s, e, tok, kind in spans:                                      # keep at most `max_words` words
        if kind == "word":
            n_seen += 1
            if n_seen > max_words:
                cut, truncated = s, True
                break
    spans = [sp for sp in spans if sp[1] <= cut]
    norm = norm[:cut]

    sentences, cur = [], []                                            # the context restarts after . ! ?
    for i, (_, _, tok, _) in enumerate(spans):
        cur.append(i)
        if tok in PP.TERMINAL:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    cand = E.word_ids(vocab)
    cand_set = set(cand.tolist())
    ids_sents = [[vocab.bos_id] + vocab.encode([spans[i][2] for i in sent]) + [vocab.eos_id] for sent in sentences]
    if hasattr(model, "iter_sentence_probs"):
        matrices = list(model.iter_sentence_probs(ids_sents))
    else:
        matrices = [np.stack([model.next_word_probs(ids[:j + 1]) for j in range(len(ids) - 1)]) for ids in ids_sents]

    info = {}
    for sent, P in zip(sentences, matrices):
        for j, si in enumerate(sent):
            _, _, tok, kind = spans[si]
            if kind != "word":
                continue
            p = P[j]                                                   # distribution of the j-th token of the sentence
            tid = vocab.stoi.get(tok)
            top3 = cand[np.lexsort((cand, -p[cand]))[:3]]
            if tid is None or tid not in cand_set:
                rank, prob = None, None
            else:
                rank, prob = E._rank(p[cand], cand, float(p[tid]), tid), float(p[tid])
            info[si] = {"rank": rank, "prob": prob, "top3": [vocab.itos[t] for t in top3], "oov": tid is None}

    segments, pos = [], 0
    for si, (s, e, tok, kind) in enumerate(spans):
        if s > pos:
            segments.append({"t": norm[pos:s], "k": "gap"})
        seg = {"t": norm[s:e], "k": kind}
        if kind == "word":
            r = info[si]
            seg.update({"rank": r["rank"], "prob": r["prob"], "c": colour(r["rank"]), "top3": r["top3"], "oov": r["oov"]})
        segments.append(seg)
        pos = e
    if pos < len(norm):
        segments.append({"t": norm[pos:], "k": "gap"})

    words = [seg for seg in segments if seg["k"] == "word"]
    n = len(words)
    share = lambda f: (sum(1 for w in words if f(w["rank"])) / n) if n else 0.0
    return {"segments": segments, "n_words": n, "truncated": truncated, "max_words": max_words,
            "shares": {"top1": share(lambda r: r is not None and r <= 1), "top3": share(lambda r: r is not None and r <= 3),
                       "top10": share(lambda r: r is not None and r <= 10),
                       "missed": share(lambda r: r is None or r > 10)}}
