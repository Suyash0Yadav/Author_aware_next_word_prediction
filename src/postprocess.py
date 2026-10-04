"""
postprocess.py -- from raw model output to what the user sees on the keyboard.

Typed text is processed with the SAME normalisation / tokenizer as the training data
(src/preprocess.py: stage1_normalize, tokenize); nothing is re-implemented here.

Stages (each one is a separate, testable function; `suggest(..., trace=True)` returns the output of every stage)

  1. raw_topk        top-k tokens straight from the model's distribution over the whole vocabulary: may contain
                     <s> </s> <unk> <num>, punctuation; everything lowercase; model probabilities.
  2. filter_tokens   remove special tokens and punctuation (only real words stay); probabilities unchanged.
  3. prefix_filter   if the user is mid-word keep only the words starting with the typed prefix and renormalise
                     the probabilities over the kept words (with no prefix: renormalised over all words).
  4. restore_case    capitalise at the start of a sentence (start of text or after . ! ?) and use case_map.json for
                     proper nouns and words such as "I" and "O"; a capital letter typed by the user is kept.
  5. insert_word     replace the partial word with the chosen word plus a space; insert_punctuation removes the
                     space before . , ; : ! ?
  6. rerank_keystrokes  (optional) order the candidates by the expected keystrokes saved,
                     P(w) * (len(w) - len(typed prefix)), instead of by P(w) alone, so that short frequent words
                     ("the", "a") no longer crowd out longer words that save more keys.

The context of a prediction is the CURRENT SENTENCE ONLY (tokens after the last . ! ?), as in the evaluation.
"""
import re

from . import data as D
from . import evaluate as E
from . import preprocess as P

TERMINAL = P.TERMINAL                                            # {".", "!", "?"}
MAX_CHARS = 1000                                                 # only the tail of very long texts is used


# --------------------------------------------------------------------------------------
# typed text -> (context tokens, partial word, sentence start)
# --------------------------------------------------------------------------------------
def split_partial(text):
    """(text without the partial word, partial word). The partial word is the trailing run of letters and
    apostrophes; text ending in a space / punctuation / digit has no partial word."""
    i = len(text)
    while i > 0 and (text[i - 1].isalpha() or text[i - 1] == "'"):
        i -= 1
    partial = text[i:]
    if not any(c.isalpha() for c in partial):                    # nothing, or only apostrophes
        return text, ""
    return text[:i], partial


def parse_typed(text):
    """Normalise and tokenize typed text with the preprocessing code and return
    (context, partial, sentence_start): context = lowercase tokens of the current sentence (without the partial
    word), partial = lowercase partial word ('' if the user just typed a space/punctuation), sentence_start = True
    if the word being typed / predicted is the first of a sentence."""
    norm, _ = P.stage1_normalize(text[-MAX_CHARS:])
    head, partial = split_partial(norm)
    context = []
    for tok in (t.lower() for t in P.tokenize(head)):
        if tok in TERMINAL:
            context = []                                         # a new sentence starts after . ! ?
        else:
            context.append(tok)
    return context, partial.lower(), len(context) == 0


def tokenize_spans(text):
    """Like preprocess.tokenize but with character positions: returns (normalised text, spans) where each span is
    (start, end, lowercase token, kind) and kind is "word", "num" or "punct". It uses the SAME normalisation and the
    SAME patterns as the training data (P.stage1_normalize, P.DASH_RE, P.NUM_RE, P.TOKEN_RE), so the tokens are
    exactly those of P.tokenize(normalised text) (tests/test_site.py checks this)."""
    norm, _ = P.stage1_normalize(text)
    blank = P.DASH_RE.sub(" ", norm)                                  # 1:1 character replacement: positions are kept
    spans, masked = [], list(blank)
    for m in P.NUM_RE.finditer(blank):
        spans.append((m.start(), m.end(), "<num>", "num"))
        masked[m.start():m.end()] = " " * (m.end() - m.start())
    for m in P.TOKEN_RE.finditer("".join(masked)):
        tok = m.group(0)
        if tok == "<num>":
            continue
        spans.append((m.start(), m.end(), tok.lower(), "punct" if tok in D.PUNCT else "word"))
    spans.sort()
    return norm, spans


# --------------------------------------------------------------------------------------
# stages 1-3 on a probability vector
# --------------------------------------------------------------------------------------
def _order(p, ids):
    """ids sorted by probability (descending), ties by lower id."""
    import numpy as np
    ids = np.asarray(ids)
    return ids[np.lexsort((ids, -p[ids]))]


def raw_topk(p, vocab, k=5):
    """Stage 1: the k most probable tokens of the full distribution (specials and punctuation included)."""
    import numpy as np
    top = _order(p, np.arange(len(vocab)))[:k]
    return [(vocab.itos[i], float(p[i])) for i in top]


def filter_tokens(p, vocab, k=5):
    """Stage 2: remove special tokens and punctuation; probabilities are the model's (not renormalised)."""
    cand = E.word_ids(vocab)
    return [(vocab.itos[i], float(p[i])) for i in _order(p, cand)[:k]]


def prefix_filter(p, vocab, prefix="", k=5):
    """Stage 3: keep the words starting with `prefix` and renormalise over them (no prefix: over all words).
    k=None returns every matching word (needed by the reranking stage)."""
    cand = E.word_ids(vocab)
    if prefix:
        cand = [i for i in cand if vocab.itos[i].startswith(prefix)]
    if len(cand) == 0:
        return []
    z = float(p[cand].sum())
    return [(vocab.itos[i], float(p[i]) / z if z > 0 else 0.0) for i in _order(p, cand)[:k]]


def rerank_keystrokes(cands, prefix_len=0):
    """Stage 6 (optional): sort (token, prob) pairs by expected keystrokes saved, prob * (len(token) - prefix_len);
    ties keep the probability order. A word that is already completely typed saves nothing (score 0)."""
    return sorted(cands, key=lambda tp: -(tp[1] * max(len(tp[0]) - prefix_len, 0)))


# --------------------------------------------------------------------------------------
# stages 4-5 on text
# --------------------------------------------------------------------------------------
def capitalize_first_letter(word):
    for i, ch in enumerate(word):
        if ch.isalpha():
            return word[:i] + ch.upper() + word[i + 1:]
    return word


def restore_case(word, sentence_start, case_map, typed_prefix=""):
    """Stage 4: display form of a lowercase word. Sentence start -> capitalised; otherwise the most frequent
    mid-sentence casing from the case map (proper nouns, "I", "O"). A capital first letter typed by the user is kept."""
    out = capitalize_first_letter(word) if sentence_start else case_map.get(word, word)
    if typed_prefix[:1].isupper() and not out[:1].isupper():
        out = capitalize_first_letter(out)
    return out


def insert_word(text, partial_typed, word):
    """Stage 5: replace the partial word at the end of `text` by `word` followed by a space."""
    base = text[:len(text) - len(partial_typed)] if partial_typed else text
    return base + word + " "


def insert_punctuation(text, ch):
    """No space before punctuation: if the text ends with the space that insertion added, drop it first."""
    if ch in D.PUNCT and text.endswith(" ") and not text.endswith("  ") and text.strip():
        text = text[:-1]
    return text + ch


# --------------------------------------------------------------------------------------
# the whole pipeline
# --------------------------------------------------------------------------------------
def suggest(model, vocab, text, case_map, k=3, trace=False, rerank=False):
    """Suggestions for the text typed so far.

    Returns {"mode": "next" | "complete", "partial": ..., "suggestions": [{"word", "token", "prob", "new_text"}]}
    and, with trace=True, also the output of every stage under "stages". rerank=True applies stage 6."""
    context, partial, sentence_start = parse_typed(text)
    typed_partial = split_partial(text[-MAX_CHARS:])[1]          # as typed (original case), for insertion
    ids = [vocab.bos_id] + vocab.encode(context)                 # current sentence only, padded with <s>
    p = model.next_word_probs(ids)
    kk = max(k, 5) if trace else k
    stage3 = prefix_filter(p, vocab, partial, kk)
    if rerank:                                                   # stage 6 needs ALL matching words, then keeps the best kk
        stage6 = rerank_keystrokes(prefix_filter(p, vocab, partial, None), len(partial))[:kk]
        stage3 = stage6
    out = []
    for tok, prob in stage3[:k]:
        shown = restore_case(tok, sentence_start, case_map, typed_partial)
        out.append({"word": shown, "token": tok, "prob": prob, "new_text": insert_word(text, typed_partial, shown)})
    res = {"mode": "complete" if partial else "next", "partial": partial, "sentence_start": sentence_start,
           "context": context, "suggestions": out}
    if trace:
        res["stages"] = {
            "1 raw top-k": raw_topk(p, vocab, kk),
            "2 filtered (words only)": filter_tokens(p, vocab, kk),
            "3 prefix filter + renormalise": stage3,
            "4 case restored": [(restore_case(t, sentence_start, case_map, typed_partial), pr) for t, pr in stage3],
            "5 inserted text": [insert_word(text, typed_partial, restore_case(t, sentence_start, case_map, typed_partial))
                                for t, _ in stage3],
        }
        if rerank:
            res["stages"]["(3 before reranking)"] = prefix_filter(p, vocab, partial, kk)
    return res
