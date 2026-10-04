"""
ngram.py -- n-gram language models written from scratch (numpy only, no nltk.lm).

All models work on token IDS from src/data.py (vocabulary built on train) and expose

    model.next_word_probs(context_ids) -> np.ndarray, shape (V,), non-negative, sums to 1

`context_ids` is any list of ids; only the last n-1 are used and a context shorter than n-1 is
padded on the left with <s>. The evaluator passes [<s>, w1, ..., wi] (current sentence only).

Models
    UnigramModel(vocab_size, bos_id)        relative frequencies - the floor baseline
    LaplaceNgram(n, vocab_size, bos_id)     add-one smoothing, no back-off (weak baseline)
    KneserNeyNgram(n, vocab_size, bos_id, d)  interpolated Kneser-Ney of any order n >= 2

Training data = a list of sentences as id lists WITH boundaries: [<s>, w1 ... wk, </s>]
(what `src.data.load_split(split, vocab=vocab)` returns).

--------------------------------------------------------------------------------------------
Interpolated Kneser-Ney, as implemented (single absolute discount d for all orders)

Every training event is (context of n-1 tokens, target); a sentence is left-padded with n-1
copies of <s> so each real token has a full context, and <s> is never a target. For a k-gram
g = (h, w) of order k define the *adjusted count*
    a_k(g) = raw count c(g)                        if k = n  or g starts with <s>
           = N1+(. g) = #distinct left neighbours  otherwise   (continuation count)
and, for a context h of k-1 tokens,
    tot_k(h)  = sum_w a_k(h, w)         N1+_k(h) = #{w : a_k(h, w) > 0}

    P_k(w | h) = max(a_k(h,w) - d, 0) / tot_k(h)  +  gamma_k(h) * P_{k-1}(w | h')
    gamma_k(h) = d * N1+_k(h) / tot_k(h)           (h' = h without its first token)
    P_k(w | h) = P_{k-1}(w | h')                   if h was never seen (full back-off)
    P_1(w)     = max(a_1(w) - d, 0) / sum a_1  +  gamma_1 / V      (interpolated with uniform,
                                                                    so every id has p > 0)
so each P_k sums to 1. The sparse higher-order terms are added on top of a precomputed
V-vector for the unigram level, so one call costs microseconds-to-a-millisecond.

Options only used by the tests: `unigram_floor=False` and `raw_start=False` reproduce the
definitions used by nltk.lm.KneserNeyInterpolated (undiscounted continuation unigram,
continuation counts also for <s>-initial grams), see tests/test_ngram.py.
"""

import numpy as np


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _bits(vocab_size):
    return max(1, int(vocab_size - 1).bit_length())


def build_events(sentences, n, bos_id):
    """Return windows W of shape (N, n): row = (n-1 context tokens, target), one per real token.

    Sentences are [<s>, ..., </s>]; they are re-padded to n-1 <s> on the left."""
    pad = [bos_id] * (n - 1)
    seq, ok = [], []
    for s in sentences:
        body = list(s[1:]) if s and s[0] == bos_id else list(s)
        seq.extend(pad + body)
        ok.extend([False] * (n - 1) + [True] * len(body))
    seq = np.asarray(seq, dtype=np.int64)
    ok = np.asarray(ok, dtype=bool)
    # window j covers seq[j : j+n]; its target is seq[j+n-1], valid iff that position is a real token
    windows = np.lib.stride_tricks.sliding_window_view(seq, n)
    return windows[ok[n - 1:]]


def _encode(cols, bits):
    """Pack token columns (first = most significant) into one int64 key."""
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        key = (key << bits) | c
    return key


class _Level:
    """Sparse table for one order: for every seen context the (sorted) next words and their values."""

    def __init__(self, ctx_key, word, value):
        order = np.lexsort((word, ctx_key))
        ctx_key, word, value = ctx_key[order], word[order], value[order]
        first = np.ones(len(ctx_key), dtype=bool)
        first[1:] = ctx_key[1:] != ctx_key[:-1]
        self.starts = np.flatnonzero(first)
        self.ends = np.append(self.starts[1:], len(ctx_key))
        self.keys = ctx_key[self.starts]
        self.words = word.astype(np.int32)
        self.vals = value.astype(np.float64)
        self.tot = np.add.reduceat(self.vals, self.starts) if len(self.vals) else np.zeros(0)

    def find(self, key):
        pos = int(np.searchsorted(self.keys, key))
        if pos < len(self.keys) and self.keys[pos] == key:
            return pos
        return -1

    def __len__(self):
        return len(self.words)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
class _Base:
    name = "base"
    n = 1

    def __init__(self, vocab_size, bos_id):
        self.V, self.bos_id = int(vocab_size), int(bos_id)
        self.bits = _bits(self.V)
        self.mask = (1 << self.bits) - 1

    def hyperparams(self):
        return {}

    def num_ngrams(self):
        raise NotImplementedError

    def _full_key(self, context):
        """Key of the last n-1 tokens (left-padded with <s>); lower-order contexts are masked out of it."""
        m = self.n - 1
        tail = context[-m:] if m else []
        key = 0
        for _ in range(m - len(tail)):
            key = (key << self.bits) | self.bos_id
        for t in tail:
            key = (key << self.bits) | int(t)
        return key

    def logprob(self, context, target):
        return float(np.log(self.next_word_probs(context)[target]))


class UnigramModel(_Base):
    name = "unigram"
    n = 1

    def fit(self, sentences):
        counts = np.zeros(self.V, dtype=np.float64)
        for s in sentences:
            body = s[1:] if s and s[0] == self.bos_id else s
            np.add.at(counts, np.asarray(body, dtype=np.int64), 1.0)
        self.p = counts / counts.sum()
        self._nnz = int((counts > 0).sum())
        return self

    def next_word_probs(self, context=()):
        return self.p.copy()

    def num_ngrams(self):
        return self._nnz


class LaplaceNgram(_Base):
    """Add-one smoothing of the order-n estimate: (c(h,w)+1) / (c(h)+V). Unseen context -> uniform."""

    def __init__(self, n, vocab_size, bos_id):
        super().__init__(vocab_size, bos_id)
        assert n >= 2 and self.bits * (n - 1) <= 62
        self.n = n
        self.name = f"laplace{n}"

    def fit(self, sentences):
        W = build_events(sentences, self.n, self.bos_id)
        ctx = _encode([W[:, i] for i in range(self.n - 1)], self.bits)
        w = W[:, -1]
        order = np.lexsort((w, ctx))
        ctx, w = ctx[order], w[order]
        new = np.ones(len(ctx), dtype=bool)
        new[1:] = (ctx[1:] != ctx[:-1]) | (w[1:] != w[:-1])
        idx = np.flatnonzero(new)
        counts = np.diff(np.append(idx, len(ctx))).astype(np.float64)
        self.level = _Level(ctx[idx], w[idx], counts)
        return self

    def next_word_probs(self, context):
        pos = self.level.find(self._full_key(context))
        if pos < 0:
            return np.full(self.V, 1.0 / self.V)
        lv = self.level
        s, e = lv.starts[pos], lv.ends[pos]
        denom = lv.tot[pos] + self.V
        p = np.full(self.V, 1.0 / denom)
        p[lv.words[s:e]] += lv.vals[s:e] / denom
        return p

    def num_ngrams(self):
        return len(self.level)


class KneserNeyNgram(_Base):
    """Interpolated Kneser-Ney of order n >= 2 (see the module docstring)."""

    def __init__(self, n, vocab_size, bos_id, d=0.75, unigram_floor=True, raw_start=True):
        super().__init__(vocab_size, bos_id)
        assert n >= 2 and self.bits * (n - 1) <= 62, "context key must fit in an int64"
        self.n = n
        self.name = f"kn{n}"
        self.unigram_floor, self.raw_start = unigram_floor, raw_start
        self._d = None
        self.d = d

    # ---- discount: only needed at query time, so it can be changed without re-counting -------
    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self._d = float(value)
        if hasattr(self, "a1"):
            self._build_unigram_level()

    def hyperparams(self):
        return {"d": self.d}

    def _build_unigram_level(self):
        a1, d = self.a1, self._d
        total = a1.sum()
        if self.unigram_floor:
            n1 = float((a1 > 0).sum())
            self.P1 = np.maximum(a1 - d, 0.0) / total + (d * n1 / total) / self.V
        else:                                                 # nltk-style: plain continuation probability
            self.P1 = a1 / total

    # ---- training --------------------------------------------------------------------------
    def fit(self, sentences):
        n, bits, bos = self.n, self.bits, self.bos_id
        W = build_events(sentences, n, bos)
        self.levels = {}

        # level 1: continuation count of w = number of distinct left neighbours
        x, w = W[:, n - 2], W[:, n - 1]
        order = np.lexsort((x, w))
        xs, ws = x[order], w[order]
        new = np.ones(len(xs), dtype=bool)
        new[1:] = (ws[1:] != ws[:-1]) | (xs[1:] != xs[:-1])
        self.a1 = np.bincount(ws[new], minlength=self.V).astype(np.float64)
        self._build_unigram_level()

        for k in range(2, n + 1):
            cols = [W[:, j] for j in range(n - k, n)]               # the k tokens of the k-gram
            ctx_key = _encode(cols[:-1], bits)
            w = cols[-1]
            if k == n:
                order = np.lexsort((w, ctx_key))
                ck, ww = ctx_key[order], w[order]
                new = np.ones(len(ck), dtype=bool)
                new[1:] = (ck[1:] != ck[:-1]) | (ww[1:] != ww[:-1])
                idx = np.flatnonzero(new)
                a = np.diff(np.append(idx, len(ck))).astype(np.float64)
                self.levels[k] = _Level(ck[idx], ww[idx], a)
                continue
            gkey = (ctx_key << bits) | w                          # k tokens -> <= 56 bits
            left = W[:, n - k - 1]                                # token before the k-gram
            # raw counts of each distinct k-gram
            ug, raw = np.unique(gkey, return_counts=True)
            # continuation counts: number of distinct (left, k-gram) pairs per k-gram
            order = np.lexsort((left, gkey))
            gs, ls = gkey[order], left[order]
            new = np.ones(len(gs), dtype=bool)
            new[1:] = (gs[1:] != gs[:-1]) | (ls[1:] != ls[:-1])
            ug2, cont = np.unique(gs[new], return_counts=True)
            assert np.array_equal(ug, ug2)
            a = cont.astype(np.float64)
            if self.raw_start:                                    # grams starting with <s> cannot be extended
                starts_bos = ((ug >> (bits * (k - 1))) & self.mask) == bos
                a = np.where(starts_bos, raw.astype(np.float64), a)
            self.levels[k] = _Level(ug >> bits, ug & self.mask, a)
        return self

    # ---- prediction ------------------------------------------------------------------------
    def next_word_probs(self, context):
        full = self._full_key(context)
        d = self._d
        p = self.P1.copy()
        for k in range(2, self.n + 1):
            lv = self.levels[k]
            pos = lv.find(full & ((1 << (self.bits * (k - 1))) - 1))
            if pos < 0:                                           # unseen context: full back-off
                continue
            s, e = lv.starts[pos], lv.ends[pos]
            tot = lv.tot[pos]
            p *= d * (e - s) / tot
            p[lv.words[s:e]] += np.maximum(lv.vals[s:e] - d, 0.0) / tot
        return p

    def num_ngrams(self):
        """Number of stored n-gram types over all orders (unigram level counts words with a_1 > 0)."""
        return int((self.a1 > 0).sum()) + sum(len(lv) for lv in self.levels.values())
