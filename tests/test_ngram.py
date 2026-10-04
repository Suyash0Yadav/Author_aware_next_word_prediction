"""Correctness tests for src/ngram.py.

Run:  python -m pytest tests/test_ngram.py -q
"""
import math
import random

import numpy as np
import pytest

from src import data as D
from src import ngram as N


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def real():
    vocab = D.load_vocab()
    return vocab, D.load_split("train", vocab=vocab)


@pytest.fixture(scope="module")
def toy():
    """V = 5 ids: 0 <s>, 1 </s>, 2 a, 3 b, 4 c ; corpus: 'a b' and 'a c'."""
    return [[0, 2, 3, 1], [0, 2, 4, 1]]


# ----------------------------------------------------------------------------- 1. normalisation
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_kn_probabilities_sum_to_one_and_are_positive(real, n):
    vocab, train = real
    model = N.KneserNeyNgram(n, len(vocab), vocab.bos_id, d=0.75).fit(train)
    rng = random.Random(0)
    contexts = []
    for _ in range(50):                                         # real prefixes of training sentences
        s = rng.choice(train)
        contexts.append(s[:rng.randint(1, len(s) - 1)])
    for _ in range(50):                                         # random (mostly unseen) contexts
        contexts.append([rng.randrange(len(vocab)) for _ in range(rng.randint(0, 6))])
    for c in contexts:
        p = model.next_word_probs(c)
        assert p.shape == (len(vocab),)
        assert abs(p.sum() - 1.0) < 1e-6
        assert p.min() > 0.0


def test_laplace_and_unigram_are_proper_distributions(real):
    vocab, train = real
    uni = N.UnigramModel(len(vocab), vocab.bos_id).fit(train)
    assert abs(uni.next_word_probs([]).sum() - 1) < 1e-9
    for n in (2, 3):
        lap = N.LaplaceNgram(n, len(vocab), vocab.bos_id).fit(train)
        for c in (train[5][:3], [7, 8, 9], []):
            p = lap.next_word_probs(c)
            assert abs(p.sum() - 1) < 1e-9 and p.min() > 0


# ----------------------------------------------------------------------------- 2. hand-computed values
def test_kn_bigram_probability_matches_hand_computation(toy):
    """n=2, d=0.75, V=5, corpus  <s> a b </s>  /  <s> a c </s>.

    unigram level (continuation counts = #distinct left neighbours):
        a:1 (<s>)  b:1 (a)  c:1 (a)  </s>:2 (b, c)   -> S = 5, N1+ = 4, gamma_1 = 0.75*4/5 = 0.6
        P1(w) = max(cc-0.75, 0)/5 + 0.6/V(=5)         P1(b) = 0.05 + 0.12 = 0.17,  P1(</s>) = 0.25 + 0.12 = 0.37
    bigram level, context 'a': raw counts (a,b)=1 (a,c)=1 -> tot 2, N1+ 2, gamma = 0.75*2/2 = 0.75
        P(b|a) = (1-0.75)/2 + 0.75 * P1(b) = 0.125 + 0.1275 = 0.2525
    context 'b': (b,</s>)=1 -> tot 1, gamma = 0.75
        P(</s>|b) = 0.25 + 0.75 * 0.37 = 0.5275
    context '</s>' was never seen -> full back-off: P(b|</s>) = P1(b) = 0.17
    """
    m = N.KneserNeyNgram(2, 5, 0, d=0.75).fit(toy)
    assert m.next_word_probs([0, 2])[3] == pytest.approx(0.2525, abs=1e-12)
    assert m.next_word_probs([0, 2, 3])[1] == pytest.approx(0.5275, abs=1e-12)
    assert m.next_word_probs([0, 2, 3, 1])[3] == pytest.approx(0.17, abs=1e-12)
    assert m.next_word_probs([0, 2]).sum() == pytest.approx(1.0, abs=1e-12)


def test_laplace_bigram_matches_hand_computation(toy):
    """P(b|a) = (c(a,b)+1) / (c(a)+V) = (1+1)/(2+5) = 2/7 ; unseen context -> uniform 1/5."""
    m = N.LaplaceNgram(2, 5, 0).fit(toy)
    assert m.next_word_probs([0, 2])[3] == pytest.approx(2 / 7, abs=1e-12)
    assert m.next_word_probs([0, 2, 3, 1])[3] == pytest.approx(1 / 5, abs=1e-12)


def test_unigram_matches_relative_frequencies(toy):
    p = N.UnigramModel(5, 0).fit(toy).next_word_probs([])
    # targets: a a b c </s> </s>  (6 events)
    assert p.tolist() == pytest.approx([0, 2 / 6, 2 / 6, 1 / 6, 1 / 6])      # ids: <s>, </s>, a, b, c


def test_changing_the_discount_keeps_a_proper_distribution(real):
    vocab, train = real
    m = N.KneserNeyNgram(3, len(vocab), vocab.bos_id, d=0.75).fit(train)
    c = train[100][:4]
    before = m.next_word_probs(c)
    m.d = 0.5
    after = m.next_word_probs(c)
    assert abs(after.sum() - 1) < 1e-9 and after.min() > 0
    assert not np.allclose(before, after)
    with pytest.raises(ValueError):
        m.d = 1.5


# ----------------------------------------------------------------------------- 3. comparison with nltk
def _nltk_model(train_events, n, d, vocab_tokens):
    from nltk.lm import KneserNeyInterpolated
    lm = KneserNeyInterpolated(n, discount=d)
    # feed nltk exactly our training events: every suffix (1..n tokens) of (context + target)
    grams = [tuple(map(str, row[n - k:])) for row in train_events for k in range(1, n + 1)]
    lm.fit([grams], vocabulary_text=[str(t) for t in vocab_tokens])
    return lm


def _small_sample(real, n_train=300, n_test=120):
    vocab, train = real
    val = D.load_split("val", vocab=vocab)
    return vocab, train[:n_train], val[:n_test]


@pytest.mark.parametrize("n", [2, 3])
def test_kn_matches_nltk_when_definitions_coincide(real, n):
    """With nltk's conventions (plain continuation unigram, continuation counts also for <s>-grams)
    our log-probabilities agree with nltk.lm.KneserNeyInterpolated to numerical precision."""
    vocab, train, val = _small_sample(real)
    d = 0.75
    mine = N.KneserNeyNgram(n, len(vocab), vocab.bos_id, d=d, unigram_floor=False, raw_start=False).fit(train)
    train_ev = N.build_events(train, n, vocab.bos_id)
    seen_targets = set(train_ev[:, -1].tolist())
    ref = _nltk_model(train_ev, n, d, np.unique(train_ev))
    test_ev = N.build_events(val, n, vocab.bos_id)
    test_ev = [r for r in test_ev if r[-1] in seen_targets]               # nltk would map unseen words to <UNK>
    assert len(test_ev) > 500
    lp_mine, lp_ref = [], []
    for row in test_ev:
        ctx = [int(t) for t in row[:-1]]
        lp_mine.append(math.log(mine.next_word_probs(ctx)[row[-1]]))
        lp_ref.append(math.log(ref.score(str(row[-1]), tuple(map(str, ctx)))))
    assert np.max(np.abs(np.array(lp_mine) - np.array(lp_ref))) < 1e-9
    ppl_mine, ppl_ref = math.exp(-np.mean(lp_mine)), math.exp(-np.mean(lp_ref))
    assert ppl_mine == pytest.approx(ppl_ref, rel=1e-9)


def test_default_kn_is_close_to_nltk_perplexity_but_never_infinite(real):
    """Our default model (uniform floor at the unigram level, raw counts for <s>-initial grams) differs
    from nltk's by design; on a small sample the perplexities stay within 15 %."""
    vocab, train, val = _small_sample(real)
    n, d = 3, 0.75
    mine = N.KneserNeyNgram(n, len(vocab), vocab.bos_id, d=d).fit(train)
    train_ev = N.build_events(train, n, vocab.bos_id)
    seen = set(train_ev[:, -1].tolist())
    ref = _nltk_model(train_ev, n, d, np.unique(train_ev))
    test_ev = [r for r in N.build_events(val, n, vocab.bos_id) if r[-1] in seen]
    lp_mine = [math.log(mine.next_word_probs([int(t) for t in r[:-1]])[r[-1]]) for r in test_ev]
    lp_ref = [math.log(ref.score(str(r[-1]), tuple(str(t) for t in r[:-1]))) for r in test_ev]
    ppl_mine, ppl_ref = math.exp(-np.mean(lp_mine)), math.exp(-np.mean(lp_ref))
    assert abs(ppl_mine / ppl_ref - 1) < 0.15
