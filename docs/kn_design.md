# Kneser-Ney design notes (`src/ngram.py`)

Our interpolated Kneser-Ney (KN) follows Chen & Goodman / KenLM, with one absolute discount `d` shared by
all orders. Training events are `(n-1 padded context tokens, target)`; every real token (including `</s>`)
is a target, `<s>` never is. Every k-gram `g` gets an *adjusted count*

```
a_k(g) = c(g)                     if k = n  (highest order)  or  g starts with <s>
       = N1+(• g)                 otherwise (continuation count = # distinct left neighbours)

P_k(w|h) = max(a_k(hw) - d, 0) / tot_k(h) + gamma_k(h) * P_{k-1}(w|h')      gamma_k(h) = d * N1+_k(h) / tot_k(h)
P_k(w|h) = P_{k-1}(w|h')                                                    if h was never seen (full back-off)
P_1(w)   = max(a_1(w) - d, 0) / sum a_1  +  gamma_1 / V                      (discounted unigram + uniform floor)
```

`nltk.lm.KneserNeyInterpolated` computes the same recursion except for two points where we deliberately
differ. With the options `unigram_floor=False, raw_start=False` our code reproduces nltk's definitions, and
`tests/test_ngram.py` checks that the two agree to 1e-9 (log-probabilities and perplexity) when fed the same
events; with our defaults the perplexities agree within 15 % on the small test sample.

## Difference 1 - uniform floor under the unigram level

* **nltk:** `P_1(w) = N1+(•w) / sum N1+(•w')` - a pure continuation probability, no discount, no floor. Any
  vocabulary id that was never seen as a continuation (e.g. `<s>`) gets probability **0**, so a log-probability
  or a "full-vocabulary distribution" is undefined for it.
* **Ours:** `P_1(w) = max(a_1(w)-d, 0)/S + (d * N1+ / S) / V`, i.e. the lowest order is interpolated with the
  uniform distribution over the whole vocabulary of size `V`.

**Why it is reasonable**

1. The evaluator and the RNN comparison need a *proper distribution over the entire vocabulary with p > 0*
   (`next_word_probs` must sum to 1 and a softmax model also gives every id non-zero mass). The floor is the
   standard way to get this (KenLM interpolates its unigram level with a uniform distribution; Chen & Goodman
   discuss the same device).
2. It is nearly free. The floor costs `d*N1+/(S*V)` = 3.8e-6 per word, and the model puts on average 6.6e-7 on
   `<s>`, the one id that is never a target.
3. **Measured effect on the reported numbers: none.** Because every vocabulary id except `<s>` has been seen
   (`min_freq = 2`), `max(a-d,0)/S + dN1+/(S V)` is almost exactly `a/S`: the mass taken by the discount is
   handed back uniformly. KN-3 (d = 0.9) validation perplexity is 136.666 with and without the floor (they differ
   by < 0.001). So the floor is a correctness guarantee (positivity / normalisation over the full vocabulary),
   not a modelling choice that moves the results.

## Difference 2 - raw counts for n-grams that start with `<s>`

* **nltk:** every lower-order estimate uses continuation counts, including k-grams that start with `<s>`.
* **Ours:** k-grams starting with `<s>` keep their *raw* count; only the other lower-order k-grams use
  continuation counts.

**Why it is reasonable in principle.** The continuation count of `(<s>, w)` is the number of distinct tokens
that can precede it, and *nothing can precede a sentence start*. In our padded data the only "left neighbour"
is the artificial padding `<s>`, so the continuation count is 1 for every sentence-initial word however often
the sentence starts with it; the lower-order estimate would then say "all sentence starters are equally
likely" and throw away the evidence of how often each word opens a sentence. SRILM and KenLM therefore keep the
raw count for n-grams that begin with `<s>`.

**What the data say here (validation, KN-3, d = 0.9) - the evidence is mixed:**

| uniform floor | raw counts for `<s>`-initial grams | validation perplexity | perplexity of the first token of a sentence |
|---|---|---:|---:|
| yes (default) | yes (default) | 136.666 | 224.9 |
| no | yes | 136.666 | 224.9 |
| yes | no (nltk-style) | 136.087 | 209.9 |
| no | no (= nltk) | 136.087 | 209.9 |

In *our* implementation the raw-count rule is slightly **worse** (+0.4 % overall, +7 % on the first token).
The reason is the n-1 left padding: the top-order context `(<s>, <s>)` already carries exactly the same raw
counts as the lower-order context `(<s>)`, so using raw counts again at the lower level double-counts the same
evidence instead of smoothing it, whereas flat continuation counts give a useful type-based back-off
("which kinds of words open sentences"). KenLM avoids the duplication by *truncating* contexts at the sentence
start instead of padding them; the raw-count rule is right for that design.

**Consequence / options.** The principle is sound but with explicit padding the benefit does not materialise;
the cost is small (0.4 % perplexity). All reported n-gram results (validation, test, the ensemble) use the
default `raw_start=True`. Switching to the nltk convention (`raw_start=False`) is a one-flag change and would
improve validation perplexity by about 0.4 %; it would require re-running `03_ngram.ipynb` and everything built on
the KN trigram. We keep the default for consistency with the already-reported results and mention the 0.4 % as a
known, deliberate cost.
