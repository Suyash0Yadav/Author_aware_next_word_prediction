"""stats.py -- paired bootstrap over sentences and McNemar's test (numpy only)."""
import math

import numpy as np


def sentence_table(details, n_sent, groups=None):
    """Per-sentence sufficient statistics from evaluate(..., details=True).

    Returns a dict of arrays of length n_sent: nll, ntok, n_word, top1, top3, kw, kwo, arch_n, arch_top1 and, for
    each (name -> boolean mask over word targets) in `groups`, g_<name>_n and g_<name>_top1."""
    d = details

    def bc(idx, w=None):
        return np.bincount(idx, weights=w, minlength=n_sent).astype(float)

    t = {"nll": bc(d["tok_sent"], -d["tok_logp"]), "ntok": bc(d["tok_sent"]),
         "n_word": bc(d["sent"]), "top1": bc(d["sent"], d["rank"] <= 1), "top3": bc(d["sent"], d["rank"] <= 3),
         "kw": bc(d["sent"], d["keys_with"]), "kwo": bc(d["sent"], d["keys_without"]),
         "arch_n": bc(d["sent"], d["arch"]), "arch_top1": bc(d["sent"], d["arch"] & (d["rank"] <= 1))}
    for name, mask in (groups or {}).items():
        t[f"g_{name}_n"] = bc(d["sent"], mask)
        t[f"g_{name}_top1"] = bc(d["sent"], mask & (d["rank"] <= 1))
    return t


METRIC_DEFS = {                                  # metric -> function of summed per-sentence statistics
    "ppl": lambda s: np.exp(s["nll"] / s["ntok"]),
    "top1": lambda s: s["top1"] / s["n_word"],
    "top3": lambda s: s["top3"] / s["n_word"],
    "ksr": lambda s: 1.0 - s["kw"] / s["kwo"],
    "archaic_top1": lambda s: s["arch_top1"] / s["arch_n"],
}


def metric_fn(name):
    if name in METRIC_DEFS:
        return METRIC_DEFS[name]
    if name.startswith("group:"):                # group:<name> -> top-1 on that target group
        g = name.split(":", 1)[1]
        return lambda s: s[f"g_{g}_top1"] / s[f"g_{g}_n"]
    raise KeyError(name)


def paired_bootstrap(tab_a, tab_b, metrics, n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap over SENTENCES: the same resampled sentences are used for models A and B.

    Returns {metric: dict(a, b, diff, ci_low, ci_high, rel_diff_pct, significant)} for diff = A - B; the
    interval is the percentile interval of the bootstrap distribution of the difference."""
    n = len(tab_a["nll"])
    rng = np.random.default_rng(seed)
    keys = list(tab_a)
    A = np.stack([tab_a[k] for k in keys], axis=1)               # (n_sent, n_stats)
    B = np.stack([tab_b[k] for k in keys], axis=1)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)    # resample weights per bootstrap
    sa_all, sb_all = counts @ A, counts @ B
    out = {}
    for m in metrics:
        f = metric_fn(m)
        a_pt = float(f({k: np.array([tab_a[k].sum()]) for k in keys})[0])
        b_pt = float(f({k: np.array([tab_b[k].sum()]) for k in keys})[0])
        da = f({k: sa_all[:, i] for i, k in enumerate(keys)})
        db = f({k: sb_all[:, i] for i, k in enumerate(keys)})
        diffs = da - db
        lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = {"a": a_pt, "b": b_pt, "diff": a_pt - b_pt, "ci_low": float(lo), "ci_high": float(hi),
                  "rel_diff_pct": 100 * (a_pt - b_pt) / b_pt if b_pt else float("nan"),
                  "significant": bool(lo > 0 or hi < 0)}
    return out


def mcnemar(correct_a, correct_b):
    """McNemar's test on paired binary outcomes: counts, exact two-sided binomial p, and the
    continuity-corrected chi-square statistic with its p-value (1 degree of freedom)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    both, only_a, only_b, neither = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    n = only_a + only_b
    if n == 0:
        return {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither, "p_exact": 1.0, "chi2": 0.0, "p_chi2": 1.0}
    k = min(only_a, only_b)
    p_exact = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    chi2 = (abs(only_a - only_b) - 1) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))
    return {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "p_exact": p_exact, "chi2": chi2, "p_chi2": p_chi2}
