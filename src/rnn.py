"""
rnn.py -- vanilla (tanh) RNN and LSTM language models in PyTorch, comparable with the n-grams.

Fairness with the n-gram main comparison
  * same vocabulary, same loader (src/data.py), same sentences; every sentence is run through the
    network on its own, starting from a zero state at <s>  (context = current sentence only)
  * targets = every token after <s> (words, punctuation, <num>, <unk>, </s>); the loss and the
    perplexity are over ALL of them, exactly like the evaluator
  * the softmax covers the WHOLE vocabulary including <s>, which is never a target - like the n-gram
    models, which also give <s> a (tiny) non-zero probability (see docs/kn_design.md)
  * whole sentences are batched (bucketed by length, right-padded, loss masked); nothing is truncated

Architecture: embedding (size = hidden size) -> dropout -> RNN/LSTM stack (dropout between layers) ->
dropout -> output layer TIED to the input embedding (+ bias).  Adam, lr 1e-3, ReduceLROnPlateau on
validation perplexity, gradient clipping at norm 1.0, early stopping (patience 3), max 30 epochs.

`RNNPredictor` exposes the evaluator interface: next_word_probs(context) for one prefix and the
efficient `iter_sentence_probs(sentences)` (one pass per sentence -> distributions at every position).
"""

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
IGNORE = -100


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, hidden, layers, dropout, cell="lstm"):
        super().__init__()
        self.cell, self.hidden, self.layers, self.dropout_p = cell, hidden, layers, dropout
        self.emb = nn.Embedding(vocab_size, hidden)
        self.drop = nn.Dropout(dropout)
        inter = dropout if layers > 1 else 0.0
        if cell == "lstm":
            self.rnn = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=inter)
        elif cell == "rnn":
            self.rnn = nn.RNN(hidden, hidden, layers, batch_first=True, dropout=inter, nonlinearity="tanh")
        else:
            raise ValueError(cell)
        self.out_drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        if cell == "lstm":                                    # forget-gate bias 1 (standard trick)
            for name, p in self.rnn.named_parameters():
                if name.startswith("bias_ih"):
                    p.data[hidden:2 * hidden] = 1.0

    def forward(self, x):                                      # x: (B, L) ids -> logits (B, L, V)
        h, _ = self.rnn(self.drop(self.emb(x)))
        return F.linear(self.out_drop(h), self.emb.weight, self.bias)   # tied input/output embeddings

    def num_params(self):
        return sum(p.numel() for p in self.parameters())          # tied weights are counted once


# --------------------------------------------------------------------------------------
# batching (whole sentences, bucketed by length, padded + masked)
# --------------------------------------------------------------------------------------
def make_batches(lengths, max_tokens, max_sent, shuffle, rng):
    """Index batches with <= max_sent sentences and <= max_tokens padded tokens; similar lengths together."""
    idx = np.arange(len(lengths))
    if shuffle:
        rng.shuffle(idx)
    chunk = max_sent * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = idx[c:c + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        cur, maxlen = [], 0
        for j in part:
            new_max = max(maxlen, int(lengths[j]))
            if cur and (len(cur) >= max_sent or (len(cur) + 1) * new_max > max_tokens):
                batches.append(cur)
                cur, new_max = [], int(lengths[j])
            cur.append(int(j))
            maxlen = new_max
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(arrs, idxs, device):
    """arrs: list of np.int64 arrays [<s> ... </s>]. inputs = s[:-1], targets = s[1:] (pad: ignored)."""
    L = max(len(arrs[j]) for j in idxs) - 1
    x = np.zeros((len(idxs), L), dtype=np.int64)
    y = np.full((len(idxs), L), IGNORE, dtype=np.int64)
    for r, j in enumerate(idxs):
        a = arrs[j]
        x[r, :len(a) - 1] = a[:-1]
        y[r, :len(a) - 1] = a[1:]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def eval_nll(model, arrs, device, max_tokens=2048):
    """(total negative log-likelihood, number of target tokens) over all targets, eval mode."""
    model.eval()
    lengths = np.array([len(a) for a in arrs])
    tot, n = 0.0, 0
    for b in make_batches(lengths, max_tokens, 256, False, None):
        x, y = collate(arrs, b, device)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum").item()
        n += int((y != IGNORE).sum())
    return tot, n


def eval_ppl(model, arrs, device):
    tot, n = eval_nll(model, arrs, device)
    return math.exp(tot / n)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def run_name(cfg):
    """e.g. lstm_h512_l1_d0.5 (seed 42, Shakespeare); other seeds get _s<seed>, other corpora a prefix."""
    name = f"{cfg['cell']}_h{cfg['hidden']}_l{cfg['layers']}_d{cfg['dropout']}"
    if cfg.get("corpus", "shakespeare") != "shakespeare":
        name = f"{cfg['corpus']}_{name}"
    if cfg.get("seed", 42) != 42:
        name += f"_s{cfg['seed']}"
    return name


def default_cfg(**kw):
    cfg = dict(cell="lstm", hidden=256, layers=1, dropout=0.3, lr=1e-3, clip=1.0, max_epochs=30,
               patience=3, lr_patience=1, lr_factor=0.5, seed=42, max_tokens=4096, max_sent=128)
    cfg.update(kw)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(cfg, train_sents, val_sents, vocab_size, device, log=print, max_epochs=None):
    """Train one model; returns (best_model, history). Early stopping on validation perplexity."""
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = RNNLM(vocab_size, cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["cell"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["lr_factor"],
                                                       patience=cfg["lr_patience"])
    tr = [np.asarray(s, dtype=np.int64) for s in train_sents]
    va = [np.asarray(s, dtype=np.int64) for s in val_sents]
    tr_len = np.array([len(a) for a in tr])
    sub = [tr[i] for i in np.random.default_rng(0).choice(len(tr), min(4000, len(tr)), replace=False)]   # fixed train subset
    hist, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, (max_epochs or cfg["max_epochs"]) + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b in make_batches(tr_len, cfg["max_tokens"], cfg["max_sent"], True, rng):
            x, y = collate(tr, b, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE, reduction="sum")
            k = int((y != IGNORE).sum())
            opt.zero_grad(set_to_none=True)
            (loss / k).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += loss.item()
            n += k
        if device != "cpu":
            torch.cuda.empty_cache()
        val_ppl = eval_ppl(model, va, device)
        row = {"epoch": epoch, "train_ppl_dropout": math.exp(tot / n), "train_ppl_eval": eval_ppl(model, sub, device),
               "val_ppl": val_ppl, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        hist.append(row)
        log(f"  {run_name(cfg)} epoch {epoch:2d}  train(drop) {row['train_ppl_dropout']:7.1f}  "
            f"train(eval) {row['train_ppl_eval']:7.1f}  val {val_ppl:7.1f}  lr {row['lr']:.1e}  {row['seconds']:.0f}s")
        sched.step(val_ppl)
        if val_ppl < best:
            best, bad = val_ppl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("  early stopping")
                break
    model.load_state_dict(best_state)
    return model, hist


def checkpoint_path(cfg):
    return MODEL_DIR / f"{run_name(cfg)}.pt"


def save_run(model, cfg, hist):
    MODEL_DIR.mkdir(exist_ok=True)
    best_epoch = int(np.argmin([h["val_ppl"] for h in hist])) + 1
    torch.save({"cfg": cfg, "state_dict": model.state_dict(), "history": hist, "best_epoch": best_epoch,
                "best_val_ppl": min(h["val_ppl"] for h in hist), "num_params": model.num_params()},
               checkpoint_path(cfg))


def load_run(cfg_or_name, vocab_size, device="cpu"):
    path = checkpoint_path(cfg_or_name) if isinstance(cfg_or_name, dict) else MODEL_DIR / f"{cfg_or_name}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    model = RNNLM(vocab_size, c["hidden"], c["layers"], c["dropout"], c["cell"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def train_or_load(cfg, train_sents, val_sents, vocab_size, device, log=print, force=False):
    """Load models/<run>.pt if it exists, else train and save it. Returns (model, checkpoint dict)."""
    if checkpoint_path(cfg).exists() and not force:
        model, ck = load_run(cfg, vocab_size, device)
        log(f"loaded {run_name(cfg)} (best epoch {ck['best_epoch']}, val ppl {ck['best_val_ppl']:.2f})")
        return model, ck
    model, hist = train_model(cfg, train_sents, val_sents, vocab_size, device, log)
    save_run(model, cfg, hist)
    return load_run(cfg, vocab_size, device)


# --------------------------------------------------------------------------------------
# evaluator interface
# --------------------------------------------------------------------------------------
class RNNPredictor:
    """Wraps a trained RNNLM for src/evaluate.py (and sampling)."""

    n = None

    def __init__(self, model, device="cpu", name=None, cfg=None):
        # evaluation in full fp32: cuDNN's default TF32 mode gives batched and single-sequence runs
        # slightly different (~1e-3 relative) probabilities
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = model.to(device).eval()
        self.device = device
        self.name = name or "rnn"
        self.cfg = cfg or {}
        self._cpu = None

    def hyperparams(self):
        return {k: self.cfg[k] for k in ("cell", "hidden", "layers", "dropout") if k in self.cfg}

    # ---- one prefix (stateless: the whole prefix is re-read from zero state) -------------
    @torch.no_grad()
    def _probs_one(self, model, context, device):
        x = torch.tensor([list(context)], dtype=torch.long, device=device)
        p = torch.softmax(model(x)[0, -1].double(), dim=-1).cpu().numpy()
        return p / p.sum()

    def next_word_probs(self, context):
        return self._probs_one(self.model, context, self.device)

    # ---- every position of a sentence in one pass ------------------------------------------
    @torch.no_grad()
    def iter_sentence_probs(self, sentences, chunk=128, max_tokens=4096):
        """Yield, for each sentence [<s> ... </s>] (in order), an array (len-1, V): row i = distribution of
        token i+1 given tokens 0..i. Sentences are processed in length-sorted batches inside chunks."""
        # keep the (batch tokens x vocabulary) float64 tensors bounded: ~4e7 elements, whatever the vocabulary size
        max_tokens = min(max_tokens, max(256, int(4e7 // self.model.emb.num_embeddings)))
        for c in range(0, len(sentences), chunk):
            part = [np.asarray(s, dtype=np.int64) for s in sentences[c:c + chunk]]
            lengths = np.array([len(a) for a in part])
            out = {}
            for b in make_batches(lengths, max_tokens, 256, False, None):
                x, _ = collate(part, b, self.device)
                P = torch.softmax(self.model(x).double(), dim=-1).cpu().numpy()
                for r, j in enumerate(b):
                    p = P[r, :len(part[j]) - 1]
                    out[j] = p / p.sum(axis=1, keepdims=True)
            if self.device != "cpu":
                torch.cuda.empty_cache()
            for j in range(len(part)):
                yield out.pop(j)

    # ---- latency of ONE stateless call, on CPU (comparable with the n-gram models) --------------
    def latency_ms(self, contexts, warmup=5):
        if self._cpu is None:
            self._cpu = copy.deepcopy(self.model).cpu().eval()
        for c in contexts[:warmup]:
            self._probs_one(self._cpu, c, "cpu")
        t0 = time.perf_counter()
        for c in contexts:
            self._probs_one(self._cpu, c, "cpu")
        return 1000.0 * (time.perf_counter() - t0) / max(len(contexts), 1)

    def num_params(self):
        return self.model.num_params()


class RNNStateful:
    """What a real keyboard would do: keep the hidden state and feed ONLY the newest token.

    session = RNNStateful(model); p = session.step(<s> id); p = session.step(w1); ...  (reset() starts a new sentence).
    Runs on CPU in eval mode; step() returns the (normalised, float64) next-word distribution, exactly like the
    stateless RNNPredictor.next_word_probs on the same prefix."""

    def __init__(self, model, device="cpu"):
        self.model = copy.deepcopy(model).to(device).eval()
        self.device = device
        self.state = None

    def reset(self):
        self.state = None

    @torch.no_grad()
    def step(self, token):
        m = self.model
        x = torch.tensor([[int(token)]], dtype=torch.long, device=self.device)
        out, self.state = m.rnn(m.emb(x), self.state)
        p = torch.softmax(F.linear(out[0, -1], m.emb.weight, m.bias).double(), dim=-1).cpu().numpy()
        return p / p.sum()


class Interpolated:
    """P = lam * P_a + (1 - lam) * P_b  (a = neural model with iter_sentence_probs, b = n-gram)."""

    n = None

    def __init__(self, a, b, lam, name="ensemble"):
        self.a, self.b, self.lam, self.name = a, b, float(lam), name

    def hyperparams(self):
        return {"lambda": self.lam, "a": self.a.name, "b": self.b.name}

    def next_word_probs(self, context):
        return self.lam * self.a.next_word_probs(context) + (1 - self.lam) * self.b.next_word_probs(context)

    def iter_sentence_probs(self, sentences, **kw):
        for s, Pa in zip(sentences, self.a.iter_sentence_probs(sentences, **kw)):
            Pb = np.stack([self.b.next_word_probs(s[:i]) for i in range(1, len(s))])
            yield self.lam * Pa + (1 - self.lam) * Pb

    def latency_ms(self, contexts):
        t0 = time.perf_counter()
        for c in contexts:
            self.b.next_word_probs(c)
        b_ms = 1000.0 * (time.perf_counter() - t0) / max(len(contexts), 1)
        return self.a.latency_ms(contexts) + b_ms
