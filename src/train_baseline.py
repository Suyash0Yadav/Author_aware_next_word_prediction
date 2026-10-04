"""Train the WikiText-2 LSTM (the best Shakespeare LSTM configuration, early stopping on WikiText validation).

    python -m src.train_baseline         (needs data/processed/wikitext/ from `python -m src.wikitext`)
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data as D, rnn as R  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = D.load_vocab(corpus="wikitext")
    train = D.load_split("train", vocab=vocab, corpus="wikitext")
    val = D.load_split("val", vocab=vocab, corpus="wikitext")
    print(f"WikiText vocab {len(vocab):,}; train {len(train):,} sentences, val {len(val):,}", flush=True)
    t0 = time.time()
    # no tuning; max_tokens=2048: the 25k-word vocabulary makes the logits large, a smaller batch token budget keeps the
    # GPU memory in check (an implementation detail - model, optimiser and schedule are those of the Shakespeare LSTM)
    cfg = R.default_cfg(cell="lstm", hidden=512, layers=1, dropout=0.5, corpus="wikitext", max_tokens=2048)
    _, ck = R.train_or_load(cfg, train, val, len(vocab), device, log=lambda m: print(m, flush=True))
    print(f"== {R.run_name(cfg)}: best val ppl {ck['best_val_ppl']:.2f} (epoch {ck['best_epoch']}) "
          f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
