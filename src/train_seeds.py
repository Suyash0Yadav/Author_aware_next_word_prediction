"""Retrain the selected LSTM and vanilla RNN configurations with seeds 1 and 2 (seed 42 already exists).

    python -m src.train_seeds
No tuning: the configurations are the ones selected on validation for seed 42.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data as D, rnn as R  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = D.load_vocab()
    train, val = D.load_split("train", vocab=vocab), D.load_split("val", vocab=vocab)
    t0 = time.time()
    for seed in (1, 2):
        for cell in ("lstm", "rnn"):
            cfg = R.default_cfg(cell=cell, hidden=512, layers=1, dropout=0.5, seed=seed)
            _, ck = R.train_or_load(cfg, train, val, len(vocab), device, log=lambda m: print(m, flush=True))
            print(f"== {R.run_name(cfg)}: best val ppl {ck['best_val_ppl']:.2f} (epoch {ck['best_epoch']}) "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
