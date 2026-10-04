"""Train the LSTM grid and the vanilla RNN (idempotent: finished runs are loaded, not retrained).

    python -m src.train_rnn            # LSTM: hidden {256,512} x layers {1,2} x dropout {0.3,0.5}, then the RNN
                                       # with the best LSTM's size and layer count (dropout 0.3 and 0.5)
Checkpoints: models/<cell>_h<H>_l<L>_d<p>.pt (weights + per-epoch history). Selection is on validation only.
"""
import itertools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data as D, rnn as R  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    vocab = D.load_vocab()
    train, val = D.load_split("train", vocab=vocab), D.load_split("val", vocab=vocab)
    t0 = time.time()
    best = None
    for hidden, layers, dropout in itertools.product((256, 512), (1, 2), (0.3, 0.5)):
        cfg = R.default_cfg(cell="lstm", hidden=hidden, layers=layers, dropout=dropout)
        _, ck = R.train_or_load(cfg, train, val, len(vocab), device, log=lambda m: print(m, flush=True))
        print(f"== {R.run_name(cfg)}: best val ppl {ck['best_val_ppl']:.2f} (epoch {ck['best_epoch']}) "
              f"[{(time.time() - t0) / 60:.1f} min elapsed]", flush=True)
        if best is None or ck["best_val_ppl"] < best[0]:
            best = (ck["best_val_ppl"], cfg)
    print("best LSTM:", R.run_name(best[1]), flush=True)
    for dropout in (0.3, 0.5):
        cfg = R.default_cfg(cell="rnn", hidden=best[1]["hidden"], layers=best[1]["layers"], dropout=dropout)
        _, ck = R.train_or_load(cfg, train, val, len(vocab), device, log=lambda m: print(m, flush=True))
        print(f"== {R.run_name(cfg)}: best val ppl {ck['best_val_ppl']:.2f} (epoch {ck['best_epoch']})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
