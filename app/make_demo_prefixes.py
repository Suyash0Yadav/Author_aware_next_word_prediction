"""Pick 8-10 prefixes from the TEST plays on which the Shakespeare LSTM and the WikiText LSTM clearly differ.

    python app/make_demo_prefixes.py          -> writes app/demo_prefixes.txt

Criteria: the Shakespeare LSTM ranks the real next word first, the WikiText LSTM ranks it 10th or worse (or does not
know the word); readable prefixes (4-9 tokens, no <unk>/<num>); at most 3 per play; archaic words preferred.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data as D, evaluate as E, rnn as R  # noqa: E402


def main(n_out=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vs, vw = D.load_vocab(), D.load_vocab(corpus="wikitext")
    cm = D.load_case_map()
    ns, _ = R.load_run("lstm_h512_l1_d0.5", len(vs), device)
    nw, _ = R.load_run("wikitext_lstm_h512_l1_d0.5", len(vw), device)
    ps, pw = R.RNNPredictor(ns, device), R.RNNPredictor(nw, device)
    split = pd.read_csv(ROOT / "data" / "split.csv")
    meta = pd.read_csv(ROOT / "data" / "metadata.csv")
    tests = set(split.loc[split["split"] == "test", "slug"])
    sents, plays = [], []
    for slug in meta["slug"]:
        if slug in tests:
            lines = [l.split() for l in (ROOT / "data" / "interim" / "stage8_lowercased" / f"{slug}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
            sents += lines
            plays += [slug] * len(lines)
    ids_s = [[vs.bos_id] + vs.encode(t) + [vs.eos_id] for t in sents]
    ids_w = [[vw.bos_id] + vw.encode(t) + [vw.eos_id] for t in sents]
    cs, cw = E.word_ids(vs), E.word_ids(vw)
    arch = E.archaic_set()
    cands = []
    for si, (Ps, Pw) in enumerate(zip(ps.iter_sentence_probs(ids_s), pw.iter_sentence_probs(ids_w))):
        toks = sents[si]
        for i in range(4, min(len(toks), 10)):                    # 4-9 tokens of context
            t = toks[i]
            if t not in vs.stoi or any(x in ("<unk>", "<num>") or x not in vs.stoi for x in toks[:i + 1]) or len(t) < 3:
                continue
            tid = vs.stoi[t]
            if tid not in set(cs.tolist()):
                continue
            ranks = 1 + int((Ps[i - 1][cs] > Ps[i - 1][tid]).sum())
            if ranks != 1:
                continue
            wid = vw.stoi.get(t)
            if wid is None or wid not in set(cw.tolist()):
                rw = 10 ** 9
            else:
                rw = 1 + int((Pw[i - 1][cw] > Pw[i - 1][wid]).sum())
            if rw < 10:
                continue
            w_top = vw.itos[cw[np.argmax(Pw[i - 1][cw])]]
            s_top = vs.itos[cs[np.argmax(Ps[i - 1][cs])]]
            cands.append({"play": plays[si], "tokens": toks[:i], "target": t, "rank_w": rw, "w_top": w_top,
                          "score": (2 if t.strip("'") in arch else 0) + (1 if t not in vw.stoi else 0) + min(np.log10(rw), 5) / 5})
    print(f"{len(cands)} candidate prefixes")
    cands.sort(key=lambda c: -c["score"])
    out, per_play, seen = [], {}, set()
    for c in cands:
        key = " ".join(c["tokens"])
        if per_play.get(c["play"], 0) >= 3 or key in seen:
            continue
        out.append(c); per_play[c["play"]] = per_play.get(c["play"], 0) + 1; seen.add(key)
        if len(out) == n_out:
            break
    lines = ["# Prefixes from the TEST plays where the Shakespeare model predicts the next word and the WikiText model does not.",
             "# format:  prefix  |||  note"]
    for c in out:
        text = D.restore_case(c["tokens"], cm)
        rw = "not in its vocabulary" if c["rank_w"] >= 10 ** 9 else f"rank {c['rank_w']:,}"
        lines.append(f"{text}  |||  {c['play'].replace('_', ' ')}: Shakespeare model -> \"{c['target']}\", WikiText model -> \"{c['w_top']}\" ({rw})")
    (ROOT / "app" / "demo_prefixes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
