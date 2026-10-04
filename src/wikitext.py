"""
wikitext.py -- finish the preprocessing of WikiText-2 (raw) with the same rules as the Shakespeare data.

Same rules as src/preprocess.py: NFC + straight apostrophes, sentences split at . ! ? (a paragraph end
always ends a sentence), tokens = words with inner apostrophes / <num> / the six punctuation marks,
hyphenated words split, lowercase. WikiText-specific clean-up first (undo its own pre-tokenisation):
  "game 's" -> "game's", "wasn 't" -> "wasn't", "1 @,@ 000" -> "1000", " @-@ " -> hyphen (then split).
Section headings ("= Title =") are dropped - they are structure, like the ACT/SCENE headers of the plays.

Outputs
  data/interim/wikitext/{train_full,val,test}_tokens.txt   unmapped tokenised sentences (the "stage 8" text)
  data/processed/wikitext/{train,val,test}.txt            <unk>-mapped, one sentence per line (same layout as the
                                                          Shakespeare files, loaded with src.data corpus="wikitext")
  data/processed/wikitext/vocab.json                      vocabulary of the (subsampled) TRAIN split, min_freq = 2
The WikiText train split is randomly subsampled (whole sentences, seed 42) to the token count of the
Shakespeare train split so that comparisons are about domain, not data size.

Run:  python -m src.wikitext
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data as D, preprocess as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "data" / "baseline"
INTERIM = ROOT / "data" / "interim" / "wikitext"
OUT = D.CORPUS_DIRS["wikitext"]
HEADING_RE = re.compile(r"^\s*(=\s*)+.*?(\s*=)+\s*$")
NUMJOIN_RE = re.compile(r"(\d) @[,.]@ (?=\d)")
CLITIC_RE = re.compile(r" '(s|t|re|ve|ll|d|m)\b", re.I)


def clean_line(line):
    """WikiText's own tokenisation artefacts removed (text still has spaces around punctuation)."""
    line = unicodedata.normalize("NFC", line).replace("’", "'").replace("‘", "'")
    line = NUMJOIN_RE.sub(r"\1", line)
    line = line.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    return CLITIC_RE.sub(r"'\1", line)


def sentences_of_line(line):
    """List of token lists (lowercase, cased-stage rules of the Shakespeare pipeline) for one paragraph."""
    out = []
    for sent in P.split_sentences(re.sub(r"\s+", " ", clean_line(line)).strip()):
        toks = [t.lower() for t in P.tokenize(sent)]
        if any(P.is_word(t) or t == "<num>" for t in toks):
            out.append(toks)
    return out


def read_split(name):
    sents = []
    for line in (BASELINE / f"wikitext2_{name}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or HEADING_RE.match(line):
            continue
        sents.extend(sentences_of_line(line))
    return sents


def n_tokens(sents):
    return sum(len(s) for s in sents)


def build_case_map():
    """Most frequent casing of each word when it is NOT the first token of a sentence (WikiText-2 train, raw
    text; same idea as data/processed/case_map.json for Shakespeare). Saved to data/processed/wikitext/case_map.json."""
    counts = {}
    for line in (BASELINE / "wikitext2_train.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or HEADING_RE.match(line):
            continue
        for sent in P.split_sentences(re.sub(r"\s+", " ", clean_line(line)).strip()):
            toks = P.tokenize(sent)                                  # cased
            for t in toks[1:]:
                if P.is_word(t):
                    counts.setdefault(t.lower(), Counter())[t] += 1
    cmap = {w: c.most_common(1)[0][0] for w, c in sorted(counts.items())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case_map.json").write_text(json.dumps(cmap, ensure_ascii=False, indent=0), encoding="utf-8")
    return cmap


def main(seed=42):
    OUT.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    train_full, val, test = read_split("train"), read_split("validation"), read_split("test")
    target = sum(len(l.split()) for l in (D.PROCESSED / "train.txt").read_text(encoding="utf-8").splitlines())

    # subsample the train split to the Shakespeare train token count (random whole sentences)
    order = list(range(len(train_full)))
    random.Random(seed).shuffle(order)
    chosen, total = [], 0
    for i in order:
        L = len(train_full[i])
        if total + L <= target:
            chosen.append(i)
            total += L
        if total == target:
            break
    chosen.sort()                                           # keep the original relative order
    train = [train_full[i] for i in chosen]

    # vocabulary: subsampled train only, min_freq = 2 (same recipe as the Shakespeare vocabulary)
    wc = Counter(t for s in train for t in s if P.is_word(t))
    kept = sorted((w for w, c in wc.items() if c >= P.MIN_FREQ), key=lambda w: (-wc[w], w))
    itos = P.SPECIALS + P.PUNCT + kept
    vset = set(itos)
    pc = Counter(t for s in train for t in s if t in P.PUNCT)
    counts = {"<s>": len(train), "</s>": len(train), "<unk>": sum(c for w, c in wc.items() if w not in vset),
              "<num>": sum(1 for s in train for t in s if t == "<num>")}
    counts.update({p: pc.get(p, 0) for p in P.PUNCT})
    counts.update({w: wc[w] for w in kept})
    vocab = {"min_freq": P.MIN_FREQ, "built_on": "WikiText-2 train, subsampled to the Shakespeare train token count",
             "size": len(itos), "specials": P.SPECIALS, "punctuation": P.PUNCT, "itos": itos, "counts": counts}
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")

    def write(path, sents):
        path.write_text("\n".join(" ".join(s) for s in sents) + "\n", encoding="utf-8", newline="\n")

    for nm, sents in (("train_full", train_full), ("val", val), ("test", test), ("train", train)):
        if nm != "train":
            write(INTERIM / f"{nm}_tokens.txt", sents)
    write(INTERIM / "train_tokens.txt", train)
    for nm, sents in (("train", train), ("val", val), ("test", test)):
        write(OUT / f"{nm}.txt", [[t if t in vset else "<unk>" for t in s] for s in sents])

    info = {"shakespeare_train_tokens": target, "wikitext_train_full_tokens": n_tokens(train_full),
            "wikitext_train_tokens": total, "train_sentences": len(train), "val_tokens": n_tokens(val),
            "val_sentences": len(val), "test_tokens": n_tokens(test), "test_sentences": len(test),
            "vocab_size": len(itos), "unk_pct_train": round(100 * counts["<unk>"] / max(sum(wc.values()), 1), 2)}
    (OUT / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    info["case_map_words"] = len(build_case_map())
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
