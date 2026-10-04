"""
data.py -- loaders for the preprocessed Shakespeare data (shared by the n-gram and RNN/LSTM code).

Files (written by src/preprocess.py):
    data/processed/{train,val,test}.txt   one sentence per line, space-separated, already <unk>-mapped,
                                          WITHOUT <s> / </s>
    data/processed/vocab.json             vocabulary built on train (min_freq=2) with counts
    data/processed/case_map.json          most frequent mid-sentence casing of each word (for display)
    data/processed/works_index.csv        which lines of each split file belong to which work

Typical use:
    from src.data import load_vocab, load_split
    vocab = load_vocab()
    train = load_split("train")                  # list of ["<s>", "to", "be", ..., "</s>"]
    ids   = load_split("train", vocab=vocab)     # same, as lists of ints
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CORPUS_DIRS = {"shakespeare": PROCESSED, "wikitext": PROCESSED / "wikitext"}   # same file layout in both

BOS, EOS, UNK, NUM = "<s>", "</s>", "<unk>", "<num>"
SPECIALS = [BOS, EOS, UNK, NUM]
PUNCT = [".", ",", ";", ":", "!", "?"]
SPLITS = ("train", "val", "test")


class Vocab:
    """Token <-> id mapping. Ids: <s>=0, </s>=1, <unk>=2, <num>=3, then . , ; : ! ?, then words by frequency."""

    def __init__(self, itos, counts=None):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts = counts or {}
        self.bos_id, self.eos_id, self.unk_id = self.stoi[BOS], self.stoi[EOS], self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def __contains__(self, tok):
        return tok in self.stoi

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def load_vocab(path=None, corpus="shakespeare"):
    path = Path(path) if path else CORPUS_DIRS[corpus] / "vocab.json"
    v = json.loads(path.read_text(encoding="utf-8"))
    return Vocab(v["itos"], v["counts"])


def load_case_map(path=None, corpus="shakespeare"):
    path = Path(path) if path else CORPUS_DIRS[corpus] / "case_map.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_sentences(split, corpus="shakespeare"):
    """Raw sentences of a split as token lists, no boundary tokens."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = CORPUS_DIRS[corpus] / f"{split}.txt"
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split(split, add_boundaries=True, vocab=None, corpus="shakespeare"):
    """Sentences of a split.

    add_boundaries : wrap every sentence in <s> ... </s>
    vocab          : if given, return lists of token ids instead of strings
    corpus         : "shakespeare" (default) or "wikitext" (data/processed/wikitext/, built by src/wikitext.py)
    """
    sents = read_sentences(split, corpus)
    if add_boundaries:
        sents = [[BOS] + s + [EOS] for s in sents]
    if vocab is not None:
        sents = [vocab.encode(s) for s in sents]
    return sents


def iter_tokens(split, add_boundaries=True):
    """Flat token stream of a split (useful for counting n-grams)."""
    for sent in load_split(split, add_boundaries):
        yield from sent


def load_works_index():
    """DataFrame: split, slug, category, first_line, n_sentences (line offsets within <split>.txt)."""
    return pd.read_csv(PROCESSED / "works_index.csv", encoding="utf-8")


def load_work_sentences(split, slug, add_boundaries=True):
    """Sentences of one work inside a split (e.g. to build qualitative examples from a test play)."""
    idx = load_works_index()
    row = idx[(idx["split"] == split) & (idx["slug"] == slug)].iloc[0]
    sents = load_split(split, add_boundaries)
    return sents[row.first_line: row.first_line + row.n_sentences]


def restore_case(tokens, case_map=None):
    """Turn a lowercased token list back into display text (for demos, not for scoring).

    Words get their most frequent mid-sentence casing from case_map.json; the first word of a
    sentence is capitalised; punctuation is attached to the previous word.
    """
    case_map = case_map if case_map is not None else load_case_map()
    out, start = [], True
    for tok in tokens:
        if tok in PUNCT:
            if out:
                out[-1] += tok
            else:
                out.append(tok)
            start = tok in (".", "!", "?")
            continue
        word = case_map.get(tok, tok)
        if start and word[:1].isalpha():
            word = word[0].upper() + word[1:]
        out.append(word)
        start = False
    return " ".join(out)
