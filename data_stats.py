#!/usr/bin/env python3
"""
data_stats.py -- print statistics for the Shakespeare dataset built by
get_shakespeare.py. Read-only: it never modifies any data file.

Reports: totals (works, lines, words, chars), vocabulary size and lexical
diversity, per-category breakdown, most frequent words, line/word length
distributions, and the largest / smallest works.

Words here use a simple whitespace split (same as metadata.csv). The
"vocabulary" section additionally lowercases and strips punctuation, but
ONLY in memory for counting; no files are changed.

Usage:  python data_stats.py [--top N]
"""


## run the python file in dedicated terminal 


import argparse
import csv
import re
import statistics
from collections import Counter
from pathlib import Path

DATA = Path("data")
WORKS_DIR = DATA / "works"
METADATA_FILE = DATA / "metadata.csv"

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")  # alphabetic words, keeps don't / o'er


def load_metadata() -> list[dict]:
    with METADATA_FILE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("num_lines", "num_words", "num_chars"):
            r[k] = int(r[k])
    return rows


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="most frequent words to show")
    args = ap.parse_args()

    if not METADATA_FILE.exists():
        raise SystemExit("data/metadata.csv not found -- run get_shakespeare.py first.")
    rows = load_metadata()

    # ---- Overall totals ---------------------------------------------------
    section("OVERALL")
    total_words = sum(r["num_words"] for r in rows)
    print(f"Works:        {len(rows):,}")
    print(f"Lines:        {sum(r['num_lines'] for r in rows):,}")
    print(f"Words:        {total_words:,}  (whitespace split)")
    print(f"Characters:   {sum(r['num_chars'] for r in rows):,}")
    wc = [r["num_words"] for r in rows]
    print(f"Words/work:   mean {statistics.mean(wc):,.0f}, median {statistics.median(wc):,.0f}, "
          f"min {min(wc):,}, max {max(wc):,}")

    # ---- Per category -----------------------------------------------------
    section("BY CATEGORY")
    print(f"{'Category':<10}{'Works':>6}{'Words':>12}{'Share':>8}{'Avg/work':>11}")
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r["num_words"])
    for cat, ws in sorted(cats.items(), key=lambda kv: -sum(kv[1])):
        print(f"{cat:<10}{len(ws):>6}{sum(ws):>12,}{sum(ws) / total_words:>8.1%}"
              f"{sum(ws) / len(ws):>11,.0f}")

    # ---- Read all works for text-level stats ------------------------------
    tokens = Counter()
    line_lengths = []          # words per non-empty line
    for r in rows:
        text = (WORKS_DIR / f"{r['slug']}.txt").read_text(encoding="utf-8")
        for line in text.splitlines():
            n = len(line.split())
            if n:
                line_lengths.append(n)
        tokens.update(WORD_RE.findall(text.lower()))

    # ---- Vocabulary -------------------------------------------------------
    section("VOCABULARY (lowercased, punctuation stripped, in memory only)")
    n_tokens, vocab = sum(tokens.values()), len(tokens)
    hapax = sum(1 for c in tokens.values() if c == 1)
    print(f"Tokens:                {n_tokens:,}")
    print(f"Unique words (vocab):  {vocab:,}")
    print(f"Type/token ratio:      {vocab / n_tokens:.4f}")
    print(f"Words seen once:       {hapax:,} ({hapax / vocab:.1%} of vocab)")
    for k in (1_000, 5_000, 10_000):
        if k < vocab:
            cover = sum(c for _, c in tokens.most_common(k)) / n_tokens
            print(f"Top {k:>6,} words cover {cover:.1%} of all tokens")

    section(f"TOP {args.top} MOST FREQUENT WORDS")
    for i, (w, c) in enumerate(tokens.most_common(args.top), 1):
        print(f"{i:>3}. {w:<12}{c:>9,}  {c / n_tokens:6.2%}")

    # ---- Line lengths -----------------------------------------------------
    section("LINE LENGTH (words per non-empty line)")
    print(f"Non-empty lines: {len(line_lengths):,}")
    print(f"Mean {statistics.mean(line_lengths):.1f}, median {statistics.median(line_lengths):.0f}, "
          f"max {max(line_lengths)}")

    # ---- Largest / smallest ----------------------------------------------
    by_size = sorted(rows, key=lambda r: r["num_words"], reverse=True)
    section("5 LARGEST WORKS")
    for r in by_size[:5]:
        print(f"{r['num_words']:>9,}  {r['title']}  [{r['category']}]")
    section("5 SMALLEST WORKS")
    for r in by_size[-5:]:
        print(f"{r['num_words']:>9,}  {r['title']}  [{r['category']}]")


if __name__ == "__main__":
    main()
