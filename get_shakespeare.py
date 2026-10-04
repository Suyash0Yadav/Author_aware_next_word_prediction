#!/usr/bin/env python3
"""
get_shakespeare.py -- build the Shakespeare dataset for the next-word
prediction project (n-gram vs RNN/LSTM).

Pipeline
    1. Download   "The Complete Works of William Shakespeare" (Gutenberg #100)
                  -> data/raw/shakespeare_raw.txt
    2. Strip      Project Gutenberg header/footer
                  -> data/interim/shakespeare_no_boilerplate.txt
    3. Split      into individual works
                  -> data/works/<slug>.txt
    4. Label      each work as Comedy / Tragedy / History / Romance / Poetry
    5. Metadata   -> data/metadata.csv, plus a printed summary and a check
                  that the corpus has at least 500,000 words.

Deliberately NOT done here: lowercasing, removing speaker names, stage
directions, act/scene headers or punctuation. That is the separate
preprocessing phase, which needs these files untouched for before/after
comparisons.

Dependencies: `requests` + Python standard library only.
"""

import csv
import re
import sys
import time
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PRIMARY_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
FALLBACK_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
USER_AGENT = (
    "UniversityNLPProject-NextWordPrediction/1.0 "
    "(educational; single-file download of Gutenberg eBook #100)"
)
TIMEOUT = 60          # seconds per request
MAX_RETRIES = 3       # attempts per URL
BACKOFF = 2           # seconds; doubles after each failed attempt
MIN_WORDS = 500_000   # project requirement

DATA = Path("data")
RAW_FILE = DATA / "raw" / "shakespeare_raw.txt"
INTERIM_FILE = DATA / "interim" / "shakespeare_no_boilerplate.txt"
WORKS_DIR = DATA / "works"
METADATA_FILE = DATA / "metadata.csv"

# --------------------------------------------------------------------------
# Hardcoded category mapping (keys use straight apostrophes, upper case).
# Titles that don't match are printed so you can add/fix them here.
# --------------------------------------------------------------------------
CATEGORIES = {
    # Poetry: sonnets and narrative/lyric poems
    "THE SONNETS": "Poetry",
    "A LOVER'S COMPLAINT": "Poetry",
    "THE PASSIONATE PILGRIM": "Poetry",
    "THE PHOENIX AND THE TURTLE": "Poetry",
    "THE RAPE OF LUCRECE": "Poetry",
    "VENUS AND ADONIS": "Poetry",
    # Comedy
    "ALL'S WELL THAT ENDS WELL": "Comedy",
    "AS YOU LIKE IT": "Comedy",
    "THE COMEDY OF ERRORS": "Comedy",
    "LOVE'S LABOUR'S LOST": "Comedy",
    "MEASURE FOR MEASURE": "Comedy",
    "THE MERCHANT OF VENICE": "Comedy",
    "THE MERRY WIVES OF WINDSOR": "Comedy",
    "A MIDSUMMER NIGHT'S DREAM": "Comedy",
    "MUCH ADO ABOUT NOTHING": "Comedy",
    "THE TAMING OF THE SHREW": "Comedy",
    "TWELFTH NIGHT; OR, WHAT YOU WILL": "Comedy",
    "THE TWO GENTLEMEN OF VERONA": "Comedy",
    # Tragedy (Troilus and Cressida is a "problem play"; filed here)
    "THE TRAGEDY OF ANTONY AND CLEOPATRA": "Tragedy",
    "THE TRAGEDY OF CORIOLANUS": "Tragedy",
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK": "Tragedy",
    "THE TRAGEDY OF JULIUS CAESAR": "Tragedy",
    "THE TRAGEDY OF KING LEAR": "Tragedy",
    "THE TRAGEDY OF MACBETH": "Tragedy",
    "THE TRAGEDY OF OTHELLO, MOOR OF VENICE": "Tragedy",
    "THE TRAGEDY OF ROMEO AND JULIET": "Tragedy",
    "THE LIFE OF TIMON OF ATHENS": "Tragedy",
    "THE TRAGEDY OF TITUS ANDRONICUS": "Tragedy",
    "THE HISTORY OF TROILUS AND CRESSIDA": "Tragedy",
    # History
    "THE LIFE AND DEATH OF KING JOHN": "History",
    "KING JOHN": "History",
    "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE": "Tragedy",
    "TROILUS AND CRESSIDA": "Tragedy",
    "KING RICHARD THE SECOND": "History",
    "KING RICHARD THE THIRD": "History",
    "THE FIRST PART OF KING HENRY THE FOURTH": "History",
    "THE SECOND PART OF KING HENRY THE FOURTH": "History",
    "THE LIFE OF KING HENRY THE FIFTH": "History",
    "THE FIRST PART OF HENRY THE SIXTH": "History",
    "THE SECOND PART OF KING HENRY THE SIXTH": "History",
    "THE THIRD PART OF KING HENRY THE SIXTH": "History",
    "KING HENRY THE EIGHTH": "History",
    # Romance (the late plays)
    "CYMBELINE": "Romance",
    "PERICLES, PRINCE OF TYRE": "Romance",
    "THE TEMPEST": "Romance",
    "THE WINTER'S TALE": "Romance",
    "THE TWO NOBLE KINSMEN": "Romance",
}

# Leading words dropped when building file slugs (hamlet_prince_of_denmark.txt)
SLUG_PREFIXES = (
    "THE TRAGEDY OF ", "THE COMEDY OF ", "THE HISTORY OF ",
    "THE LIFE OF ", "THE LIFE AND DEATH OF ", "THE ", "A ",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm(s: str) -> str:
    """Normalise a title/line for comparison: curly -> straight apostrophes,
    collapse whitespace, upper-case. Used for MATCHING only; saved text is
    never altered."""
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().upper()


def slugify(title: str) -> str:
    t = norm(title)
    for prefix in SLUG_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def download() -> None:
    if RAW_FILE.exists():
        print(f"[1/5] {RAW_FILE} already exists -- skipping download.")
        return

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}

    for url in (PRIMARY_URL, FALLBACK_URL):
        delay = BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[1/5] GET {url} (attempt {attempt}/{MAX_RETRIES})")
                resp = requests.get(url, headers=headers, timeout=TIMEOUT)
                resp.raise_for_status()
                # Save the bytes exactly as received (no decoding/re-encoding).
                RAW_FILE.write_bytes(resp.content)
                print(f"      saved {len(resp.content):,} bytes -> {RAW_FILE}")
                return
            except requests.RequestException as exc:
                print(f"      failed: {exc}")
                if attempt < MAX_RETRIES:
                    print(f"      retrying in {delay}s ...")
                    time.sleep(delay)
                    delay *= 2
        print(f"      giving up on {url}")

    sys.exit("ERROR: could not download the text from either URL.")


# --------------------------------------------------------------------------
# 2. Strip Project Gutenberg boilerplate
# --------------------------------------------------------------------------
# Tolerates "THE"/"THIS", extra spaces, and any trailing title text.
START_RE = re.compile(r"\*{3}\s*START OF (?:THE|THIS)\s+PROJECT GUTENBERG EBOOK[^\n]*\n?",
                      re.IGNORECASE)
END_RE = re.compile(r"\*{3}\s*END OF (?:THE|THIS)\s+PROJECT GUTENBERG EBOOK",
                    re.IGNORECASE)


def strip_boilerplate(raw: str) -> str:
    start = START_RE.search(raw)
    if not start:
        sys.exit("ERROR: could not find the Gutenberg START marker.")
    end = END_RE.search(raw, start.end())
    if not end:
        sys.exit("ERROR: could not find the Gutenberg END marker.")
    body = raw[start.end():end.start()]
    INTERIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    INTERIM_FILE.write_text(body, encoding="utf-8", newline="\n")
    print(f"[2/5] boilerplate removed -> {INTERIM_FILE}")
    return body


# --------------------------------------------------------------------------
# 3. Split into individual works
# --------------------------------------------------------------------------
def parse_contents(lines: list[str]) -> tuple[list[str], int]:
    """Return (titles, index of the first line after the Contents block).

    The block starts at a line reading 'Contents' and runs over the
    consecutive non-blank lines that follow (after skipping blank lines)."""
    for i, line in enumerate(lines):
        if line.strip().lower() == "contents":
            break
    else:
        sys.exit("ERROR: no 'Contents' heading found.")

    j = i + 1
    while j < len(lines) and not lines[j].strip():   # skip blanks after heading
        j += 1
    titles = []
    while j < len(lines) and lines[j].strip():       # until the next blank line
        titles.append(lines[j].strip())
        j += 1
    if not titles:
        sys.exit("ERROR: Contents list is empty.")
    return titles, j


def split_works(body: str) -> list[tuple[str, str]]:
    """Return [(title, text), ...] in the order the works appear in the book."""
    lines = body.splitlines()
    titles, after_contents = parse_contents(lines)
    print(f"[3/5] Contents lists {len(titles)} titles.")

    # Index of every line (after the Contents block) by normalised content,
    # so each title is matched as a WHOLE line, not a substring.
    wanted = {norm(t): t for t in titles}
    starts = {}                                       # normalised title -> line no.
    for n in range(after_contents, len(lines)):
        key = norm(lines[n])
        if key in wanted and key not in starts:       # first occurrence wins
            starts[key] = n

    missing = [t for t in titles if norm(t) not in starts]
    if missing:
        print("      WARNING: title not found in body (skipped):")
        for t in missing:
            print(f"        - {t}")

    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    works = []
    for idx, (key, begin) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        text = "\n".join(lines[begin:end]).strip("\n") + "\n"
        works.append((wanted[key], text))
    return works


# --------------------------------------------------------------------------
# 4 + 5. Label, save, metadata, validation
# --------------------------------------------------------------------------
def main() -> None:
    for d in (RAW_FILE.parent, INTERIM_FILE.parent, WORKS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    download()

    # utf-8-sig silently drops a BOM if present; otherwise identical to utf-8.
    raw = RAW_FILE.read_text(encoding="utf-8-sig")
    body = strip_boilerplate(raw)
    works = split_works(body)

    # Hardcoded lookup keyed on normalised titles (handles ’ vs ').
    category_of = {norm(k): v for k, v in CATEGORIES.items()}

    rows, unmatched, used_slugs = [], [], set()
    for title, text in works:
        slug = slugify(title)
        if slug in used_slugs:                        # avoid overwriting a file
            slug = re.sub(r"[^a-z0-9]+", "_", norm(title).lower()).strip("_")
        used_slugs.add(slug)

        category = category_of.get(norm(title))
        if category is None:
            unmatched.append(title)
            category = "Unknown"

        (WORKS_DIR / f"{slug}.txt").write_text(text, encoding="utf-8", newline="\n")
        rows.append({
            "title": title,
            "slug": slug,
            "category": category,
            "num_lines": len(text.splitlines()),
            "num_words": len(text.split()),           # simple whitespace split
            "num_chars": len(text),
        })

    with METADATA_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "slug", "category",
                           "num_lines", "num_words", "num_chars"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[4/5] saved {len(rows)} works -> {WORKS_DIR}/ and {METADATA_FILE}")

    # ---- Summary ----------------------------------------------------------
    print("\n[5/5] SUMMARY")
    print(f"Number of works: {len(rows)}")

    print("Works per category:")
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<8} {n}")

    total_words = sum(r["num_words"] for r in rows)
    print(f"Total words: {total_words:,}")

    by_size = sorted(rows, key=lambda r: r["num_words"], reverse=True)
    print("5 largest works:")
    for r in by_size[:5]:
        print(f"  {r['num_words']:>9,}  {r['title']}")
    print("5 smallest works:")
    for r in by_size[-5:]:
        print(f"  {r['num_words']:>9,}  {r['title']}")

    if unmatched:
        print("\nWARNING: no category for these titles (labelled 'Unknown'); "
              "add them to CATEGORIES:")
        for t in unmatched:
            print(f"  - {t!r}")

    if total_words >= MIN_WORDS:
        print(f"\nOK: {total_words:,} words >= {MIN_WORDS:,} required.")
    else:
        print(f"\nERROR: only {total_words:,} words; at least {MIN_WORDS:,} "
              "are required. Check the download/splitting.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
