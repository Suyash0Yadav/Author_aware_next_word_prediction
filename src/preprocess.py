#!/usr/bin/env python3
"""
preprocess.py -- turn the 44 raw Shakespeare works into model-ready text.

Pipeline (each stage's output is saved under data/interim/stageN_<name>/<slug>.txt):

  0  raw works                (data/works, untouched; for the stage table only)
  1  normalized               NFC, curly -> straight quotes/apostrophes, single-quote
                              quotation marks removed
  2  structure_removed        front matter (title, Contents, character list, setting),
                              ACT/SCENE headers + location lines, sonnet/poem numbers,
                              PROLOGUE/EPILOGUE/INDUCTION/THE END headings.
                              Each removed header leaves a "<BREAK>" line = hard unit boundary.
  3  directions_removed       bracketed stage directions [...] / (_..._) (also inline, also
                              multi-line), and un-bracketed Enter/Exit/Exeunt/... paragraphs
  4  italics_stripped         "_" markers removed, the words inside are kept
  5  speaker_tags_removed     "HAMLET." tags removed (before lowercasing: they are detected by
                              being ALL-CAPS). Output = one block per speech, verse lines intact.
  6  sentences                lines of a speech joined, split after . ! ?  (one sentence/line)
  7  tokenized                words / apostrophe-words / 6 punctuation marks / <num>, cased
  8  lowercased               everything lowercased (a case map is saved first, see below)
  9  vocabulary               built on the TRAIN split only (min_freq=2), others -> <unk>
                              => data/processed/{train,val,test}.txt

Also written: data/processed/case_map.json, data/processed/vocab.json,
data/processed/works_index.csv, data/processed/wikitext2_{train,val,test}.txt
(WikiText-2 tokenized with the same punctuation/apostrophe/lowercase rules, no vocabulary yet)
and tables/preprocessing_stages.csv.

Run from anywhere:  python src/preprocess.py
Only the Python standard library + pandas are needed.
"""

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORKS_DIR = DATA / "works"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
BASELINE = DATA / "baseline"
TABLES = ROOT / "tables"

BREAK = "<BREAK>"                       # hard unit boundary left behind by stage 2
SPECIALS = ["<s>", "</s>", "<unk>", "<num>"]
PUNCT = [".", ",", ";", ":", "!", "?"]  # the only punctuation that survives
TERMINAL = {".", "!", "?"}
MIN_FREQ = 2

STAGES = {                               # stage number -> folder name
    1: "stage1_normalized",
    2: "stage2_structure_removed",
    3: "stage3_directions_removed",
    4: "stage4_italics_stripped",
    5: "stage5_speaker_tags_removed",
    6: "stage6_sentences",
    7: "stage7_tokenized",
    8: "stage8_lowercased",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def read(path):
    return Path(path).read_text(encoding="utf-8")


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def paragraphs(text):
    """Split into blank-line separated paragraphs (list of line lists)."""
    out, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def join_paragraphs(paras):
    return "\n\n".join("\n".join(p) for p in paras) + "\n"


# --------------------------------------------------------------------------
# STAGE 1 -- normalize characters
# --------------------------------------------------------------------------
# Words that end in an apostrophe because of an elision (o' the, th' ambition, i' th').
# Used to avoid mistaking them for the closing mark of a single-quote quotation.
ELISION_WORDS = {"o", "th", "i", "a", "t", "e", "d", "s", "m", "wi", "gi"}
CLOSER_PREV = set(".,;:!?)\"”—–-")
CLOSER_NEXT = set(",.;:!?)\"”—–-")


def _is_closing_quote(text, i):
    """Is the right single quote text[i] (U+2019) closing a quotation, not an apostrophe?"""
    prev = text[i - 1] if i > 0 else " "
    nxt = text[i + 1] if i + 1 < len(text) else " "
    if nxt.isalnum():                       # 'tis, nor's, Caesar's  -> apostrophe
        return False
    if prev in CLOSER_PREV:                 # ...thou.'   ...no,'
        return True
    if prev.isalnum():
        if nxt in CLOSER_NEXT:              # 'baby',   'Go, go'-
            return True
        if nxt.isspace():                   # 'O' shall end;  but not  o' th'
            w = re.search(r"([^\W\d_]+)$", text[max(0, i - 12):i])
            word = w.group(1) if w else ""
            return not (word in ELISION_WORDS)   # lowercase elision words only
    return False


def strip_single_quote_marks(text):
    """Remove single-quote *quotation* marks (U+2018 openers and their U+2019 closers).

    Openers are unambiguous. A U+2019 is only deleted when it looks like a closer AND a quote is
    currently open, so ordinary apostrophes (o'er, Caesar's, lords') are never touched.
    A quote opened at the start of every stanza (A Lover's Complaint) counts as one open quote.
    Returns (text, n_openers, n_closers_deleted)."""
    out, depth, n_open, n_close = [], 0, 0, 0
    for i, ch in enumerate(text):
        if ch == "‘":
            n_open += 1
            at_para_start = text[max(0, i - 2):i].count("\n") >= 2 or i == 0
            if not (depth > 0 and at_para_start):    # continuation of a multi-stanza quote
                depth += 1
            continue
        if ch == "’" and depth > 0 and _is_closing_quote(text, i):
            depth -= 1
            n_close += 1
            continue
        out.append(ch)
    return "".join(out), n_open, n_close


def stage1_normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("﻿", "")
    text, n_open, n_close = strip_single_quote_marks(text)
    text = text.replace("’", "'").replace("‘", "'")      # remaining = apostrophes
    text = text.replace("“", '"').replace("”", '"')
    return text, (n_open, n_close)


# --------------------------------------------------------------------------
# STAGE 2 -- remove structure
# --------------------------------------------------------------------------
HEADER_RE = re.compile(r"^\s*(ACT|SCENE)\s+[IVXLC]+\b", re.I)            # ACT I / SCENE II. Rome... / Scene III.
SETTING_RE = re.compile(r"^\s*(?:The\s+)?scene\s*(?::|\.|\s+lies\b)", re.I)  # SCENE: Verona / The scene lies...
HEADING_RE = re.compile(r"^(THE\s+)?(PROLOGUE|EPILOGUE|INDUCTION|END|FINIS)$")  # no period => heading
ROMAN_RE = re.compile(r"^[IVXL]+$")


def stage2_structure(text, is_play):
    paras = paragraphs(text)
    out = []
    if is_play:
        # front matter = everything up to and including the paragraph holding the setting line
        # ("SCENE: Elsinore." / "SCENE. Verona" / "The scene lies ..."). It contains the title,
        # Contents, Dramatis Personae and the setting line.
        cut = next((k for k, p in enumerate(paras)
                    if any(SETTING_RE.match(l) for l in p)), None)
        if cut is None:                                    # fallback: before the first body ACT I
            cut = next((k for k, p in enumerate(paras) if p[0].strip() == "ACT I"
                        and k + 1 < len(paras) and paras[k + 1][0].startswith("SCENE I")), 0) - 1
        out.append([BREAK])
        paras = paras[cut + 1:]
        after_induction = False
        for p in paras:
            first = p[0].strip()
            if HEADER_RE.match(first):                     # whole paragraph: header + wrapped location
                out.append([BREAK])
            elif len(p) == 1 and HEADING_RE.match(first):
                out.append([BREAK])
                after_induction = first == "INDUCTION"
                continue
            elif (after_induction and len(p) == 1 and first.endswith(".") and len(first.split()) <= 8
                  and not match_speaker(first)):           # "Warkworth. Before the castle."
                out.append([BREAK])
            else:
                out.append(p)
            after_induction = False
    else:                                                  # sonnets and poems
        first_done = False
        for k, p in enumerate(paras):
            # a Latin epigraph (Venus and Adonis: Ovid) = an all-italic paragraph right after the title
            if first_done and k <= 2 and re.fullmatch(r"_.*_", " ".join(l.strip() for l in p), re.S):
                out.append([BREAK])
                continue
            q = []
            for l in p:
                s = l.strip()
                if not first_done:                         # title line
                    first_done = True
                    out.append([BREAK])
                    continue
                if re.fullmatch(r"\d+", s) or ROMAN_RE.match(s) or HEADING_RE.match(s):
                    if q:
                        out.append(q)
                        q = []
                    out.append([BREAK])                    # sonnet number / poem numeral / THE END
                else:
                    q.append(l)
            if q:
                out.append(q)
    return join_paragraphs(out)


# --------------------------------------------------------------------------
# STAGE 3 -- remove stage directions
# --------------------------------------------------------------------------
# bracketed, possibly multi-line, but never across a blank line
BRACKET_RE = re.compile(r"\[[^\[\]\n]*(?:\n[^\[\]\n]+)*\]")
PAREN_ITALIC_RE = re.compile(r"\(\s*_[^()]{0,200}?_\s*[.,]?\s*\)")          # AMIENS. (_Sings_.)
DIR_START_RE = re.compile(r"^(Enter|Exit|Exeunt|Re-enter|Flourish|Alarums?|Sennet|Hautboys|"
                          r"Trumpets?|Cornets|Music|Thunder|Retreat)\b")
UNCLOSED_RE = re.compile(r"^\[_?(Enter|Exit|Exeunt|Alarums?|Flourish|Re-enter)\b")
# "Dead March. Enter the funeral ..."  /  "Danish march. A flourish. Enter King, ..."
DIR_SENTENCE_RE = re.compile(r"^(?:[A-Z][^.!?]{0,45}[.!?]\s+){1,3}(Enter|Exit|Exeunt|Re-enter)\b")
DIR_WORD_RE = re.compile(r"\b(Enter|Exit|Exeunt|Re-enter)\b")


def stage3_directions(text):
    text = BRACKET_RE.sub("", text)
    text = PAREN_ITALIC_RE.sub("", text)
    out = []
    for p in paragraphs(text):
        first = p[0].strip()
        joined = " ".join(l.strip() for l in p)
        indented = all(re.match(r"^ \S", l) for l in p)   # directions are often indented by 1 space
        if (DIR_START_RE.match(first) or UNCLOSED_RE.match(first) or DIR_SENTENCE_RE.match(joined)
                or (indented and DIR_WORD_RE.search(joined))):
            continue          # un-bracketed direction paragraph: drop (not a unit boundary)
        out.append(p)
    return join_paragraphs(out)


# --------------------------------------------------------------------------
# STAGE 4 -- italics
# --------------------------------------------------------------------------
def stage4_italics(text):
    return text.replace("_", "")


# --------------------------------------------------------------------------
# STAGE 5 -- speaker tags
# --------------------------------------------------------------------------
MIXED_TAGS = {"All", "Both", "Danes"}


def match_speaker(line):
    """Return (tag, rest_of_line) if `line` starts with an ALL-CAPS speaker tag, else None.
    Handles: 'HAMLET.'  'FIRST CITIZEN.'  '1 KEEPER.'  'ROSENCRANTZ and GUILDENSTERN.'
    'FIRST GAOLER. text on the same line'  and the mixed-case 'All.' 'Both.' 'Danes.'"""
    s = line.strip()
    i = s.find(".")
    if i < 2 or i > 45:
        return None
    cand, rest = s[:i], s[i + 1:]
    if rest and not rest[0].isspace():
        return None
    if any(c in cand for c in '[]()"_'):
        return None
    if cand in MIXED_TAGS:
        return cand, rest.strip()
    words = cand.replace(",", " ").replace("&", " ").split()
    if not words or words[0] in ("ACT", "SCENE"):
        return None
    if not all(w.isupper() or w.isdigit() or w == "and" for w in words):
        return None
    if not any(w.isupper() for w in words):
        return None
    return cand, rest.strip()


def match_bare_tag(line):
    """Speaker tag without a final period on its own line (FORD / SNOUT / SECOND LORD),
    used by a few plays. Only accepted at the start of a paragraph (see stage5_speakers)."""
    s = line.strip()
    words = s.replace(",", " ").split()
    if not (1 <= len(words) <= 4) or len(s) > 30 or any(c in s for c in '.[]()"_?!;:'):
        return None
    if all(w.isupper() or w.isdigit() or w == "and" for w in words) and any(w.isupper() for w in words):
        return s
    return None


def stage5_speakers(text, is_play):
    """Return text where each unit (speech / poem) is a block separated by a blank line.
    A unit ends at: a speaker tag, a <BREAK>, or the end. Blank lines inside a speech
    (left behind by removed directions) are collapsed."""
    units, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            units.append(cur)
        cur = []

    prev_blank = True
    for line in text.splitlines():
        s = line.strip()
        if s == BREAK:
            flush()
            prev_blank = True
            continue
        if not s:
            prev_blank = True
            continue
        was_blank, prev_blank = prev_blank, False
        if is_play:
            if was_blank and match_bare_tag(line):
                flush()
                continue
            m = match_speaker(line)
            if m:
                flush()
                rest = m[1]
                while rest:                                  # 'VARRO. CLAUDIUS.' (two tags)
                    m2 = match_speaker(rest)
                    if not m2:
                        break
                    rest = m2[1]
                if rest:
                    cur.append(rest)
                continue
        cur.append(s)
    flush()
    return "\n\n".join("\n".join(u) for u in units) + "\n"


# --------------------------------------------------------------------------
# STAGE 6 -- sentences
# --------------------------------------------------------------------------
SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s|$|[—–-])")


def split_sentences(text):
    out, start = [], 0
    for m in SENT_END_RE.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def stage6_sentences(text):
    sents = []
    for unit in text.split("\n\n"):
        unit = " ".join(l.strip() for l in unit.splitlines())
        unit = re.sub(r"\s+", " ", unit).strip()
        if unit:
            sents.extend(split_sentences(unit))
    # drop punctuation-only fragments (dotted leaders ". . . .", a lone dash): they hold no words
    return [x for x in sents if re.search(r"\w", x)]


# --------------------------------------------------------------------------
# STAGE 7/8 -- tokenize, lowercase
# --------------------------------------------------------------------------
DASH_RE = re.compile(r"[‐-―−-]")                 # hyphens and dashes -> split words
NUM_RE = re.compile(r"\w*\d(?:[.,]\d+)*\w*")                   # any token with a digit
TOKEN_RE = re.compile(r"'?[^\W\d_]+(?:'[^\W\d_]+)*'?|<num>|[.,;:!?]")


def tokenize(s):
    """Cased tokens: words (apostrophes stay inside: 'tis o'er th' kill'd Caesar's),
    <num>, and the six punctuation marks. Everything else is dropped."""
    s = DASH_RE.sub(" ", s)
    s = NUM_RE.sub(" <num> ", s)
    return TOKEN_RE.findall(s)


def is_word(tok):
    return tok not in PUNCT and tok != "<num>"


def case_stats(stage5_text, counter_by_word):
    """Count casings of each word when it is NOT the first word of a verse line or of a sentence."""
    for unit in stage5_text.split("\n\n"):
        after_terminal = True                                    # unit start = sentence start
        for line in unit.splitlines():
            first_in_line = True
            for tok in tokenize(line):
                if tok in TERMINAL:
                    after_terminal = True
                    continue
                if not is_word(tok):
                    continue
                if not (first_in_line or after_terminal):
                    counter_by_word[tok.lower()][tok] += 1
                first_in_line = False
                after_terminal = False


# --------------------------------------------------------------------------
# WikiText-2 (same punctuation / apostrophe / lowercase rules; no vocabulary yet)
# --------------------------------------------------------------------------
WIKI_CLITIC_RE = re.compile(r" '(s|t|re|ve|ll|d|m)\b", re.I)    # "game 's" -> "game's"
WIKI_NUMJOIN_RE = re.compile(r"(\d) @[,.]@ (?=\d)")             # "1 @,@ 000" -> "1000"


def tokenize_wikitext_line(line):
    line = unicodedata.normalize("NFC", line)
    line = line.replace("’", "'").replace("‘", "'")
    line = WIKI_NUMJOIN_RE.sub(r"\1", line)
    line = line.replace(" @-@ ", " ").replace(" @.@ ", " ").replace(" @,@ ", " ")
    line = WIKI_CLITIC_RE.sub(r"'\1", line)
    return [t.lower() for t in tokenize(line)]


def process_wikitext():
    rows = {}
    for name, out_name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        src = BASELINE / f"wikitext2_{name}.txt"
        if not src.exists():
            print(f"  (skip WikiText-2 {name}: {src} not found)")
            continue
        n_tok, lines = 0, []
        for raw in read(src).splitlines():
            if not raw.strip():
                continue
            toks = tokenize_wikitext_line(raw)
            if toks:
                lines.append(" ".join(toks))
                n_tok += len(toks)
        write(PROCESSED / f"wikitext2_{out_name}.txt", "\n".join(lines) + "\n")
        rows[out_name] = (len(lines), n_tok)
        print(f"  WikiText-2 {out_name}: {len(lines):,} paragraphs, {n_tok:,} tokens")
    return rows


# --------------------------------------------------------------------------
# Stage table helpers
# --------------------------------------------------------------------------
def ws_counts(texts):
    """whitespace-word count and (case-sensitive) vocabulary over several texts; <BREAK> ignored."""
    c = Counter()
    for t in texts:
        c.update(w for w in t.split() if w != BREAK)
    return sum(c.values()), len(c)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    meta = pd.read_csv(DATA / "metadata.csv", encoding="utf-8")
    split = dict(pd.read_csv(DATA / "split.csv", encoding="utf-8")[["slug", "split"]].values)
    slugs = list(meta["slug"])
    is_play = {s: c != "Poetry" for s, c in zip(meta["slug"], meta["category"])}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    stage_text = {n: {} for n in range(0, 6)}     # stage -> slug -> text (stages 0-5)
    stage_sents = {}                              # slug -> list of sentences (stage 6)
    stage_tok = {7: {}, 8: {}}                    # slug -> list of token lists
    quote_stats = {}
    case_counts = defaultdict(Counter)            # train works only

    for slug in slugs:
        raw = read(WORKS_DIR / f"{slug}.txt")
        stage_text[0][slug] = raw
        t1, quote_stats[slug] = stage1_normalize(raw)
        t2 = stage2_structure(t1, is_play[slug])
        t3 = stage3_directions(t2)
        t4 = stage4_italics(t3)
        t5 = stage5_speakers(t4, is_play[slug])
        sents = stage6_sentences(t5)
        tok7 = [tokenize(s) for s in sents]
        tok7 = [t for t in tok7 if any(is_word(x) or x == "<num>" for x in t)]   # drop punctuation-only
        tok8 = [[x.lower() for x in t] for t in tok7]
        for n, t in zip(range(1, 6), (t1, t2, t3, t4, t5)):
            stage_text[n][slug] = t
            write(INTERIM / STAGES[n] / f"{slug}.txt", t)
        stage_sents[slug] = sents
        stage_tok[7][slug], stage_tok[8][slug] = tok7, tok8
        write(INTERIM / STAGES[6] / f"{slug}.txt", "\n".join(sents) + "\n")
        write(INTERIM / STAGES[7] / f"{slug}.txt", "\n".join(" ".join(t) for t in tok7) + "\n")
        write(INTERIM / STAGES[8] / f"{slug}.txt", "\n".join(" ".join(t) for t in tok8) + "\n")
        if split[slug] == "train":
            case_stats(t5, case_counts)

    # ---- case map (train works only; most frequent mid-sentence casing of each word) ----------
    case_map = {w: c.most_common(1)[0][0] for w, c in sorted(case_counts.items())}
    write(PROCESSED / "case_map.json", json.dumps(case_map, ensure_ascii=False, indent=0))

    # ---- vocabulary: train split only, min_freq = 2 --------------------------------------------
    train_slugs = [s for s in slugs if split[s] == "train"]
    word_counts = Counter(t for s in train_slugs for sent in stage_tok[8][s] for t in sent if is_word(t))
    n_train_sents = sum(len(stage_tok[8][s]) for s in train_slugs)
    punct_counts = Counter(t for s in train_slugs for sent in stage_tok[8][s] for t in sent if t in PUNCT)
    num_count = sum(1 for s in train_slugs for sent in stage_tok[8][s] for t in sent if t == "<num>")
    kept = sorted((w for w, c in word_counts.items() if c >= MIN_FREQ), key=lambda w: (-word_counts[w], w))
    itos = SPECIALS + PUNCT + kept
    vocab_set = set(itos)
    unk_count = sum(c for w, c in word_counts.items() if w not in vocab_set)
    counts = {"<s>": n_train_sents, "</s>": n_train_sents, "<unk>": unk_count, "<num>": num_count}
    counts.update({p: punct_counts.get(p, 0) for p in PUNCT})
    counts.update({w: word_counts[w] for w in kept})
    vocab = {"min_freq": MIN_FREQ, "built_on": "train split only", "size": len(itos),
             "specials": SPECIALS, "punctuation": PUNCT, "itos": itos, "counts": counts}
    write(PROCESSED / "vocab.json", json.dumps(vocab, ensure_ascii=False, indent=1))

    # ---- final files (<unk>-mapped, one sentence per line, no <s> </s>) ------------------------
    index_rows, split_tokens = [], Counter()
    for sp in ("train", "val", "test"):
        lines, n_lines = [], 0
        for slug, cat in zip(meta["slug"], meta["category"]):
            if split[slug] != sp:
                continue
            first = n_lines
            for sent in stage_tok[8][slug]:
                toks = [t if t in vocab_set else "<unk>" for t in sent]
                lines.append(" ".join(toks))
                split_tokens[sp] += len(toks)
                n_lines += 1
            index_rows.append({"split": sp, "slug": slug, "category": cat,
                               "first_line": first, "n_sentences": n_lines - first})
        write(PROCESSED / f"{sp}.txt", "\n".join(lines) + "\n")
    pd.DataFrame(index_rows).to_csv(PROCESSED / "works_index.csv", index=False)

    # ---- stage table --------------------------------------------------------------------------
    rows = []
    descr = {
        0: "raw works (data/works)",
        1: "NFC, curly->straight, single-quote marks removed",
        2: "front matter, ACT/SCENE headers, sonnet numbers, Latin epigraph removed",
        3: "bracketed + Enter/Exit/Exeunt directions removed",
        4: "italic underscores stripped",
        5: "speaker tags removed",
    }
    for n in range(0, 6):
        nt, nv = ws_counts(stage_text[n].values())
        rows.append({"stage": n, "name": "raw" if n == 0 else STAGES[n].split("_", 1)[1], "description": descr[n],
                     "unit": "whitespace words", "n_tokens": nt, "vocab_size": nv, "n_sentences": None})
    nt, nv = ws_counts(" ".join(v) for v in stage_sents.values())
    rows.append({"stage": 6, "name": "sentences", "description": "speeches joined, split at . ! ?",
                 "unit": "whitespace words", "n_tokens": nt, "vocab_size": nv,
                 "n_sentences": sum(len(v) for v in stage_sents.values())})
    for n, nm, d in [(7, "tokenized", "punctuation/apostrophe/digit rules, cased"), (8, "lowercased", "lowercased")]:
        c = Counter(t for v in stage_tok[n].values() for sent in v for t in sent)
        rows.append({"stage": n, "name": nm, "description": d, "unit": "tokens",
                     "n_tokens": sum(c.values()), "vocab_size": len(c),
                     "n_sentences": sum(len(v) for v in stage_tok[n].values())})
    rows.append({"stage": 9, "name": "vocabulary", "description": f"train vocab, min_freq={MIN_FREQ}, rest -> <unk> (all splits)",
                 "unit": "tokens", "n_tokens": sum(split_tokens.values()), "vocab_size": len(itos),
                 "n_sentences": n_train_sents + sum(len(stage_tok[8][s]) for s in slugs if split[s] != "train")})
    stage_df = pd.DataFrame(rows)
    stage_df.to_csv(TABLES / "preprocessing_stages.csv", index=False, encoding="utf-8")

    n_open = sum(v[0] for v in quote_stats.values())
    n_close = sum(v[1] for v in quote_stats.values())
    print(stage_df.to_string(index=False))
    print(f"\nsingle-quote openers removed: {n_open}, matching closers removed: {n_close}")
    print(f"vocabulary: {len(itos):,} entries ({len(kept):,} words with count >= {MIN_FREQ}); "
          f"{len(word_counts) - len(kept):,} train word types -> <unk>")
    print("tokens per split:", dict(split_tokens))
    print("\nWikiText-2:")
    process_wikitext()


if __name__ == "__main__":
    main()
