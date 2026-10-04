"""Small rule tests for src/preprocess.py and the case-map helper in src/data.py.

Run:  python -m pytest tests/test_preprocess.py -q
"""
from collections import Counter, defaultdict

import pytest

from src import preprocess as P
from src.data import restore_case

BREAK = P.BREAK


def units(text):
    """Stage-5 output as a list of speeches (each a list of lines)."""
    return [u.splitlines() for u in text.strip().split("\n\n") if u.strip()]


# ---------------------------------------------------------------- stage 1: characters
def test_curly_apostrophes_become_straight():
    text, _ = P.stage1_normalize("Caesar’s o’er ’tis")
    assert text == "Caesar's o'er 'tis"


def test_single_quote_marks_are_stripped_but_elisions_survive():
    text, (n_open, n_close) = P.stage1_normalize("‘This fair child of mine,’ she said o’ th’ wall.")
    assert text == "This fair child of mine, she said o' th' wall."
    assert (n_open, n_close) == (1, 1)


def test_double_quotes_become_straight_and_are_dropped_by_the_tokenizer():
    text, _ = P.stage1_normalize("“Who’s there?”")
    assert text == '"Who\'s there?"'
    assert P.tokenize(text) == ["Who's", "there", "?"]


# ---------------------------------------------------------------- stage 7: tokenizer
def test_tis_and_oer_are_kept_whole():
    assert P.tokenize("'Tis o'er, ne'er") == ["'Tis", "o'er", ",", "ne'er"]


def test_elisions_contractions_and_possessives_stay_in_one_token():
    assert P.tokenize("kill'd, I'll, Caesar's, i' th' morning") == \
        ["kill'd", ",", "I'll", ",", "Caesar's", ",", "i'", "th'", "morning"]


def test_hyphenated_words_are_split():
    assert P.tokenize("o'er-snowed time-honoured—fear") == ["o'er", "snowed", "time", "honoured", "fear"]


def test_only_six_punctuation_marks_survive():
    assert P.tokenize('Hark! (softly) "Who?" — yes: no, maybe; well.') == \
        ["Hark", "!", "softly", "Who", "?", "yes", ":", "no", ",", "maybe", ";", "well", "."]


def test_digits_become_num():
    assert P.tokenize("the 3rd of May, 1599") == ["the", "<num>", "of", "May", ",", "<num>"]


# ---------------------------------------------------------------- stages 3-5: removals
def test_bracketed_directions_removed_inline_and_multiline():
    text = "[_Aside._] A little more than kin.\n\n[_Exit Hamlet,\nfollowed by Ophelia._]\n\nNext speech."
    out = P.stage3_directions(text)
    assert "Aside" not in out and "Exit" not in out and "Ophelia" not in out
    assert "A little more than kin." in out and "Next speech." in out


def test_unbracketed_enter_paragraph_removed_but_speech_kept():
    text = "Enter Hamlet and Horatio.\n\nHAMLET.\nWell met.\n\nDead March. Enter the funeral of the King.\n\nHORATIO.\nMy lord."
    out = P.stage3_directions(text)
    assert "Enter" not in out and "Dead March" not in out
    assert "HAMLET." in out and "Well met." in out


def test_italic_markers_stripped_but_words_kept():
    assert P.stage4_italics("_Hic et ubique?_ Then we’ll shift") == "Hic et ubique? Then we’ll shift"


def test_speaker_tags_removed_with_period():
    out = P.stage5_speakers("HAMLET.\nTo be, or not to be.\n\nFIRST CITIZEN.\nAy, sir.", is_play=True)
    assert units(out) == [["To be, or not to be."], ["Ay, sir."]]


def test_speaker_tags_removed_without_period():
    out = P.stage5_speakers("FORD\nWell, I hope it be not so.\n\nSECOND LORD\nSo do I.", is_play=True)
    assert units(out) == [["Well, I hope it be not so."], ["So do I."]]


def test_tag_with_speech_on_the_same_line_keeps_the_speech():
    out = P.stage5_speakers("FIRST GAOLER. You shall not pass.\n\nHAMLET.\nNo.", is_play=True)
    assert units(out) == [["You shall not pass."], ["No."]]


def test_ordinary_all_caps_lookalikes_are_not_tags():
    assert P.match_speaker("Hamlet is sad.") is None
    assert P.match_speaker("ROSENCRANTZ and GUILDENSTERN.")[0] == "ROSENCRANTZ and GUILDENSTERN"
    assert P.match_speaker("1 KEEPER.")[0] == "1 KEEPER"


def test_front_matter_headers_and_locations_removed():
    play = ("THE TRAGEDY OF X\n\n\nContents\n\n ACT I\n Scene I. Elsinore.\n\n\nDramatis Personæ\n\nHAMLET, Prince\n\n"
            "SCENE. Elsinore.\n\n\n\nACT I\n\nSCENE I. Elsinore. A platform before\nthe Castle.\n\n\nBARNARDO.\nWho's there?")
    out = P.stage2_structure(play, is_play=True)
    assert "Contents" not in out and "Dramatis" not in out and "HAMLET, Prince" not in out
    assert "ACT I" not in out and "SCENE I." not in out and "the Castle." not in out
    assert "BARNARDO." in out and "Who's there?" in out


def test_sonnet_numbers_and_latin_epigraph_removed_in_poems():
    sonnets = P.stage2_structure("THE SONNETS\n\n                    1\n\nFrom fairest creatures we desire increase,", is_play=False)
    assert "THE SONNETS" not in sonnets and "1" not in sonnets.replace(BREAK, "")
    assert "From fairest creatures" in sonnets
    venus = P.stage2_structure("VENUS AND ADONIS\n\n\n            _Vilia miretur vulgus; mihi flavus Apollo\n"
                               "            Pocula Castalia plena ministret aqua._\n\n\nTO THE RIGHT HONOURABLE\nHENRY WRIOTHESLEY", is_play=False)
    assert "Vilia" not in venus and "WRIOTHESLEY" in venus          # epigraph dropped, dedication kept


# ---------------------------------------------------------------- stage 6: sentences
def test_sentences_are_punctuation_delimited_not_verse_lines():
    verse = "To be, or not to be,\nthat is the question.\nWhether 'tis nobler in the mind"
    assert P.stage6_sentences(verse) == ["To be, or not to be, that is the question.", "Whether 'tis nobler in the mind"]


def test_end_of_a_speech_always_ends_a_sentence():
    text = P.stage5_speakers("HAMLET.\nAlas, poor Yorick\n\nHORATIO.\nI knew him", is_play=True)
    assert P.stage6_sentences(text) == ["Alas, poor Yorick", "I knew him"]


def test_directions_do_not_split_a_speech():
    text = "HAMLET.\nTo be, or not\n[_Aside._]\nto be.\n"
    text = P.stage5_speakers(P.stage3_directions(text), is_play=True)
    assert P.stage6_sentences(text) == ["To be, or not to be."]


def test_punctuation_only_fragments_are_dropped():
    assert P.stage6_sentences("Hello there.\n\n.\n\n—") == ["Hello there."]


# ---------------------------------------------------------------- case map
def test_case_map_ignores_line_and_sentence_initial_words():
    counts = defaultdict(Counter)
    P.case_stats("Hamlet is here.\nI see Hamlet now.", counts)
    assert counts["hamlet"] == Counter({"Hamlet": 1})      # only the mid-line occurrence counts
    assert "i" not in counts                               # line-initial "I" is excluded


def test_case_map_restores_display_case():
    cm = {"hamlet": "Hamlet", "i": "I"}
    assert restore_case(["hamlet", ",", "come", "here", "!"], cm) == "Hamlet, come here!"
    assert restore_case(["then", "i", "go", "."], cm) == "Then I go."
