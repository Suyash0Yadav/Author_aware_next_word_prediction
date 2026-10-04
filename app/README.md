# The Bard's Keyboard - demo

    pip install flask            # (plus the project's torch / numpy / pandas, already used by the notebooks)
    python app/app.py            # loads the models once (CPU), opens http://127.0.0.1:5000
    python app/app.py --port 8000 --no-browser

Works offline (no CDN, no web fonts). Models: KN trigram, LSTM, Ensemble (0.8 LSTM + 0.2 KN), WikiText LSTM
(needs `models/wikitext_lstm_h512_l1_d0.5.pt`; skipped with a message if it is missing).

* Type normally: after a space the bar shows the next word, in the middle of a word it completes it.
* Accept with a click, `Tab` (first chip) or `Alt+1/2/3`. Accepting inserts the word and a space (one key press);
  a following `. , ; : ! ?` removes that space.
* "Compare with" shows two models side by side; "Show probabilities" adds the model probability to each chip.
* The counter shows keys pressed vs characters produced and the running keystroke savings.
* "Try a prefix" loads one of the prefixes in `app/demo_prefixes.txt` (test-play prefixes where the Shakespeare and
  WikiText models clearly differ; regenerate with `python app/make_demo_prefixes.py`).

Tests: `python -m pytest tests/test_app.py` (API, fake models) and, with the app running,
`python tests/smoke_ui.py` (drives the real page in Edge/Chrome with Playwright).
