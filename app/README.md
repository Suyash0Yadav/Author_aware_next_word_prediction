# The Bard's Keyboard - project website and keyboard demo

One Flask app: a results dashboard for every phase of the project plus the keyboard demo.

    pip install -r requirements.txt   # flask, jinja2, torch, ... (pinned)
    python app/app.py                 # loads the models once (CPU), opens http://127.0.0.1:5000
    python app/app.py --port 8000 --no-browser
    python app/app.py --warm-cache    # before a talk: pre-generates the word clouds of all 10 preprocessing stages (cached in app/cache/)

It works **offline** (no CDN, no web fonts, no chart library: figures are the project's PNGs, the interactive charts
are plain HTML/SVG). Models run on the CPU. On start-up the console reports every referenced file that is missing
(`[check] all N referenced files are present`, or the list); the same list is shown as a banner on the pages and at
`/api/health`.

## Pages

| page | what it shows |
|---|---|
| **Overview** | the problem, the pipeline diagram, four headline cards (computed from the result files) |
| **Dataset** | corpus statistics, words per category, the split (filterable list of every work), Zipf, n-gram sparsity, word clouds, archaic keywords |
| **Preprocessing** | stage table, **stage stepper** (a Hamlet / sonnet passage before and after any stage, with that stage's word clouds), residue, OOV decomposition |
| **Models & Results** | the five test-set models (sortable, best value highlighted, metric tooltips), seeds, bootstrap CIs, McNemar, figures, qualitative prefixes, sampled sentences; validation/tuning runs in their own labelled section |
| **Cross-domain** | interactive 2x2 heatmaps (trained on x tested on), archaic accuracy, Shakespeare vs WikiText suggestions side by side |
| **Postprocessing** | the six stages, before/after table, filtering statistics, word clouds, case restoration, the reranking experiment (archaic-word result labelled *exploratory*) |
| **Keyboard** | phone mockup with on-screen QWERTY + the physical keyboard, prediction inspector (top-10 words), compare mode, probabilities, "Favour keystroke savings", live keystroke counter, **Text inspector** (colour a test-play passage or pasted text by the rank each model gave every word) |
| **About** | team placeholders, related work, limitations and reproduction (rendered from the README), key numbers (rendered from `docs/key_numbers.md`) |

The sidebar collapses to a top menu on narrow screens; **Presentation mode** (sidebar / top bar) enlarges fonts and charts.
Every figure has a caption and a badge: TEST, VALIDATION or ALL DATA.

## Where the numbers come from

Nothing is typed into the pages. Everything is read when a page is opened from `results/metrics.csv`, `tables/*.csv`,
`figures/*.png`, `docs/key_numbers.md`, `README.md`, `data/split.csv`, `data/processed/*` and the stage files in
`data/interim/`. These files are never written to (a test checks that); the only thing the app writes is the word-cloud
cache `app/cache/`. The main test table uses the `split == test` rows of `results/metrics.csv`; tuning runs
(`split == val`) appear only in the clearly labelled validation section.

Files: `app.py` (routes + models), `site_data.py` (readers, table/Markdown rendering, stage samples, passages),
`pages.py` (what each page reads), `content.py` (captions, metric definitions, related work - no results),
`inspect_text.py` (text inspector), `templates/`, `static/` (CSS and JS), `demo_prefixes.txt`.

## API

`GET /api/models` · `POST /api/suggest {text, models, k<=10, rerank}` · `POST /api/inspect {text, model}` (<= 300 words) ·
`GET /api/passages`, `GET /api/passage?id=` (test plays only) · `GET /api/preprocessing/sample?sample=&stage=` ·
`GET /api/preprocessing/cloud/<stage>.png` · `GET /api/health` · `GET /figures/<name>.png`.
Typed text is processed with `src/postprocess.py`, which imports the same normalisation and tokenizer as
`src/preprocess.py`; the context is the current sentence only.

## Keyboard shortcuts and the counter

`Tab` accepts the first chip, `Alt+1/2/3` the first, second or third; clicking a chip or an inspector bar works too.
Accepting inserts the word and a space and counts as **one key press**; a following `. , ; : ! ?` removes that space.
Keystroke savings = 1 - keys pressed / characters produced (backspaces are keys; the counter restarts when the box is
emptied; pasted text and text loaded by "Try a prefix" are not counted).

## Tests

    python -m pytest tests/test_site.py tests/test_app.py   # every page returns 200 (also with an empty project), APIs, read-only, offline
    python app/app.py --no-browser --port 5057 &            # then, in another terminal:
    python tests/smoke_ui.py [url] [msedge|chrome|chromium|firefox]   # drives the real pages (Playwright; default Edge, then Chrome)

`python app/make_demo_prefixes.py` regenerates `demo_prefixes.txt`; `python app/benchmark_latency.py` measures model-call,
server-side and end-to-end latency of the keyboard (`tables/app_latency.csv`).
