"""
The project website: a results dashboard for every phase of the project and the keyboard demo, in ONE Flask app.

    python app/app.py                 # opens http://127.0.0.1:5000
    python app/app.py --port 8000 --no-browser
    python app/app.py --warm-cache    # pre-generate all stage word clouds before the server starts (do this before a talk)

* Pages: Overview, Dataset, Preprocessing, Models & Results, Cross-domain, Postprocessing, Keyboard, About.
* Every number, table and figure is read from the project files at request time (results/metrics.csv, tables/*.csv,
  figures/*.png, docs/key_numbers.md, data/split.csv, ...), which are never written to. On start-up the app lists any
  referenced file that is missing.
* All models are loaded ONCE at start-up and run on the CPU. No GPU, no internet: no CDN, no web fonts.
* Typed text goes through src/postprocess.py, which imports the SAME normalisation and tokenizer functions as
  src/preprocess.py (nothing is re-implemented); context = the current sentence only.
"""
import argparse
import pickle
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory  # noqa: E402

import pages  # noqa: E402
from content import FIGURES, NAV, STAGE_DIRS  # noqa: E402
from inspect_text import MAX_WORDS, inspect_passage  # noqa: E402
from site_data import SiteData  # noqa: E402
from src import data as D, postprocess as PP, rnn as R  # noqa: E402

MODEL_DIR = ROOT / "models"
LAMBDA = 0.8                                  # ensemble weight chosen on validation (notebook 04)
MAX_TEXT = 1000


class Entry:
    """A model together with its own vocabulary and case map."""

    def __init__(self, model, vocab, case_map, description=""):
        self.model, self.vocab, self.case_map, self.description = model, vocab, case_map, description


def load_models():
    """Load every model once (CPU). Missing optional models are skipped with a message."""
    import torch
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    models = {}
    vocab, case_map = D.load_vocab(), D.load_case_map()
    kn3 = pickle.load(open(MODEL_DIR / "kn3.pkl", "rb"))
    models["KN trigram"] = Entry(kn3, vocab, case_map, "Kneser-Ney trigram, Shakespeare")
    net, ck = R.load_run("lstm_h512_l1_d0.5", len(vocab), "cpu")
    lstm = R.RNNPredictor(net, "cpu", name="lstm", cfg=ck["cfg"])
    models["LSTM"] = Entry(lstm, vocab, case_map, "LSTM 512x1, Shakespeare")
    models["Ensemble"] = Entry(R.Interpolated(lstm, kn3, LAMBDA), vocab, case_map, "0.8 LSTM + 0.2 KN trigram, Shakespeare")
    try:
        vw = D.load_vocab(corpus="wikitext")
        netw, ckw = R.load_run("wikitext_lstm_h512_l1_d0.5", len(vw), "cpu")
        models["WikiText LSTM"] = Entry(R.RNNPredictor(netw, "cpu", name="wikitext_lstm", cfg=ckw["cfg"]), vw,
                                        D.load_case_map(corpus="wikitext"), "LSTM 512x1, modern English (WikiText-2)")
    except (FileNotFoundError, OSError) as e:
        print(f"[app] WikiText LSTM not available ({e}); skipping it")
    return models


def read_demo_prefixes(path=APP_DIR / "demo_prefixes.txt"):
    """Lines 'prefix  |||  note'; lines starting with # are comments."""
    out = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                text, _, note = line.partition("|||")
                out.append({"text": text.strip(), "note": note.strip()})
    return out


def create_app(models=None, default="LSTM", root=None):
    app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="/static", template_folder=str(APP_DIR / "templates"))
    site = SiteData(root or ROOT)
    app.config["MODELS"] = models if models is not None else load_models()
    app.config["SITE"] = site
    lock = threading.Lock()                                   # one prediction at a time (models are not thread-safe)

    # ---------------------------------------------------------------- pages
    def make_view(key):
        def view():
            try:
                ctx = pages.BUILDERS[key](site)
                return render_template(f"{key}.html", S=site, page=key, nav=NAV, missing=site.missing_files(), **ctx)
            except Exception:                                  # a real bug: show it instead of a blank page
                traceback.print_exc()
                return render_template("error.html", S=site, page=key, nav=NAV, missing=[], trace=traceback.format_exc()), 500
        view.__name__ = f"page_{key.replace('-', '_')}"
        return view

    for key, _, url in NAV:
        app.add_url_rule(url, view_func=make_view(key))

    @app.get("/favicon.ico")
    def favicon():                                            # a tiny inline icon; avoids a 404 in every browser console
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#7a1f2b"/><text x="16" y="23" font-size="20" text-anchor="middle" fill="#fff" font-family="Georgia">B</text></svg>'
        return Response(svg, mimetype="image/svg+xml", headers={"Cache-Control": "max-age=86400"})

    @app.get("/figures/<path:name>")
    def figure(name):
        p = site.figure_path(name)
        if p is None or name not in FIGURES and not name.endswith(".png"):
            abort(404)
        return send_from_directory(p.parent, p.name)

    # ---------------------------------------------------------------- keyboard API (unchanged contract, + `context`)
    @app.get("/api/models")
    def api_models():
        names = list(app.config["MODELS"])
        return jsonify({"models": [{"name": n, "description": app.config["MODELS"][n].description} for n in names],
                        "default": default if default in names else names[0],
                        "demo_prefixes": read_demo_prefixes()})

    @app.post("/api/suggest")
    def api_suggest():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("text", ""), str):
            return jsonify({"error": "expected JSON {text: str, models: [str], k: int}"}), 400
        names = data.get("models") or [default]
        unknown = [n for n in names if n not in app.config["MODELS"]]
        if unknown:
            return jsonify({"error": f"unknown model(s): {unknown}"}), 400
        text = data["text"][-MAX_TEXT:]
        k = max(1, min(int(data.get("k", 3)), 10))
        rerank = bool(data.get("rerank", False))                 # stage 6: order by expected keystrokes saved
        results = {}
        with lock:
            for name in names:
                e = app.config["MODELS"][name]
                t0 = time.perf_counter()
                r = PP.suggest(e.model, e.vocab, text, e.case_map, k=k, rerank=rerank)
                r["ms"] = round(1000 * (time.perf_counter() - t0), 2)
                results[name] = {key: r[key] for key in ("mode", "partial", "sentence_start", "context", "suggestions", "ms")}
        return jsonify({"results": results})

    # ---------------------------------------------------------------- text inspector
    @app.post("/api/inspect")
    def api_inspect():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return jsonify({"error": "expected JSON {text: str, model: str}"}), 400
        name = data.get("model") or default
        if name not in app.config["MODELS"]:
            return jsonify({"error": f"unknown model: {name}"}), 400
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "empty text"}), 400
        with lock:
            res = inspect_passage(app.config["MODELS"][name], text[:6000], max_words=min(int(data.get("max_words", MAX_WORDS)), MAX_WORDS))
        res["model"] = name
        return jsonify(res)

    @app.get("/api/passages")
    def api_passages():
        return jsonify({"passages": site.passages()})

    @app.get("/api/passage")
    def api_passage():
        text = site.passage_text(request.args.get("id", ""))
        if text is None:
            return jsonify({"error": "unknown passage"}), 404
        return jsonify({"id": request.args["id"], "text": text})

    # ---------------------------------------------------------------- preprocessing stepper
    @app.get("/api/preprocessing/sample")
    def api_sample():
        try:
            stage = int(request.args.get("stage", ""))
        except ValueError:
            return jsonify({"error": "stage must be an integer 0-9"}), 400
        res = site.stage_sample(request.args.get("sample", ""), stage)
        if res is None:
            return jsonify({"error": "unknown sample or stage"}), 404
        return jsonify(res)

    @app.get("/api/preprocessing/cloud/<int:stage>.png")
    def api_cloud(stage):
        png = site.cloud_png(stage)
        if png is None:
            abort(404)
        return Response(png, mimetype="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/health")
    def api_health():
        _, notes = site.main_test_table()
        return jsonify({"missing_files": site.missing_files(), "notes": notes, "models": list(app.config["MODELS"])})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--warm-cache", action="store_true",
                    help="generate the word clouds of all preprocessing stages before the server starts (nothing is slow later)")
    args = ap.parse_args()
    print("loading models (CPU) ...", flush=True)
    t0 = time.time()
    app = create_app()
    print(f"loaded {list(app.config['MODELS'])} in {time.time() - t0:.1f}s", flush=True)
    site = app.config["SITE"]
    missing = site.missing_files()
    total = len(site.referenced_files())
    if missing:
        print(f"[check] {len(missing)} of {total} referenced files are MISSING (the affected parts show a notice):", flush=True)
        for f in missing:
            print(f"   - {f}", flush=True)
    else:
        print(f"[check] all {total} referenced files are present", flush=True)
    _, notes = site.main_test_table()
    for n in notes:
        print(f"[check] {n}", flush=True)
    if args.warm_cache:
        t0 = time.time()
        print("[warm-cache] generating the preprocessing word clouds ...", flush=True)
        rows = site.warm_cloud_cache(log=lambda r: print(f"   stage {r['stage']}: {r['seconds']:.1f}s "
                                                         f"{'generated' if r['generated'] else 'already cached' if r['available'] else 'NOT AVAILABLE (input files missing)'}", flush=True))
        print(f"[warm-cache] done in {time.time() - t0:.1f}s ({sum(r['available'] for r in rows)}/{len(rows)} stages ready)", flush=True)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"open {url}  (Ctrl+C to stop)", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
