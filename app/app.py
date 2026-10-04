"""
Keyboard demo for the Shakespeare next-word / word-completion models.

    python app/app.py                 # opens http://127.0.0.1:5000 in the browser
    python app/app.py --port 8000 --no-browser

* All models are loaded ONCE at start-up and run on the CPU (no GPU, no internet needed).
* Typed text goes through src/postprocess.py, which imports the SAME normalisation and tokenizer
  functions as src/preprocess.py (nothing is re-implemented); context = the current sentence only.
* The page (app/static/index.html) has no external dependencies: it works offline.
"""
import argparse
import pickle
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from src import data as D, postprocess as PP, rnn as R  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
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
    models["Ensemble"] = Entry(R.Interpolated(lstm, kn3, LAMBDA), vocab, case_map, f"0.8 LSTM + 0.2 KN trigram, Shakespeare")
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


def create_app(models=None, default="LSTM"):
    app = Flask(__name__, static_folder=None)
    app.config["MODELS"] = models if models is not None else load_models()
    lock = threading.Lock()                                   # one prediction at a time (models are not thread-safe)

    @app.get("/")
    def index():
        return send_from_directory(APP_DIR / "static", "index.html")

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
                results[name] = {key: r[key] for key in ("mode", "partial", "sentence_start", "suggestions", "ms")}
        return jsonify({"results": results})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("loading models (CPU) ...", flush=True)
    t0 = time.time()
    app = create_app()
    print(f"loaded {list(app.config['MODELS'])} in {time.time() - t0:.1f}s", flush=True)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"open {url}  (Ctrl+C to stop)", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
