"""Latency of the keyboard demo, measured three ways (CPU, real models, real HTTP on localhost).

    python app/benchmark_latency.py            -> tables/app_latency.csv

  model_call_ms   one `next_word_probs` call (what results/metrics.csv and the RNN notebooks call "latency")
  suggest_ms      server side of a request: parse typed text + model call + the postprocessing stages
  end_to_end_ms   HTTP round trip as the page sees it: JSON in, JSON out, Flask development server
                  (a new connection per request), without the browser's own rendering time

The requests replay typing on 60 validation sentences: after each of the first 5 words (next-word mode) and with 2
letters of the following word typed (completion mode).
"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as demo  # noqa: E402
from src import data as D, postprocess as PP  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402


def typed_states(case_map, n_sent=60):
    sents = D.read_sentences("val")[:600]
    out = []
    for toks in sents:
        if len(toks) < 7:
            continue
        for i in range(1, 6):
            prefix = D.restore_case(toks[:i], case_map) + " "
            out.append(prefix)                                    # next-word mode
            out.append(prefix + toks[i][:2])                      # completion mode (2 letters typed)
        if len(out) >= n_sent * 10:
            break
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def main(port=5099):
    app = demo.create_app()
    models = app.config["MODELS"]
    server = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    texts = typed_states(D.load_case_map())
    print(f"{len(texts)} typed states", flush=True)
    rows = []
    for name, e in models.items():
        # 1. bare model call
        calls = []
        for t in texts[:300]:
            ctx, partial, _ = PP.parse_typed(t)
            ids = [e.vocab.bos_id] + e.vocab.encode(ctx)
            t0 = time.perf_counter(); e.model.next_word_probs(ids); calls.append(1000 * (time.perf_counter() - t0))
        # 2 + 3. through the HTTP API
        srv, e2e = [], []
        for t in texts:
            body = json.dumps({"text": t, "models": [name], "k": 3}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/suggest", data=body, headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            res = json.load(urllib.request.urlopen(req))
            e2e.append(1000 * (time.perf_counter() - t0))
            srv.append(res["results"][name]["ms"])
        calls, srv, e2e = calls[20:], srv[20:], e2e[20:]            # drop warm-up
        rows.append({"model": name, "requests": len(e2e),
                     "model_call_ms_median": np.median(calls), "model_call_ms_p95": pct(calls, 95),
                     "suggest_ms_median": np.median(srv), "suggest_ms_p95": pct(srv, 95),
                     "end_to_end_ms_median": np.median(e2e), "end_to_end_ms_p95": pct(e2e, 95)})
        print(rows[-1], flush=True)
    server.shutdown()
    df = pd.DataFrame(rows).round(2)
    df.to_csv(ROOT / "tables" / "app_latency.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
