"""Browser smoke test of the whole website (not part of the pytest suite: needs Playwright + Microsoft Edge/Chrome).

    python app/app.py --no-browser --port 5057        # terminal 1
    python tests/smoke_ui.py [http://127.0.0.1:5057] [msedge|chrome|chromium|firefox]    # terminal 2   (screenshots -> figures/app_*.png)

The browser defaults to Edge, then Chrome (installed browsers). "chromium" / "firefox" use Playwright's own builds
(`playwright install chromium firefox` first).

Drives the real pages: every page loads without console errors, the sortable main table, the preprocessing stage
stepper, the cross-domain heatmaps, the phone keyboard (physical + on-screen keys, Tab, Alt+N, punctuation rule,
inspector bars, compare mode, probabilities and reranking toggles), the text inspector, presentation mode and the
narrow-screen layout (no horizontal scrolling).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:5057"
BROWSER = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(__file__).resolve().parent.parent / "figures"
PAGES = ["/", "/dataset", "/preprocessing", "/models", "/cross-domain", "/postprocessing", "/keyboard", "/about"]
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""))


with sync_playwright() as pw:
    if BROWSER == "firefox":
        browser = pw.firefox.launch()
    elif BROWSER == "chromium":
        browser = pw.chromium.launch()
    elif BROWSER in ("msedge", "chrome"):
        browser = pw.chromium.launch(channel=BROWSER)
    else:
        try:
            browser = pw.chromium.launch(channel="msedge")
        except Exception:
            browser = pw.chromium.launch(channel="chrome")
    print("browser:", BROWSER or "msedge/chrome", browser.version)
    page = browser.new_page(viewport={"width": 1400, "height": 1100})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    # ---------------------------------------------------------------- all pages load
    for url in PAGES:
        resp = page.goto(BASE + url)
        page.wait_for_load_state("networkidle")
        check(f"page {url} loads", resp.status == 200 and page.locator("h1").count() >= 1)
    check("no console / page errors on any page", errors == [], str(errors))

    # ---------------------------------------------------------------- main table
    page.goto(BASE + "/models")
    th = page.locator("table.main thead th")
    th.nth(1).click()
    ppl = [float(x) for x in page.locator("table.main tbody tr td:nth-child(2)").all_inner_texts()]
    check("main table sorts by perplexity (ascending)", ppl == sorted(ppl), str(ppl))
    check("best cells are highlighted", page.locator("table.main td.best").count() >= 8)

    # ---------------------------------------------------------------- stepper
    page.goto(BASE + "/preprocessing")
    page.locator('#stage-buttons button[data-stage="3"]').click()
    page.wait_for_function("document.getElementById('after').textContent.length > 20 && document.getElementById('cloud-after').naturalWidth > 0", timeout=60000)
    before, after = page.locator("#before").inner_text(), page.locator("#after").inner_text()
    check("stepper: directions are in 'before' and gone in 'after'", "Enter Francisco" in before and "Enter Francisco" not in after)
    page.locator("#stepper").screenshot(path=str(OUT / "app_stepper.png"))

    # ---------------------------------------------------------------- heatmaps
    page.goto(BASE + "/cross-domain")
    page.select_option("#metric", "ksr")
    check("heatmaps: 2 models x 2x2 cells", page.locator("#heatmaps td").count() == 8)

    # ---------------------------------------------------------------- keyboard
    page.goto(BASE + "/keyboard")
    page.wait_for_selector(".chip")
    ed = page.locator("#editor")
    for k in "my lord ":
        page.locator(f'.key[data-k="{"SPACE" if k == " " else k}"]').click()
    page.wait_for_timeout(300)
    check("on-screen keys type into the box", ed.input_value() == "my lord ", repr(ed.input_value()))
    check("inspector shows 10 bars; first bar = first chip",
          page.locator(".bar-row").count() == 10 and page.locator(".bar-row .bar-label").first.inner_text() == page.locator(".chip").first.inner_text().split("\n")[0])
    page.locator('.key[data-k="SHIFT"]').click()
    page.locator('.key[data-k="h"]').click()
    check("shift capitalises", ed.input_value().endswith("H"))
    page.keyboard.type("a", delay=30)
    page.wait_for_timeout(300)
    check("completion mode after a typed prefix", "completing" in page.locator(".sg-status").first.inner_text())
    page.keyboard.press("Tab")
    page.wait_for_timeout(300)
    check("Tab accepts the first chip and adds a space", ed.input_value().endswith(" ") and len(ed.input_value()) > 10, repr(ed.input_value()))
    before = ed.input_value()
    page.locator('.key[data-k=","]').click()
    check("no space before punctuation (on-screen key)", ed.input_value() == before[:-1] + ",")
    page.locator("#reset").click()
    page.keyboard.type("my lord ", delay=20)
    page.wait_for_timeout(400)
    page.keyboard.press("Alt+2")
    page.wait_for_timeout(300)
    check("Alt+2 accepts the second chip", ed.input_value().startswith("my lord ") and ed.input_value().endswith(" ") and len(ed.input_value()) > 9, repr(ed.input_value()))
    page.locator("#reset").click()
    page.keyboard.type("my lord ", delay=20)
    page.wait_for_timeout(400)
    page.locator(".bar-row").nth(4).click()
    page.wait_for_timeout(300)
    check("clicking an inspector bar accepts that word", ed.input_value().endswith(" ") and len(ed.input_value()) > 9)
    page.locator("#compare").check()
    page.select_option("#model2", "WikiText LSTM")
    page.locator("#probs").check()
    page.locator("#rerank").check()
    page.wait_for_timeout(500)
    check("compare mode: 2 chip rows, 2 inspector panels, probabilities on chips", page.locator(".sg-row").count() == 2 and page.locator(".insp-panel").count() == 2 and page.locator(".chip .prob").count() >= 3)
    page.screenshot(path=str(OUT / "app_keyboard.png"), full_page=True)

    # ---------------------------------------------------------------- text inspector
    page.locator('[data-tab="tab-inspect"]').click()
    page.wait_for_function("document.querySelectorAll('#ti-passage option').length > 3")
    page.select_option("#ti-passage", index=2)
    page.wait_for_selector(".ti-card .tok", timeout=30000)
    classes = set(page.locator(".ti-card .tok").evaluate_all("els => els.map(e => e.className.split(' ')[1])"))
    check("text inspector colours words (green and red present)", {"c-g", "c-r"} <= classes, str(sorted(classes)))
    page.locator("#ti-compare").check()
    page.select_option("#ti-model2", "KN trigram")
    page.locator("#ti-go").click()
    page.wait_for_function("document.querySelectorAll('.ti-card').length == 2", timeout=30000)
    check("compare coloring: two cards", page.locator(".ti-card").count() == 2)
    page.screenshot(path=str(OUT / "app_text_inspector.png"), full_page=True)

    # ---------------------------------------------------------------- presentation mode + narrow screen
    page.locator("#sidebar .pres-toggle").click()
    check("presentation mode enlarges the fonts", float(page.evaluate("getComputedStyle(document.documentElement).fontSize")[:-2]) > 20)
    narrow = browser.new_page(viewport={"width": 400, "height": 800})
    for url in PAGES:
        narrow.goto(BASE + url)
        narrow.wait_for_load_state("networkidle")
        narrow.wait_for_timeout(300)
        check(f"narrow screen: no horizontal scrolling on {url}", not narrow.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"))
    narrow.goto(BASE + "/")
    narrow.locator("#menu").click()
    check("narrow screen: the menu opens", narrow.locator("#sidebar nav a").first.is_visible())
    browser.close()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
