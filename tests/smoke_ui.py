"""Browser smoke test of the keyboard demo (not part of the pytest suite: needs Playwright + Microsoft Edge/Chrome).

    python app/app.py --no-browser --port 5057        # terminal 1
    python tests/smoke_ui.py                       # terminal 2  (saves screenshots to figures/app_*.png)
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:5057/"
OUT = Path(__file__).resolve().parent.parent / "figures"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""))


def chips(page, row=0):
    return page.locator(".row").nth(row).locator(".chip").all_inner_texts()


def words(page, row=0):
    return [t.split("\n")[0] for t in chips(page, row)]


with sync_playwright() as pw:
    try:
        browser = pw.chromium.launch(channel="msedge")
    except Exception:
        browser = pw.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.goto(URL)
    page.wait_for_selector(".chip")
    ed = page.locator("#editor")

    # next word at the start of a sentence, capitalised
    check("sentence start: 3 chips", len(words(page)) == 3 and words(page)[0] != "·", str(words(page)))

    # type with the keyboard, mid-word completion after debounce
    ed.click()
    page.keyboard.type("my lord, th", delay=30)
    page.wait_for_timeout(300)
    w = words(page)
    check("mid-word completion starts with the typed prefix", all(x.lower().startswith("th") for x in w), str(w))
    check("status says completing", "completing" in page.locator(".status").first.inner_text())

    # Tab accepts the first chip: word + space, counter counts it as ONE key
    first = w[0]
    keys_before = int(page.locator("#s-keys").inner_text())
    page.keyboard.press("Tab")
    page.wait_for_timeout(300)
    check("Tab inserts first chip + space", ed.input_value() == f"my lord, {first} ", repr(ed.input_value()))
    check("a tap counts as one key", int(page.locator("#s-keys").inner_text()) == keys_before + 1)

    # after a space: next-word prediction; Alt+2 accepts the second chip
    nxt = words(page)
    check("after a space: next-word mode", "next word" in page.locator(".label").first.inner_text())
    page.keyboard.press("Alt+2")
    page.wait_for_timeout(300)
    check("Alt+2 inserts second chip", ed.input_value().endswith(f"{nxt[1]} "), repr(ed.input_value()))

    # no space before punctuation after an accepted word; sentence restarts after '.'
    page.keyboard.type(".", delay=30)
    page.wait_for_timeout(200)
    check("no space before punctuation", ed.input_value().endswith(f"{nxt[1]}.") and not ed.input_value().endswith(" ."), repr(ed.input_value()))
    page.keyboard.type(" ", delay=30)
    page.wait_for_timeout(300)
    check("new sentence after '.': suggestions are capitalised", words(page)[0][:1].isupper(), str(words(page)))

    # clicking a chip
    before = ed.input_value()
    page.locator(".row").first.locator(".chip").nth(2).click()
    page.wait_for_timeout(300)
    check("clicking a chip inserts it", len(ed.input_value()) > len(before) and ed.input_value().endswith(" "), repr(ed.input_value()))

    # counter: savings are computed from keys vs characters
    keys, chars = int(page.locator("#s-keys").inner_text()), int(page.locator("#s-chars").inner_text())
    saved = page.locator("#s-saved").inner_text()
    check("counter consistent with text length", chars == len(ed.input_value()), f"keys={keys} chars={chars} saved={saved}")

    # probabilities toggle
    page.locator("#probs").check()
    page.wait_for_timeout(300)
    check("probabilities shown on chips", page.locator(".chip .prob").count() >= 3, page.locator(".chip").first.inner_text().replace("\n", " "))
    page.locator("#probs").uncheck()

    # model selector + compare mode
    page.select_option("#model", "KN trigram")
    page.locator("#compare").check()
    page.select_option("#model2", "WikiText LSTM")
    ed.fill("")
    page.keyboard.press("Control+a")
    ed.click()
    page.keyboard.type("Ah, villain, ", delay=20)
    page.wait_for_timeout(400)
    check("compare mode shows two rows", page.locator(".row").count() == 2)
    a, b = words(page, 0), words(page, 1)
    check("the two models disagree on 'Ah, villain,'", a != b, f"KN={a} WikiText={b}")
    page.screenshot(path=str(OUT / "app_compare.png"))

    # demo prefix loader resets the counter
    page.locator("#compare").uncheck()
    page.select_option("#model", "LSTM")
    opts = page.locator("#demo option").count()
    page.select_option("#demo", index=1)
    page.wait_for_timeout(400)
    check("demo prefixes available and loadable", opts >= 9 and len(ed.input_value()) > 10 and page.locator("#s-keys").inner_text() == "0",
          f"{opts - 1} prefixes; text={ed.input_value()!r}")
    page.screenshot(path=str(OUT / "app_demo.png"))
    browser.close()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
