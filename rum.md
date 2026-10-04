pip install -r requirements.txt  # flask, jinja2, torch, ... (pinned)

# open the UI (loads models once on CPU, opens http://127.0.0.1:5000 in the browser)
python app/app.py

# before a talk: pre-generate all word clouds so nothing is slow
python app/app.py --warm-cache

# other port, without opening a browser
python app/app.py --port 8000 --no-browser
