"""Execute a notebook top to bottom and save the outputs in place (the way all notebooks here were produced).

    python tools/run_notebook.py notebooks/03_ngram.ipynb

The kernel's working directory is the notebook's folder. NB_DRY=1 (04_rnn.ipynb only) scores validation instead of test.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print("executed", path.name)
