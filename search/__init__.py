"""Decision-time search (docs/reference/06_search.md).

The pilot (06 §0) runs from the repo root as `python -m search ...`. Its oracle
stage reads arena seeds, so it is lab-only and never part of a deployed bot.
"""
import os
import sys

SEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SEARCH_DIR)
MORTAL_DIR = os.path.join(ROOT, 'mortal')
RUNS_DIR = os.path.join(ROOT, 'workdir', 'search')

# libriichi.so and the flat mortal modules live in mortal/
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)
