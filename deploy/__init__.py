"""Deployment on riichi.dev (docs/reference/07_campaign_and_products.md §1).

Run from the repo root, in the riichidev venv: the mortal env plus riichienv and websockets,
kept out of the frozen mortal env:
    ~/miniconda3/envs/mortal/bin/python -m venv --system-site-packages ~/.venvs/riichidev
    ~/.venvs/riichidev/bin/pip install riichienv==0.4.10 websockets==17.2
"""
import os
import sys

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEPLOY_DIR)
MORTAL_DIR = os.path.join(ROOT, 'mortal')
# Per-game logs, parity reports. Bulky, and they stay out of git.
RUNS_DIR = os.path.join(ROOT, 'workdir', 'riichidev')

# libriichi.so and the flat mortal modules live in mortal/
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)
