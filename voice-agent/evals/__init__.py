"""Evaluation tooling for the local Voice Reliability Lab checkout.

The LiveKit agent module is kept in the project's src/ directory while the
scenario/evaluation modules are kept in evals/. Make the existing agent module
importable when the bridge is started with `python -m evals.pixi_bridge`.
"""
from pathlib import Path
import sys

_SOURCE_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)
