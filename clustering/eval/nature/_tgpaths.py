"""Host-independent roots for the code in this deposit.

ROOT = root of this deposit (env TG_ROOT overrides).
DATA = folder for downloaded public data sets (env TG_DATA overrides).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TG_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
DATA = os.environ.get("TG_DATA", os.path.join(ROOT, "data"))
