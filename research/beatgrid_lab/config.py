import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BEATGRID_DATA", REPO / "data"))
CACHE = DATA / "cache"
REFS = DATA / "refs"
RUNS = REPO / "research" / "runs"
