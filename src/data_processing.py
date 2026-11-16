
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STARTER = ROOT / "starter"

os.environ["MTAH_DATA_DIR"] = str(ROOT / "data" / "raw")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

import baseline


def load_tracking_predictions():

    codes = baseline.build()

    if len(codes) != 199319:
        raise ValueError(
            f"Expected 199319 frames, got {len(codes)}"
        )

    if not set(np.unique(codes)).issubset({-1, 0, 1}):
        raise ValueError("Unexpected tracking prediction codes")

    return codes


def get_team_names():

    paths = __import__("games").asset_paths()

    return baseline.team_names(paths["atd_meta"])


if __name__ == "__main__":

    codes = load_tracking_predictions()

    print("Frames:", len(codes))
    print("Teams:", get_team_names())

    print("Home:", int((codes == 0).sum()))
    print("Away:", int((codes == 1).sum()))
    print("No answer:", int((codes == -1).sum()))
