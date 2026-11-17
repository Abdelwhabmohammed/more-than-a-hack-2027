
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STARTER = ROOT / "official" / "more-than-a-hack-2027" / "starter"

if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

from baseline import smooth


def generate_possession_labels(codes, window=50):

    codes = np.asarray(codes)

    if len(codes) != 199319:
        raise ValueError("Invalid number of frames")

    if window < 0:
        raise ValueError("Window cannot be negative")

    labels = smooth(codes, window)

    if not set(np.unique(labels)).issubset({"H", "A", "D"}):
        raise ValueError("Invalid output labels")

    return labels


def count_possessions(labels):

    labels = np.asarray(labels)

    count = 0
    previous = None

    for label in labels:

        if label in ("H", "A") and label != previous:
            count += 1

        previous = label if label in ("H", "A") else None

    return count
