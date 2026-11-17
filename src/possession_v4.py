import os
import sys
import csv
from pathlib import Path

import numpy as np


# ==================================================
# 1. Project paths and official baseline
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

OFFICIAL = ROOT / "official" / "more-than-a-hack-2027"
STARTER = OFFICIAL / "starter"

os.environ["MTAH_DATA_DIR"] = str(ROOT / "data" / "raw")

sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(STARTER))

import baseline


# ==================================================
# 2. Match configuration
# ==================================================

N_FRAMES = 199319

SMOOTH_RADIUS = 75

LABELS = {
    0: "H",
    1: "A",
    2: "D",
}

OUTPUT_PATH = ROOT / "outputs" / "team_submission_v4.csv"


# ==================================================
# 3. Validate raw predictions
# ==================================================

def validate_raw_codes(raw_codes):

    raw_codes = np.asarray(raw_codes)

    if raw_codes.ndim != 1:
        raise ValueError("Predictions must be one-dimensional")

    if len(raw_codes) != N_FRAMES:
        raise ValueError(
            f"Expected {N_FRAMES} frames, got {len(raw_codes)}"
        )

    allowed_codes = {-1, 0, 1}

    if not set(np.unique(raw_codes)).issubset(allowed_codes):
        raise ValueError("Unexpected prediction codes")

    return raw_codes.astype(np.int8)


# ==================================================
# 4. Temporal smoothing
# ==================================================

def smooth_predictions(raw_codes):

    raw_codes = validate_raw_codes(raw_codes)

    # Use the official baseline smoothing method.
    # H/A votes are counted.
    # If neither team has a vote, the output is D.

    labels = baseline.smooth(
        raw_codes,
        SMOOTH_RADIUS
    )

    return np.asarray(labels)


# ==================================================
# 5. Write submission
# ==================================================

def write_submission(labels):

    if len(labels) != N_FRAMES:
        raise ValueError("Incorrect number of output labels")

    if not set(np.unique(labels)).issubset({"H", "A", "D"}):
        raise ValueError("Invalid submission labels")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["frame", "label"])

        for frame, label in enumerate(labels):
            writer.writerow([frame, label])


# ==================================================
# 6. Main pipeline
# ==================================================

def main():

    print("=" * 50)
    print("MTAH 2027 - Possession Estimation V4")
    print("=" * 50)

    print("\nLoading official tracking predictions...")

    raw_codes = baseline.build()

    raw_codes = validate_raw_codes(raw_codes)

    print("Frames loaded:", len(raw_codes))

    print("\nApplying temporal smoothing...")
    print("Smoothing radius:", SMOOTH_RADIUS)

    labels = smooth_predictions(raw_codes)

    print("\nGenerating submission...")

    write_submission(labels)

    print("\nV4 generated successfully.")
    print("Output:", OUTPUT_PATH)

    print("\nLabel distribution:")

    for label in ["H", "A", "D"]:

        count = int(np.sum(labels == label))
        percentage = count / N_FRAMES * 100

        print(
            f"{label}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\nTotal frames:", len(labels))


if __name__ == "__main__":
    main()