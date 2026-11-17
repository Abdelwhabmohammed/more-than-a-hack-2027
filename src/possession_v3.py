import os
import sys
import csv
from pathlib import Path

import numpy as np


# --------------------------------------------------
# 1. Project paths and official baseline
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

OFFICIAL = ROOT / "official" / "more-than-a-hack-2027"
STARTER = OFFICIAL / "starter"

os.environ["MTAH_DATA_DIR"] = str(ROOT / "data" / "raw")

sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(STARTER))

import baseline


# --------------------------------------------------
# 2. Match configuration
# --------------------------------------------------

N_FRAMES = 199319

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

SMOOTH_RADIUS = 50

LABELS = {
    0: "H",
    1: "A",
    2: "D",
}


# --------------------------------------------------
# 3. Temporal smoothing
# --------------------------------------------------

def smooth_predictions(raw_codes, radius=SMOOTH_RADIUS):

    raw_codes = np.asarray(raw_codes)

    if len(raw_codes) != N_FRAMES:
        raise ValueError(
            f"Expected {N_FRAMES} frames, got {len(raw_codes)}"
        )

    if not set(np.unique(raw_codes)).issubset({-1, 0, 1}):
        raise ValueError("Unexpected prediction codes")

    # Official baseline:
    # -1 means no reliable nearest-player answer.
    # Convert it to D for the temporal majority vote.
    codes = np.where(raw_codes == -1, 2, raw_codes).astype(np.int8)

    n = len(codes)

    # Prefix sums allow efficient rolling counts.
    prefix = np.zeros((3, n + 1), dtype=np.int32)

    for label in range(3):
        prefix[label, 1:] = np.cumsum(codes == label)

    smoothed = np.empty(n, dtype=np.int8)

    for i in range(n):

        left = max(0, i - radius)
        right = min(n, i + radius + 1)

        counts = prefix[:, right] - prefix[:, left]
        max_count = counts.max()

        tied = np.flatnonzero(counts == max_count)

        # Tie-break 1: keep the current frame's label.
        if codes[i] in tied:
            smoothed[i] = codes[i]

        # Tie-break 2: preserve temporal continuity.
        elif i > 0 and smoothed[i - 1] in tied:
            smoothed[i] = smoothed[i - 1]

        # Tie-break 3: use D if still tied.
        elif 2 in tied:
            smoothed[i] = 2

        else:
            smoothed[i] = tied[0]

    return smoothed


# --------------------------------------------------
# 4. Generate V3 submission
# --------------------------------------------------

def main():

    print("Loading official baseline predictions...")

    raw_codes = baseline.build()

    print("Raw frames:", len(raw_codes))

    print("Applying temporal smoothing...")

    final_codes = smooth_predictions(raw_codes)

    labels = [LABELS[int(code)] for code in final_codes]

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "team_submission_v3.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["frame", "label"])

        for frame, label in enumerate(labels):
            writer.writerow([frame, label])

    print("\nV3 generated successfully.")
    print("Output:", output_path)
    print("Frames:", len(labels))

    for label in ["H", "A", "D"]:
        count = labels.count(label)
        percentage = count / len(labels) * 100

        print(f"{label}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()