
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "data" / "processed" / "model_input.csv"

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "possession_candidate_v3.csv"
)

FPS = 25

HYSTERESIS_THRESHOLD = 0.25

WEIGHT_DIRECT = 0.65
WEIGHT_1S = 0.25
WEIGHT_3S = 0.10

MIN_EVIDENCE_WEIGHT = 0.20
MIN_CONFIDENCE = 0.60

# ATD frame_id is one-based.
# Reset hysteresis at the first graded frame of each half.
FIRST_HALF_START_ID = 11186
FIRST_HALF_END_ID = 90652

SECOND_HALF_START_ID = 110809

# Submission frame numbers are zero-based.
GRADED_RANGES = [
    (11185, 90651),
    (110808, 190197),
]


# ============================================================
# 2. Hysteresis function
# ============================================================

def apply_hysteresis(margins, threshold):

    """
    Stateful hysteresis over the direct distance-margin signal.

    Positive margin -> Barcelona.
    Negative margin -> Athletic.

    Inside the deadband, keep the previous state.
    Missing observations produce no direct evidence,
    but do not erase the internal state.
    """

    from starter import baseline

    result = np.full(
        len(margins),
        baseline.NOBODY,
        dtype=np.int8
    )

    state = None

    for i, margin in enumerate(margins):

        if not np.isfinite(margin):
            continue

        if state is None:

            if margin > threshold:
                state = baseline.HOME

            elif margin < -threshold:
                state = baseline.AWAY

        elif state == baseline.HOME:

            if margin < -threshold:
                state = baseline.AWAY

        elif state == baseline.AWAY:

            if margin > threshold:
                state = baseline.HOME

        if state is not None:
            result[i] = state

    return result


# ============================================================
# 3. Load and validate data
# ============================================================

def load_data():

    print("Loading prepared features...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "frame_id",
        "ball_visible",
        "distance_margin",
        "barcelona_ratio_1s",
        "athletic_ratio_1s",
        "barcelona_ratio_3s",
        "athletic_ratio_3s",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) != 199319:
        raise ValueError(
            f"Expected 199319 rows, got {len(df)}"
        )

    df = df.sort_values(
        "frame_id"
    ).reset_index(drop=True)

    expected_ids = np.arange(1, 199320)

    actual_ids = df["frame_id"].to_numpy()

    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(
            "Frame IDs must be unique and continuous "
            "from 1 to 199319."
        )

    print("Input validation passed.")
    print("Rows:", len(df))

    return df


# ============================================================
# 4. Prepare numerical features
# ============================================================

def prepare_features(df):

    ball_visible = (
        pd.to_numeric(
            df["ball_visible"],
            errors="coerce"
        )
        .fillna(0)
        .astype(bool)
        .to_numpy()
    )

    margins = pd.to_numeric(
        df["distance_margin"],
        errors="coerce"
    ).to_numpy(dtype=float)

    home_1s = (
        pd.to_numeric(
            df["barcelona_ratio_1s"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )

    away_1s = (
        pd.to_numeric(
            df["athletic_ratio_1s"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )

    home_3s = (
        pd.to_numeric(
            df["barcelona_ratio_3s"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )

    away_3s = (
        pd.to_numeric(
            df["athletic_ratio_3s"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=float)
    )

    # Direct evidence requires a visible ball
    # and an available distance margin.

    direct_signal_valid = (
        ball_visible
        & np.isfinite(margins)
    )

    margins_for_hysteresis = margins.copy()

    margins_for_hysteresis[
        ~direct_signal_valid
    ] = np.nan

    return (
        margins_for_hysteresis,
        home_1s,
        away_1s,
        home_3s,
        away_3s,
    )


# ============================================================
# 5. Apply hysteresis separately to each half
# ============================================================

def build_direct_signal(df, margins):

    from starter import baseline

    frame_ids = df["frame_id"].to_numpy()

    direct_codes = np.full(
        len(df),
        baseline.NOBODY,
        dtype=np.int8
    )

    # First half starts exactly at the first graded ATD ID.
    first_half_mask = (
        (frame_ids >= FIRST_HALF_START_ID)
        & (frame_ids <= FIRST_HALF_END_ID)
    )

    # Second half starts at its kickoff ATD ID.
    second_half_mask = (
        frame_ids >= SECOND_HALF_START_ID
    )

    # First half: fresh independent hysteresis state.
    first_indices = np.flatnonzero(
        first_half_mask
    )

    if len(first_indices):

        direct_codes[first_indices] = apply_hysteresis(
            margins[first_indices],
            HYSTERESIS_THRESHOLD
        )

    # Second half: reset state completely.
    second_indices = np.flatnonzero(
        second_half_mask
    )

    if len(second_indices):

        direct_codes[second_indices] = apply_hysteresis(
            margins[second_indices],
            HYSTERESIS_THRESHOLD
        )

    # Frames before the first graded frame and halftime
    # remain without direct hysteresis evidence.

    direct_valid = (
        direct_codes != baseline.NOBODY
    )

    direct_home = (
        direct_valid
        & (direct_codes == baseline.HOME)
    ).astype(float)

    direct_away = (
        direct_valid
        & (direct_codes == baseline.AWAY)
    ).astype(float)

    return (
        direct_valid,
        direct_home,
        direct_away,
    )


# ============================================================
# 6. Build temporal evidence
# ============================================================

def build_temporal_evidence(
    home_1s,
    away_1s,
    home_3s,
    away_3s
):

    total_1s = home_1s + away_1s
    total_3s = home_3s + away_3s

    valid_1s = total_1s > 0
    valid_3s = total_3s > 0

    home_1s_normalized = np.divide(
        home_1s,
        total_1s,
        out=np.zeros_like(home_1s),
        where=valid_1s
    )

    away_1s_normalized = np.divide(
        away_1s,
        total_1s,
        out=np.zeros_like(away_1s),
        where=valid_1s
    )

    home_3s_normalized = np.divide(
        home_3s,
        total_3s,
        out=np.zeros_like(home_3s),
        where=valid_3s
    )

    away_3s_normalized = np.divide(
        away_3s,
        total_3s,
        out=np.zeros_like(away_3s),
        where=valid_3s
    )

    return (
        valid_1s,
        valid_3s,
        home_1s_normalized,
        away_1s_normalized,
        home_3s_normalized,
        away_3s_normalized,
    )


# ============================================================
# 7. Combine evidence
# ============================================================

def estimate_possession(
    direct_valid,
    direct_home,
    direct_away,
    temporal_data
):

    (
        valid_1s,
        valid_3s,
        home_1s,
        away_1s,
        home_3s,
        away_3s,
    ) = temporal_data

    home_score = (
        WEIGHT_DIRECT * direct_home
        + WEIGHT_1S * home_1s * valid_1s
        + WEIGHT_3S * home_3s * valid_3s
    )

    away_score = (
        WEIGHT_DIRECT * direct_away
        + WEIGHT_1S * away_1s * valid_1s
        + WEIGHT_3S * away_3s * valid_3s
    )

    evidence_weight = (
        WEIGHT_DIRECT * direct_valid.astype(float)
        + WEIGHT_1S * valid_1s.astype(float)
        + WEIGHT_3S * valid_3s.astype(float)
    )

    total_score = home_score + away_score

    home_confidence = np.divide(
        home_score,
        total_score,
        out=np.zeros_like(home_score),
        where=total_score > 0
    )

    away_confidence = np.divide(
        away_score,
        total_score,
        out=np.zeros_like(away_score),
        where=total_score > 0
    )

    labels = np.full(
        len(home_score),
        "X",
        dtype=object
    )

    enough_evidence = (
        evidence_weight >= MIN_EVIDENCE_WEIGHT
    )

    home_condition = (
        enough_evidence
        & (home_confidence >= MIN_CONFIDENCE)
        & (home_confidence > away_confidence)
    )

    away_condition = (
        enough_evidence
        & (away_confidence >= MIN_CONFIDENCE)
        & (away_confidence > home_confidence)
    )

    labels[home_condition] = "H"
    labels[away_condition] = "A"

    return (
        labels,
        evidence_weight,
        home_confidence,
        away_confidence,
        direct_valid,
        valid_1s,
        valid_3s,
    )


# ============================================================
# 8. Save candidate submission
# ============================================================

def save_candidate(df, labels):

    submission = pd.DataFrame({
        "frame": df["frame_id"].astype(int) - 1,
        "label": labels,
    })

    if len(submission) != 199319:
        raise ValueError(
            "Incorrect submission row count."
        )

    if submission["frame"].duplicated().any():
        raise ValueError(
            "Duplicate frames detected."
        )

    if not submission["label"].isin(
        ["H", "A", "D", "X"]
    ).all():
        raise ValueError(
            "Invalid output labels."
        )

    if submission["frame"].iloc[0] != 0:
        raise ValueError(
            "First frame must be zero."
        )

    if submission["frame"].iloc[-1] != 199318:
        raise ValueError(
            "Last frame must be 199318."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    submission.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    return submission


# ============================================================
# 9. Report
# ============================================================

def print_distribution(submission, name):

    print(f"\n{name}")

    counts = submission["label"].value_counts()

    for label in ["H", "A", "D", "X"]:

        count = int(counts.get(label, 0))

        percentage = (
            count / len(submission) * 100
        )

        print(
            f"{label}: {count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 65)
    print("POSSESSION ESTIMATOR V3")
    print("=" * 65)

    df = load_data()

    (
        margins,
        home_1s,
        away_1s,
        home_3s,
        away_3s,
    ) = prepare_features(df)

    print("\nApplying Hysteresis...")
    print("Threshold:", HYSTERESIS_THRESHOLD)
    print("First half reset ID:", FIRST_HALF_START_ID)
    print("Second half reset ID:", SECOND_HALF_START_ID)

    (
        direct_valid,
        direct_home,
        direct_away,
    ) = build_direct_signal(df, margins)

    temporal_data = build_temporal_evidence(
        home_1s,
        away_1s,
        home_3s,
        away_3s
    )

    print("\nCombining evidence...")

    (
        labels,
        evidence_weight,
        home_confidence,
        away_confidence,
        direct_valid,
        valid_1s,
        valid_3s,
    ) = estimate_possession(
        direct_valid,
        direct_home,
        direct_away,
        temporal_data
    )

    submission = save_candidate(df, labels)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print_distribution(
        submission,
        "Full dataset distribution"
    )

    frame_numbers = submission["frame"]

    graded_mask = pd.Series(
        False,
        index=submission.index
    )

    for start, end in GRADED_RANGES:

        graded_mask |= frame_numbers.between(
            start,
            end
        )

    graded_submission = submission.loc[
        graded_mask
    ]

    print_distribution(
        graded_submission,
        "Graded-frame distribution"
    )

    print("\nEvidence diagnostics:")

    print(
        "Direct evidence:",
        int(direct_valid.sum())
    )

    print(
        "1-second temporal evidence:",
        int(valid_1s.sum())
    )

    print(
        "3-second temporal evidence:",
        int(valid_3s.sum())
    )

    print(
        "Insufficient evidence:",
        int(
            (
                evidence_weight
                < MIN_EVIDENCE_WEIGHT
            ).sum()
        )
    )

    print(
        "\nFrames with H/A/X:",
        submission["label"].value_counts().to_dict()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nImportant:")
    print("- This is a heuristic candidate, not a trained ML model.")
    print("- X means insufficient evidence, not dead ball.")
    print("- D is not inferred from missing ball observations.")
    print("- No ground-truth labels were used.")

    print("\nV3 completed successfully.")


if __name__ == "__main__":
    main()
