
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_processing import load_tracking_predictions
from possession import generate_possession_labels, count_possessions
from evaluation import validate_submission

import pandas as pd


def main():

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "team_submission.csv"

    print("=" * 50)
    print("MORE THAN A HACK 2027")
    print("BALL POSSESSION PIPELINE")
    print("=" * 50)

    print("\nStep 1: Loading tracking predictions")

    codes = load_tracking_predictions()

    print("Frames loaded:", len(codes))

    print("\nStep 2: Generating possession labels")

    labels = generate_possession_labels(
        codes,
        window=50
    )

    print("Possessions:", count_possessions(labels))

    print("\nStep 3: Creating submission CSV")

    submission = pd.DataFrame({
        "frame": range(199319),
        "label": labels
    })

    submission.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print("Saved:", output_file)

    print("\nStep 4: Running official validator")

    validate_submission(output_file)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
