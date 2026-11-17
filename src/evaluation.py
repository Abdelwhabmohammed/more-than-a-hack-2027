
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VALIDATOR = (
    ROOT
    / "official"
    / "more-than-a-hack-2027"
    / "evaluation"
    / "validate_submission.py"
)


def validate_submission(csv_path):

    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(csv_path)
        ],
        cwd=str(ROOT / "official" / "more-than-a-hack-2027"),
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Official submission validation failed")

    return True
