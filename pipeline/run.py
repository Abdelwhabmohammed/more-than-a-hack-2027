import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR
N_FRAMES = GAMES[GAME]["n_video_frames"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

def write_submission(path, labels):
    """write a frame,label CSV."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as f:
        f.write("frame,label\n")
        for frame, label in enumerate(labels):
            f.write(f"{frame},{label}\n")
    return path


def run_pipeline(data_dir=None, output_path=None, run_scorer=False):
    """execute the full pipeline and return the labels."""
    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    t_total = time.time()

    # stage_1: feature extraction
    print("=" * 70)
    print("Stage 1: Feature extraction from ATD")
    print("=" * 70)
    t0 = time.time()
    from pipeline.features import extract_features
    features, feature_names, ball_visible, ball_xy = extract_features(data_dir=data_dir)
    print(f"  {features.shape[1]} features, ball on {ball_visible.sum()} frames")
    print(f"  [{time.time() - t0:.1f} s]")

    # stage_2: ball interpolation
    print("\n" + "=" * 70)
    print("Stage 2: Ball interpolation")
    print("=" * 70)
    t0 = time.time()
    from pipeline.dead_ball_and_interp import interpolate_ball
    interp_xy, interp_conf = interpolate_ball(ball_xy, ball_visible)
    n_recovered = int(np.sum((interp_conf > 0) & ~ball_visible))
    print(f"  Recovered {n_recovered} ball positions via interpolation")
    print(f"  [{time.time() - t0:.1f} s]")

    # stage_3: smart Tagging prior
    print("\n" + "=" * 70)
    print("Stage 3: Smart Tagging prior")
    print("=" * 70)
    t0 = time.time()
    from pipeline.smart_tagging_prior import build_smart_tagging_prior, H, A, D, UNKNOWN
    st_labels, st_conf, st_dead_mask, restarts = build_smart_tagging_prior(data_dir=data_dir)
    covered = int(np.sum(st_labels != UNKNOWN))
    print(f"  Smart Tagging covers {covered} frames ({100*covered/N_FRAMES:.1f}%)")
    print(f"  Set piece restarts: {len(restarts)}")
    print(f"  [{time.time() - t0:.1f} s]")

    # stage_4: dead-ball detection
    print("\n" + "=" * 70)
    print("Stage 4: Dead-ball detection")
    print("=" * 70)
    t0 = time.time()
    from pipeline.dead_ball_and_interp import detect_dead_ball
    dead_score = detect_dead_ball(features, ball_visible, ball_xy, interp_xy, interp_conf)

    # incorporate the smart tagging dead-ball mask
    dead_score[st_dead_mask] = np.maximum(dead_score[st_dead_mask], 0.85)
    print(f"  Frames with dead score > 0.3: {int(np.sum(dead_score > 0.3))}")
    print(f"  [{time.time() - t0:.1f} s]")

    # stage_5: transition-first Possession Model
    print("\n" + "=" * 70)
    print("Stage 5: Transition-first possession inference & segmentation")
    print("=" * 70)
    t0 = time.time()
    from pipeline.transition_model import build_transition_labels
    labels = build_transition_labels(
        features, feature_names, ball_visible, ball_xy,
        st_labels, st_conf, st_dead_mask, dead_score,
        sensitivity=1.0, min_segment_s=0.4,
        min_poss_s=0.6, min_dead_s=3.0,
        prominence=None
    )
    print(f"  [{time.time() - t0:.1f} s]")

    # stage_6: label statistics
    print("\n" + "=" * 70)
    print("Stage 6: Final label statistics")
    print("=" * 70)
    from pipeline.postprocess import label_stats
    label_stats(labels)

    # Write output
    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "Lets_Play.csv")
    path = write_submission(output_path, labels)
    print(f"Wrote {path}")
    print(f"Total pipeline time: {time.time() - t_total:.1f} s")

    # Optional: run scorer
    if run_scorer:
        print("\n" + "=" * 70)
        print("Scoring")
        print("=" * 70)
        # first validate
        validate_cmd = f'python "{os.path.join(ROOT, "evaluation", "validate_submission.py")}" "{path}"'
        print(f"validating: {validate_cmd}")
        os.system(validate_cmd)

        # then score
        score_cmd = f'python "{os.path.join(ROOT, "evaluation", "run_evaluation.py")}"'
        print(f"\n  Scoring: {score_cmd}")
        os.system(score_cmd)

    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball possession pipeline")
    parser.add_argument("--output", "-o", type=str, default=None, help="output CSV path (default: evaluation/results/Lets_Play.csv)")
    parser.add_argument("--data-dir", type=str, default=None, help="path to dataset (default: MTAH_DATA_DIR env or ./data)")
    parser.add_argument("--score", action="store_true", help="run the scorer after generating the submission")
    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir, output_path=args.output, run_scorer=args.score)

