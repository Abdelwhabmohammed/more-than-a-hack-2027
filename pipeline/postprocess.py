import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
HALVES = half_video_blocks(GAME)

def _find_runs(labels, first, last):
    # find maximal runs of identical labels in labels[first:last+1].
    runs = []
    if first > last:
        return runs
    run_start = first
    run_label = labels[first]
    for f in range(first + 1, last + 1):
        if labels[f] != run_label:
            runs.append((run_start, f - 1, run_label))
            run_start = f
            run_label = labels[f]
    runs.append((run_start, last, run_label))
    return runs

def remove_micro_possessions(labels, min_frames=15):
    # remove possessions shorter than min_frames (default 0.6s)
    labels = labels.copy()
    for _, ff, lf in HALVES:
        runs = _find_runs(labels, ff, lf)
        for i, (start, end, label) in enumerate(runs):
            if label not in ("H", "A"):
                continue
            length = end - start + 1
            if length >= min_frames:
                continue

            # replace with the label of the longer adjacent run
            prev_label = runs[i - 1][2] if i > 0 else "D"
            next_label = runs[i + 1][2] if i < len(runs) - 1 else "D"

            # prefer D if either neighbour is D
            if prev_label == "D" or next_label == "D":
                fill = "D"
            elif prev_label == next_label:
                fill = prev_label
            else:
                # between two different team possessions - use the longer one
                prev_len = runs[i - 1][1] - runs[i - 1][0] + 1 if i > 0 else 0
                next_len = runs[i + 1][1] - runs[i + 1][0] + 1 if i < len(runs) - 1 else 0
                fill = prev_label if prev_len >= next_len else next_label

            labels[start:end + 1] = fill

    return labels

def remove_micro_dead_balls(labels, min_frames=20):
    # remove very short D stretches between same-team possessions
    labels = labels.copy()

    for _ in range(3):
        changed = False
        for _, ff, lf in HALVES:
            runs = _find_runs(labels, ff, lf)
            for i, (start, end, label) in enumerate(runs):
                if label != "D":
                    continue
                length = end - start + 1
                if length >= min_frames:
                    continue
                if i == 0 or i == len(runs) - 1:
                    continue
                prev_label = runs[i - 1][2]
                next_label = runs[i + 1][2]

                # only merge if same team on both sides
                if prev_label == next_label and prev_label in ("H", "A"):
                    labels[start:end + 1] = prev_label
                    changed = True
        if not changed:
            break
    return labels

def merge_same_team_fragments(labels, max_gap=10):
    # merge adjacent same-team possessions separated by a tiny other-team flicker.
    labels = labels.copy()
    for _, ff, lf in HALVES:
        runs = _find_runs(labels, ff, lf)
        for i, (start, end, label) in enumerate(runs):
            if label not in ("H", "A"):
                continue
            length = end - start + 1
            if length > max_gap:
                continue
            if i == 0 or i == len(runs) - 1:
                continue

            prev_label = runs[i - 1][2]
            next_label = runs[i + 1][2]

            # if surrounded by the same team, absorb
            if prev_label == next_label and prev_label in ("H", "A") and prev_label != label:
                labels[start:end + 1] = prev_label
    return labels

def use_x_strategically(labels):
    """apply X to ungraded frames where forward-filling preserves spellF1."""
    labels = labels.copy()
    # set ungraded frames to X
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, ff, lf in HALVES:
        graded[ff:lf + 1] = True
    labels[~graded] = "X"
    return labels

def postprocess(labels, min_poss_frames=25, min_dead_frames=30, max_flicker_gap=15):
    """post-processing pipeline"""
    for _ in range(3):
        labels = remove_micro_dead_balls(labels, min_dead_frames)
        labels = remove_micro_possessions(labels, min_poss_frames)
        labels = merge_same_team_fragments(labels, max_flicker_gap)
    labels = use_x_strategically(labels)
    return labels

def count_possessions(labels):
    """Count the number of possessions"""
    n_poss = 0
    for _, ff, lf in HALVES:
        prev = "X"
        for f in range(ff, lf + 1):
            if labels[f] in ("H", "A"):
                if prev not in ("H", "A") or prev != labels[f]:
                    n_poss += 1
            prev = labels[f]
    return n_poss

def label_stats(labels):
    """print summary statistics of a label array."""
    for label in ["H", "A", "D", "X"]:
        n = int(np.sum(labels == label))
        print(f"  {label}: {n:>8d} frames ({100*n/N_FRAMES:.1f}%)")
    n_poss = count_possessions(labels)
    print(f"  Possessions: {n_poss}")

    # mean possession duration
    durations = []
    for _, ff, lf in HALVES:
        runs = _find_runs(labels, ff, lf)
        for start, end, label in runs:
            if label in ("H", "A"):
                durations.append((end - start + 1) / FPS)
    if durations:
        print(f"mean possession duration: {np.mean(durations):.1f} s")
        print(f"median possession duration: {np.median(durations):.1f} s")

if __name__ == "__main__":
    # demo: postprocess a baseline
    import pandas as pd
    results_dir = os.path.join(ROOT, "evaluation", "results")
    baseline_path = os.path.join(results_dir, "baseline_smooth50.csv")
    df = pd.read_csv(baseline_path)
    labels = df["label"].values.astype("<U1")
    print("before postprocessing:")
    label_stats(labels)
    labels_pp = postprocess(labels)

    print("\nafter postprocessing:")
    label_stats(labels_pp)
