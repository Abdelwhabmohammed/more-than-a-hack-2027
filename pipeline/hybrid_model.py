import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, half_video_blocks
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
HALVES = half_video_blocks(GAME)
H, A, D_CODE, NOBODY = 0, 1, 2, -1

def _nearest_team_per_frame(features, feature_names, ball_visible):
    """Per-frame team code based on nearest player to ball. NOBODY where no ball."""
    fi = {name: i for i, name in enumerate(feature_names)}
    dist_ratio = features[:, fi["dist_ratio"]]
    n = len(features)
    codes = np.full(n, NOBODY, dtype=np.int8)
    valid = ~np.isnan(dist_ratio)
    # dist_ratio = nearest_home / (nearest_home + nearest_away)
    codes[valid & (dist_ratio < 0.5)] = H
    codes[valid & (dist_ratio >= 0.5)] = A
    return codes

def _smooth_majority_vote(codes, window):
    # majority vote over a centred window. D where nobody has the ball
    if window <= 0:
        return codes.copy()
    n = len(codes)
    home_cum = np.concatenate(([0], np.cumsum(codes == H)))
    away_cum = np.concatenate(([0], np.cumsum(codes == A)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    home = home_cum[hi] - home_cum[lo]
    away = away_cum[hi] - away_cum[lo]
    result = np.full(n, NOBODY, dtype=np.int8)
    seen = (home + away) > 0
    result[seen] = np.where(home[seen] >= away[seen], H, A)
    return result

def _smart_tagging_dead_intervals(st_labels, st_conf, dead_mask):
    # extract dead-ball intervals from smart tagging
    from pipeline.smart_tagging_prior import D as ST_D
    dead = np.zeros(len(st_labels), dtype=bool)
    dead[st_labels == ST_D] = True
    dead[dead_mask] = True
    return dead

def build_hybrid_labels(features, feature_names, ball_visible, ball_xy, st_labels, st_conf, dead_mask, dead_score, smoothing_window=75):
    # build possession labels using a hybrid approach.
    n = len(features)
    fi = {name: i for i, name in enumerate(feature_names)}

    # get nearest-team-to-ball codes
    raw_codes = _nearest_team_per_frame(features, feature_names, ball_visible)
    # apply strong smoothing (±75 frames = ±3 seconds)
    smoothed = _smooth_majority_vote(raw_codes, smoothing_window)

    # use smart tagging selectively
    from pipeline.smart_tagging_prior import H as ST_H, A as ST_A, D as ST_D, UNKNOWN as ST_UNK

    # high-confidence team corrections
    for f in range(n):
        if st_conf[f] >= 0.75 and smoothed[f] in (H, A):
            if st_labels[f] == ST_H and smoothed[f] != H:
                smoothed[f] = H
            elif st_labels[f] == ST_A and smoothed[f] != A:
                smoothed[f] = A

    # dead-ball detection
    # combine multiple signals for dead ball
    n_tracked = np.nan_to_num(features[:, fi["n_total_tracked"]], nan=0)
    frames_since = np.nan_to_num(features[:, fi["frames_since_ball"]], nan=0)
    home_speed = np.nan_to_num(features[:, fi["home_mean_speed"]], nan=0)
    away_speed = np.nan_to_num(features[:, fi["away_mean_speed"]], nan=0)

    dead_labels = np.zeros(n, dtype=bool)
    # signal_a: smart tagging says dead
    dead_labels[st_labels == ST_D] = True
    dead_labels[dead_mask] = True
    # signal_b: high dead score from tracking
    dead_labels[dead_score > 0.5] = True
    # signal_c: very few players tracked
    dead_labels[n_tracked < 8] = True

    # apply dead labels
    smoothed[dead_labels] = D_CODE

    # segment-level cleanup
    for cleanup_pass in range(2):
        for _, ff, lf in HALVES:
            i = ff
            while i <= lf:
                j = i
                while j <= lf and smoothed[j] == smoothed[i]:
                    j += 1
                run_len = j - i
                label = smoothed[i]

                # remove micro possessions only if very short (< 0.5s = 12 frames)
                if label in (H, A) and run_len < 12:
                    prev = smoothed[max(ff, i - 1)] if i > ff else D_CODE
                    nxt = smoothed[min(lf, j)] if j <= lf else D_CODE
                    if prev == nxt and prev in (H, A, D_CODE):
                        smoothed[i:j] = prev
                    elif prev in (H, A):
                        smoothed[i:j] = prev
                    elif nxt in (H, A):
                        smoothed[i:j] = nxt

                # remove micro dead balls between same-team possessions (< 0.5s)
                if label == D_CODE and run_len < 12:
                    prev = smoothed[max(ff, i - 1)] if i > ff else D_CODE
                    nxt = smoothed[min(lf, j)] if j <= lf else D_CODE
                    if prev == nxt and prev in (H, A):
                        smoothed[i:j] = prev
                i = j

    # convert to string labels
    labels = np.full(n, "X", dtype="<U1")
    for _, ff, lf in HALVES:
        for f in range(ff, lf + 1):
            if smoothed[f] == H:
                labels[f] = "H"
            elif smoothed[f] == A:
                labels[f] = "A"
            elif smoothed[f] == D_CODE:
                labels[f] = "D"
            else:
                labels[f] = "D"  # no one: treat as dead ball
    return labels

if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    print("loading features")
    from pipeline.features import extract_features
    feats, names, bvis, bxy = extract_features(data_dir=data_dir)
    print("building smart tagging prior")
    from pipeline.smart_tagging_prior import build_smart_tagging_prior
    st_labels, st_conf, dead_mask, restarts = build_smart_tagging_prior(data_dir=data_dir)
    print("detecting dead ball")
    from pipeline.dead_ball_and_interp import interpolate_ball, detect_dead_ball
    ixy, iconf = interpolate_ball(bxy, bvis)
    dead_score = detect_dead_ball(feats, bvis, bxy, ixy, iconf)
    dead_score[dead_mask] = np.maximum(dead_score[dead_mask], 0.7)

    # trying different smoothing windows
    for window in [50, 75, 100, 125, 150]:
        print(f"\nwindow ±{window} ({window/FPS:.1f}s)")
        labels = build_hybrid_labels(feats, names, bvis, bxy, st_labels, st_conf, dead_mask, dead_score, smoothing_window=window)
        from pipeline.postprocess import count_possessions, label_stats
        label_stats(labels)

    print(f"\nDone in {time.time() - t0:.1f} s")

