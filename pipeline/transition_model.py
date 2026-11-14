import os
import sys
import time
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, half_video_blocks

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
HALVES = half_video_blocks(GAME)
PITCH = GAMES[GAME]["pitch"]

H, A, D_CODE = 0, 1, 2


# step_1: Detect candidate transition points
def _smooth(arr, window):
    """smooth with a uniform filter, handling NaN by treating as 0"""
    clean = np.nan_to_num(arr, nan=0.0)
    return uniform_filter1d(clean.astype(np.float64), size=window, mode='nearest')

def detect_transitions(features, feature_names, ball_visible, ball_xy, st_labels, st_conf, dead_score, sensitivity=1.0):
    """detect candidate possession change-points"""
    fi = {name: i for i, name in enumerate(feature_names)}
    n = len(features)
    change_scores = np.zeros(n, dtype=np.float64)

    # signal_1: dist_ratio crossover (team switching from closer to farther)
    dist_ratio = features[:, fi["dist_ratio"]]
    # smooth the dist_ratio to reduce noise
    dr_clean = np.nan_to_num(dist_ratio, nan=0.5)
    dr_smooth = _smooth(dr_clean, FPS * 2)
    dr_deriv = np.abs(np.gradient(dr_smooth))
    # normalize to [0, 1]
    if dr_deriv.max() > 0:
        dr_signal = dr_deriv / np.percentile(dr_deriv[dr_deriv > 0], 95)
        dr_signal = np.clip(dr_signal, 0, 1)
    else:
        dr_signal = np.zeros(n)
    change_scores += dr_signal * 3.0 * sensitivity

    # signal_2: dist_ratio zero-crossing (the actual team switch moment)
    above = dr_smooth > 0.5
    crossings = np.flatnonzero(above[1:] != above[:-1])
    for c in crossings:
        lo = max(0, c - FPS // 2)
        hi = min(n, c + FPS // 2)
        change_scores[lo:hi] += 2.0 * sensitivity

    # signal_3: smart tagging phase boundaries
    from pipeline.smart_tagging_prior import H as ST_H, A as ST_A, D as ST_D, UNKNOWN as ST_UNK

    # where ST label changes, it's mostly a real transition
    st_changes = np.flatnonzero(st_labels[1:] != st_labels[:-1])
    for c in st_changes:
        if st_labels[c] != ST_UNK and st_labels[c + 1] != ST_UNK:
            # both sides have known labels = strong signal
            lo = max(0, c - FPS // 4)
            hi = min(n, c + FPS // 4 + 1)
            change_scores[lo:hi] += 4.0 * sensitivity
        elif st_labels[c] != ST_UNK or st_labels[c + 1] != ST_UNK:
            # One side known → moderate signal
            lo = max(0, c - FPS // 4)
            hi = min(n, c + FPS // 4 + 1)
            change_scores[lo:hi] += 2.0 * sensitivity

    # signal_4: dead-ball score transitions
    ds_smooth = _smooth(dead_score, FPS)
    ds_deriv = np.abs(np.gradient(ds_smooth))
    if ds_deriv.max() > 0:
        ds_signal = ds_deriv / np.percentile(ds_deriv[ds_deriv > 0], 95)
        ds_signal = np.clip(ds_signal, 0, 1)
    else:
        ds_signal = np.zeros(n)
    change_scores += ds_signal * 2.0 * sensitivity

    # signal_5: ball trajectory discontinuity
    length, width = PITCH
    for f in range(1, n):
        if ball_visible[f] and ball_visible[f - 1]:
            dx = (ball_xy[f, 0] - ball_xy[f - 1, 0]) * length
            dy = (ball_xy[f, 1] - ball_xy[f - 1, 1]) * width
            jump = np.sqrt(dx**2 + dy**2)
            if jump > 10.0:
                lo = max(0, f - FPS)
                hi = min(n, f + FPS // 4)
                change_scores[lo:hi] += 3.0 * sensitivity

    # signal_6: ball reappearing after invisibility gap
    frames_since = np.nan_to_num(features[:, fi["frames_since_ball"]], nan=0)
    for f in range(1, n):
        if ball_visible[f] and not ball_visible[f - 1] and frames_since[f - 1] > FPS * 2:
            lo = max(0, f - FPS // 2)
            hi = min(n, f + FPS // 4)
            change_scores[lo:hi] += 2.5 * sensitivity
    return change_scores

def find_change_points(change_scores, min_segment_frames=None, prominence=None):
    """find possession-change boundaries from peaks in the change score signal"""
    if min_segment_frames is None:
        min_segment_frames = int(FPS * 0.8)
    all_points = []
    for _, ff, lf in HALVES:
        half_scores = change_scores[ff:lf + 1]

        # smooth the change scores a bit for peak detection
        smoothed = _smooth(half_scores, FPS // 2)

        if prominence is None:
            # use the 75th percentile of non-zero scores
            nonzero = smoothed[smoothed > 0]
            if len(nonzero) > 0:
                prom = np.percentile(nonzero, 65)
            else:
                prom = 1.0
        else:
            prom = prominence

        peaks, properties = find_peaks(
            smoothed,
            distance=min_segment_frames,
            prominence=prom,
        )

        # convert back to global frame indices
        all_points.extend((peaks + ff).tolist())

    return sorted(all_points)


# step_2: classify each segment

def classify_segments(segments, features, feature_names, ball_visible, st_labels, st_conf, dead_score):
    """classify each segment as home, away, or dead."""
    fi = {name: i for i, name in enumerate(feature_names)}
    from pipeline.smart_tagging_prior import H as ST_H, A as ST_A, D as ST_D, UNKNOWN as ST_UNK
    segment_labels = []
    segment_confidence = []
    for start, end in segments:
        n_frames = end - start
        if n_frames == 0:
            segment_labels.append(D_CODE)
            segment_confidence.append(0.0)
            continue

        sl = slice(start, end)
        ds = dead_score[sl]
        mean_dead = np.mean(ds)
        n_tracked = np.nan_to_num(features[sl, fi["n_total_tracked"]], nan=0)
        mean_tracked = np.mean(n_tracked)

        # check ST dead
        st_d_frac = np.mean(st_labels[sl] == ST_D) if n_frames > 0 else 0
        dead_evidence = 0.0
        dead_evidence += mean_dead * 3.0
        dead_evidence += st_d_frac * 4.0
        if mean_tracked < 10:
            dead_evidence += 3.0
        if mean_tracked < 5:
            dead_evidence += 5.0

        dist_ratio = features[sl, fi["dist_ratio"]]
        valid_dr = ~np.isnan(dist_ratio)

        # core signal: mean dist_ratio
        if valid_dr.sum() > 0:
            mean_dr = np.nanmean(dist_ratio)
        else:
            mean_dr = 0.5

        # smart tagging team signal
        st_h_frac = np.mean(st_labels[sl] == ST_H)
        st_a_frac = np.mean(st_labels[sl] == ST_A)

        # players within 5m
        h5 = np.nan_to_num(features[sl, fi["home_within_5m"]], nan=0)
        a5 = np.nan_to_num(features[sl, fi["away_within_5m"]], nan=0)
        mean_h5 = np.mean(h5)
        mean_a5 = np.mean(a5)

        # players within 10m
        h10 = np.nan_to_num(features[sl, fi["home_within_10m"]], nan=0)
        a10 = np.nan_to_num(features[sl, fi["away_within_10m"]], nan=0)
        mean_h10 = np.mean(h10)
        mean_a10 = np.mean(a10)

        # centroid signal
        centroid_sig = np.nan_to_num(features[sl, fi["centroid_closer_to_ball"]], nan=0)
        mean_centroid = np.mean(centroid_sig)

        # territorial advantage
        home_atk = np.nan_to_num(features[sl, fi["home_frac_attacking"]], nan=0.5)
        away_atk = np.nan_to_num(features[sl, fi["away_frac_attacking"]], nan=0.5)

        # compute H vs A score
        h_score = 0.0
        a_score = 0.0

        # dist_ratio: < 0.5 = home closer
        team_signal = 0.5 - mean_dr  # positive = home, negative = away
        h_score += team_signal * 8.0
        a_score -= team_signal * 8.0

        # Smart Tagging
        h_score += st_h_frac * 6.0
        a_score += st_a_frac * 6.0

        # players near ball
        h_score += (mean_h5 - mean_a5) * 0.8
        a_score += (mean_a5 - mean_h5) * 0.8
        h_score += (mean_h10 - mean_a10) * 0.3
        a_score += (mean_a10 - mean_h10) * 0.3

        # Centroid
        h_score += mean_centroid * 1.0
        a_score -= mean_centroid * 1.0

        # Territorial
        h_score += (np.mean(home_atk) - 0.5) * 1.5
        a_score += (np.mean(away_atk) - 0.5) * 1.5

        # Dead vs Alive
        ball_vis_frac = valid_dr.sum() / max(1, n_frames)
        is_dead = False
        if mean_tracked < 10:
            is_dead = True
        elif mean_dead >= 0.40:
            is_dead = True
        elif st_d_frac >= 0.50:
            is_dead = True
        elif mean_dead >= 0.25 and ball_vis_frac < 0.20:
            is_dead = True
        if is_dead:
            segment_labels.append(D_CODE)
            segment_confidence.append(float(max(mean_dead, st_d_frac)))
        elif h_score >= a_score:
            segment_labels.append(H)
            segment_confidence.append(min(abs(h_score - a_score) / 5.0, 1.0))
        else:
            segment_labels.append(A)
            segment_confidence.append(min(abs(a_score - h_score) / 5.0, 1.0))

    return segment_labels, segment_confidence


# step_3: segment-level post-processing

def postprocess_segments(seg_bounds, seg_labels, seg_confs, min_poss_s=0.6, min_dead_s=0.8):
    """merge adjacent teams and absorb short possessions and dead balls"""
    min_poss = int(min_poss_s * FPS)
    min_dead = int(min_dead_s * FPS)
    bounds = list(seg_bounds)
    labels = list(seg_labels)
    confs = list(seg_confs)

    for cleanup_pass in range(5):
        changed = False

        # pass_a: merge adjacent same-team segments
        i = 0
        while i < len(labels) - 1:
            if labels[i] == labels[i + 1]:
                # Merge i and i+1
                bounds[i] = (bounds[i][0], bounds[i + 1][1])
                confs[i] = max(confs[i], confs[i + 1])
                del bounds[i + 1]
                del labels[i + 1]
                del confs[i + 1]
                changed = True
            else:
                i += 1

        # pass_b: remove micro-possessions
        i = 0
        while i < len(labels):
            seg_len = bounds[i][1] - bounds[i][0]
            if labels[i] in (H, A) and seg_len < min_poss:
                if len(labels) <= 1:
                    i += 1
                    continue
                if i == 0:
                    labels[i] = labels[i + 1]
                elif i == len(labels) - 1:
                    labels[i] = labels[i - 1]
                else:
                    # pick the label of the longer neighbour
                    prev_len = bounds[i - 1][1] - bounds[i - 1][0]
                    next_len = bounds[i + 1][1] - bounds[i + 1][0]
                    if labels[i - 1] == labels[i + 1]:
                        labels[i] = labels[i - 1]
                    elif prev_len >= next_len:
                        labels[i] = labels[i - 1]
                    else:
                        labels[i] = labels[i + 1]
                changed = True
            i += 1

        # pass_c: remove micro-dead-balls between same-team
        i = 1
        while i < len(labels) - 1:
            seg_len = bounds[i][1] - bounds[i][0]
            if labels[i] == D_CODE and seg_len < min_dead:
                if labels[i - 1] == labels[i + 1] and labels[i - 1] in (H, A):
                    labels[i] = labels[i - 1]
                    changed = True
            i += 1
        if not changed:
            break

    # final merge of adjacent same-label
    i = 0
    while i < len(labels) - 1:
        if labels[i] == labels[i + 1]:
            bounds[i] = (bounds[i][0], bounds[i + 1][1])
            confs[i] = max(confs[i], confs[i + 1])
            del bounds[i + 1]
            del labels[i + 1]
            del confs[i + 1]
        else:
            i += 1
    return bounds, labels, confs

# main entry: build labels
def build_transition_labels(features, feature_names, ball_visible, ball_xy,
                            st_labels, st_conf, dead_mask, dead_score,
                            sensitivity=1.0, min_segment_s=0.4,
                            min_poss_s=0.6, min_dead_s=3.0,prominence=None):
    """build possession labels using the transition-first approach"""
    n = len(features)

    # step_1: detect transitions
    change_scores = detect_transitions(
        features, feature_names, ball_visible, ball_xy,
        st_labels, st_conf, dead_score,
        sensitivity=sensitivity
    )
    if prominence is None:
        nonzero = change_scores[change_scores > 0]
        prom = np.percentile(nonzero, 30) if len(nonzero) > 0 else 0.5
    else:
        prom = prominence

    min_seg_frames = max(4, int(min_segment_s * FPS))
    change_points = find_change_points(
        change_scores,
        min_segment_frames=min_seg_frames,
        prominence=prom
    )

    # step_2: build segments from change points
    all_seg_bounds = []
    for _, ff, lf in HALVES:
        half_cps = [cp for cp in change_points if ff < cp <= lf]
        boundaries = [ff] + half_cps + [lf + 1]
        for i in range(len(boundaries) - 1):
            all_seg_bounds.append((boundaries[i], boundaries[i + 1]))

    # step_3: classify each segment
    seg_labels, seg_confs = classify_segments(
        all_seg_bounds, features, feature_names, ball_visible,
        st_labels, st_conf, dead_score)

    # step_4: post-process at segment level
    all_seg_bounds, seg_labels, seg_confs = postprocess_segments(
        all_seg_bounds, seg_labels, seg_confs,
        min_poss_s=min_poss_s, min_dead_s=min_dead_s)

    # step_5: convert to per-frame string labels
    labels = np.full(n, "X", dtype="<U1")
    label_map = {H: "H", A: "A", D_CODE: "D"}
    for (start, end), lab in zip(all_seg_bounds, seg_labels):
        labels[start:end] = label_map[lab]
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

    # try different sensitivity / prominence combos
    configs = [
        {"sensitivity": 0.8, "min_segment_s": 0.8, "prominence": None},
        {"sensitivity": 1.0, "min_segment_s": 0.8, "prominence": None},
        {"sensitivity": 1.2, "min_segment_s": 0.8, "prominence": None},
        {"sensitivity": 1.0, "min_segment_s": 1.0, "prominence": None},
        {"sensitivity": 1.0, "min_segment_s": 1.5, "prominence": None},
    ]

    for cfg in configs:
        print(f"\nsensitivity={cfg['sensitivity']}, "
              f"min_seg={cfg['min_segment_s']}s, "
              f"prom={cfg['prominence']}")
        labels = build_transition_labels(
            feats, names, bvis, bxy,
            st_labels, st_conf, dead_mask, dead_score, **cfg)
        from pipeline.postprocess import label_stats
        label_stats(labels)

    print(f"\ndone in {time.time() - t0:.1f} s")