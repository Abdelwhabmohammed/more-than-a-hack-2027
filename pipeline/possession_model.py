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

# state codes
H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])
N_STATES = 3

# graded frame windows
HALVES = half_video_blocks(GAME)
def _build_transition_matrix():
    """log transition matrix encoding football priors"""
    # very strong self-transition to prevent fragmentation
    p_stay = 0.9995

    # small transition probabilities
    p_to_dead = 0.00025
    p_to_other_team = 0.00025

    # dead ball self-transition
    p_dead_stay = 0.995
    p_restart = 0.0025
    trans = np.array([
        # H, A, D
        [p_stay, p_to_other_team, p_to_dead], # H
        [p_to_other_team, p_stay, p_to_dead], # A
        [p_restart, p_restart, p_dead_stay], # D
    ], dtype=np.float64)

    # normalise rows
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(trans + 1e-300)

def _build_emission_scores(features, feature_names, ball_visible, interp_xy, interp_conf, st_prior_labels, st_prior_conf, dead_score):
    """per-frame log-emission scores."""
    fi = {name: i for i, name in enumerate(feature_names)}
    n = len(features)
    emissions = np.zeros((n, N_STATES), dtype=np.float64)

    # source_1: nearest player to ball
    dist_ratio = features[:, fi["dist_ratio"]]  # NaN where no ball
    valid_dr = ~np.isnan(dist_ratio)

    # dist_ratio = nearest_home / (nearest_home + nearest_away)
    # low ratio means home player is closer
    # high ratio means away player is closer
    home_score = np.zeros(n)
    away_score = np.zeros(n)
    home_score[valid_dr] = np.clip(1.0 - 2.0 * dist_ratio[valid_dr], -1, 1)
    away_score[valid_dr] = np.clip(2.0 * dist_ratio[valid_dr] - 1.0, -1, 1)
    emissions[:, H] += home_score * 3.0
    emissions[:, A] += away_score * 3.0

    # source_2: num of players near ball
    h5 = np.nan_to_num(features[:, fi["home_within_5m"]], nan=0)
    a5 = np.nan_to_num(features[:, fi["away_within_5m"]], nan=0)
    diff_5m = h5 - a5
    emissions[:, H] += np.clip(diff_5m, -3, 3) * 0.5
    emissions[:, A] += np.clip(-diff_5m, -3, 3) * 0.5

    # source_3: smart Tagging prior (vectorised)
    st_h_mask = st_prior_labels == H
    st_a_mask = st_prior_labels == A
    st_d_mask = st_prior_labels == D
    emissions[st_h_mask, H] += st_prior_conf[st_h_mask] * 4.0
    emissions[st_a_mask, A] += st_prior_conf[st_a_mask] * 4.0
    emissions[st_d_mask, D] += st_prior_conf[st_d_mask] * 4.0

    # source_4: dead ball score
    emissions[:, D] += dead_score * 5.0
    # penalise alive states when dead score is high
    alive_penalty = -dead_score * 2.5
    emissions[:, H] += alive_penalty
    emissions[:, A] += alive_penalty

    # source_5: low player count = dead or ungraded
    n_tracked = np.nan_to_num(features[:, fi["n_total_tracked"]], nan=0)
    low_players = np.clip(1.0 - n_tracked / 15.0, 0, 1)
    emissions[:, D] += low_players * 4.0

    # source_6: ball visibility = alive signal
    ball_spd = np.nan_to_num(features[:, fi["ball_speed"]], nan=0)
    alive_signal = np.zeros(n)
    alive_signal[ball_visible & (ball_spd > 1.0)] = 1.0
    alive_signal[ball_visible & (ball_spd > 3.0)] = 2.0
    emissions[:, H] += alive_signal * 0.3
    emissions[:, A] += alive_signal * 0.3
    emissions[:, D] -= alive_signal * 0.8

    # source_7: team centroid closer to ball
    centroid_signal = np.nan_to_num(features[:, fi["centroid_closer_to_ball"]], nan=0)
    emissions[:, H] += centroid_signal * 0.5   # +1 when home closer
    emissions[:, A] -= centroid_signal * 0.5

    # source_8: territorial features
    home_atk = np.nan_to_num(features[:, fi["home_frac_attacking"]], nan=0.5)
    away_atk = np.nan_to_num(features[:, fi["away_frac_attacking"]], nan=0.5)
    # more players in attacking half means more likely to have possession
    emissions[:, H] += (home_atk - 0.5) * 0.8
    emissions[:, A] += (away_atk - 0.5) * 0.8

    return emissions


def viterbi(log_trans, log_emissions, log_prior=None):
    """viterbi algorithm for the most likely state sequence."""
    T, K = log_emissions.shape
    if log_prior is None:
        log_prior = np.log(np.ones(K) / K)

    # viterbi tables
    V = np.zeros((T, K), dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int32)
    V[0] = log_prior + log_emissions[0]
    for t in range(1, T):
        for k in range(K):
            scores = V[t - 1] + log_trans[:, k]
            back[t, k] = np.argmax(scores)
            V[t, k] = scores[back[t, k]] + log_emissions[t, k]

    # backtrack
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def run_hmm(features, feature_names, ball_visible,
            interp_xy, interp_conf,
            st_prior_labels, st_prior_conf,
            dead_score):
    """run the full HMM pipeline and return per-frame labels."""
    n = len(features)
    log_trans = _build_transition_matrix()
    print("building emission scores")
    emissions = _build_emission_scores(
        features, feature_names, ball_visible,
        interp_xy, interp_conf,
        st_prior_labels, st_prior_conf,
        dead_score
    )

    # run Viterbi on each half separately (different initial priors)
    raw_states = np.full(n, D, dtype=np.int32)  # default: dead
    for half_num, first_frame, last_frame in HALVES:
        print(f"  Running Viterbi on half {half_num} "
              f"(frames {first_frame}–{last_frame})")
        half_emissions = emissions[first_frame:last_frame + 1]

        # initial state: D
        log_prior = np.log(np.array([0.05, 0.05, 0.90]))
        path = viterbi(log_trans, half_emissions, log_prior)
        raw_states[first_frame:last_frame + 1] = path

    # convert to string labels
    labels = np.full(n, "X", dtype="<U1")
    for half_num, first_frame, last_frame in HALVES:
        for f in range(first_frame, last_frame + 1):
            labels[f] = LABELS[raw_states[f]]
    return labels, raw_states


if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.environ.get("MTAH_DATA_DIR") or DATA_DIR

    print("loading features")
    from pipeline.features import extract_features
    feats, names, bvis, bxy = extract_features(data_dir=data_dir)

    print("interpolating ball")
    from pipeline.dead_ball_and_interp import interpolate_ball, detect_dead_ball
    ixy, iconf = interpolate_ball(bxy, bvis)

    print("building smart tagging prior")
    from pipeline.smart_tagging_prior import build_smart_tagging_prior
    st_labels, st_conf, _, _ = build_smart_tagging_prior(data_dir=data_dir)

    print("detecting dead ball")
    dead = detect_dead_ball(feats, bvis, bxy, ixy, iconf)

    print("\nrunning HMM")
    labels, states = run_hmm(feats, names, bvis, ixy, iconf, st_labels, st_conf, dead)

    for label in ["H", "A", "D", "X"]:
        n = int(np.sum(labels == label))
        print(f"  {label}: {n:>8d} frames ({100*n/N_FRAMES:.1f}%)")

    # count possessions
    n_poss = 0
    prev = "X"
    for half_num, ff, lf in HALVES:
        for f in range(ff, lf + 1):
            if labels[f] in ("H", "A") and labels[f] != prev:
                if prev in ("D", "X") or (prev in ("H", "A") and prev != labels[f]):
                    n_poss += 1
            prev = labels[f]
    print(f"\n  Raw possessions: {n_poss}")
    print(f"  Done in {time.time() - t0:.1f} s")
