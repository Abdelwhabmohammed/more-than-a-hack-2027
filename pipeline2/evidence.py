"""Per-frame evidence: team affinity s(f) in [-1, +1] and dead-ball probability p_D(f).

Combination weights are fixed football-prior constants, documented inline. Nothing
here is fitted to the public ground truth; calibration diagnostics (self-supervised
ball masking) live in experiments.py.

    cd more-than-a-hack-2027
    python -m pipeline2.evidence
"""

import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]

# evidence weights
W_MARGIN = 1.2 # player distance
W_CARRY = 1.2 # ball carry
W_ST_PRIOR = 0.8 # smart tagging when geometry is unclear
AFFINITY_SMOOTH = 19 # smoothing window (~0.75 s)

# dead-ball evidence scores
SC_OOB = 3.0  # ball out of bounds
SC_SETPIECE_DATA = 2.5
SC_SETPIECE_PRIOR = 1.8
SC_FREEZE = 1.6 # players at stoppage pace
# longer gaps add stoppage evidence
SC_LONG_GAP = 2.0
SC_LONG_GAP_MAX = 4.0
GAP_EVIDENCE_AFTER_S = 2.0
GAP_FULL_S = 5.0
MOTION_GATE_FAST = 3.0 # player speed threshold (m/s)
MOTION_GATE_DIV = 2.0 # speed range for the motion gate
FREEZE_SMOOTH_S = 2.0 # player speed smoothing window
SC_HALTED_BALL = 1.0 # stopped ball and players
SC_EMPTY_PITCH = 4.0 # few tracked objects
SC_SOMEWHAT_EMPTY = 1.0
DEAD_BIAS = -2.0 # dead-ball probability baseline

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# team affinity
def build_affinity(sig, smart_tagging_prior=None, use_carry=True, use_st=True):
    """Fused team-affinity score s(f) in [-1, +1] (+ = home)"""
    n = N_FRAMES
    margin = sig["margin"].copy()

    # carry the last margin through gaps
    obs = sig["has_ball_obs"] | sig["margin_filled_mask"]
    filled = np.zeros(n, dtype=bool)
    last = 0.0
    decay = np.zeros(n)
    miss_run = 0
    for f in range(n):
        if obs[f]:
            last = margin[f]
            miss_run = 0
            filled[f] = True
            margin[f] = last
        else:
            miss_run += 1
            decay[f] = np.exp(-miss_run / (4.0 * FPS))
            margin[f] = last * decay[f]
    s = np.zeros(n)
    s += W_MARGIN * np.clip(margin, -1.0, 1.0)
    if use_carry:
        carry_diff = sig["carry_home"] - sig["carry_away"]
        s += W_CARRY * np.clip(carry_diff, -1.0, 1.0)
    if use_st and smart_tagging_prior is not None:
        st_labels, st_conf = smart_tagging_prior
        # use team tags when ball evidence is weak
        quiet = ~obs | (np.abs(np.clip(margin, -1, 1)) < 0.15)
        st = np.zeros(n)
        st[(st_labels == 0) & quiet] = +st_conf[(st_labels == 0) & quiet]
        st[(st_labels == 1) & quiet] = -st_conf[(st_labels == 1) & quiet]
        s += W_ST_PRIOR * np.clip(st, -1.0, 1.0)
    s = np.tanh(s)
    # remove outliers and smooth
    s = median_filter(s, size=5, mode="nearest")
    s = uniform_filter1d(s, size=AFFINITY_SMOOTH, mode="nearest")
    return s

# dead-ball evidence
def build_dead_evidence(sig):
    """p_D(f) = sigmoid(sum of evidence scores + bias)."""
    n = N_FRAMES
    score = np.zeros(n, dtype=np.float64)
    # ball out of bounds
    score += SC_OOB * sig["oob"]
    # set-piece windows
    for w in sig["setpiece_windows"]:
        lo, hi = max(0, w["start"]), min(n, w["end"])
        if hi > lo:
            score[lo:hi] += SC_SETPIECE_DATA if w["source"] == "data" else SC_SETPIECE_PRIOR

    # players at stoppage pace
    score += SC_FREEZE * sig["freeze"]

    # use long gaps when players are moving slowly
    since = sig["frames_since_ball"].astype(np.float64)
    gap_sc = np.clip((since - GAP_EVIDENCE_AFTER_S * FPS) /
                     ((GAP_FULL_S - GAP_EVIDENCE_AFTER_S) * FPS), 0.0, 1.0)
    gap_score = SC_LONG_GAP + (SC_LONG_GAP_MAX - SC_LONG_GAP) * gap_sc
    speed_s = uniform_filter1d(np.nan_to_num(sig["mean_player_speed"], nan=0.0), size=int(FREEZE_SMOOTH_S * FPS), mode="nearest")
    motion_gate = np.clip((MOTION_GATE_FAST - speed_s) / MOTION_GATE_DIV, 0.35, 1.0)
    score += gap_score * (~sig["ball_visible"]) * (since > GAP_EVIDENCE_AFTER_S * FPS) \
        * motion_gate

    # stopped ball and players
    ball_slow = np.nan_to_num(sig["ball_speed"], nan=0.0) < 0.4
    score += SC_HALTED_BALL * ball_slow * sig["ball_visible"] * sig["freeze"]

    # few tracked players
    tracked = sig["n_tracked"].astype(np.float64)
    score += np.where(tracked < 10, SC_EMPTY_PITCH, np.where(tracked < 15, SC_SOMEWHAT_EMPTY, 0.0))

    p_d = _sigmoid(score + DEAD_BIAS)
    # smooth single-frame blips
    p_d = uniform_filter1d(p_d, size=9, mode="nearest")
    return np.clip(p_d, 0.0, 1.0)

# build signals for V1
def signals_from_v1_features(features, feature_names, ball_visible, ball_xy):
    """minimal signals dict from pipeline.features output, for variant V1"""
    fi = {name: i for i, name in enumerate(feature_names)}
    n = len(features)
    sig = {
        "ball_visible": ball_visible,
        "ball_xy": ball_xy,
        "ball_speed": features[:, fi["ball_speed"]],
        "margin": np.zeros(n),
        "has_ball_obs": ~np.isnan(features[:, fi["dist_ratio"]]),
        "margin_filled_mask": np.zeros(n, dtype=bool),
        "carry_home": np.nan_to_num(features[:, fi["home_within_5m"]]) * 0.0,
        "carry_away": np.nan_to_num(features[:, fi["away_within_5m"]]) * 0.0,
        "oob": np.zeros(n, dtype=bool),
        "freeze": np.zeros(n),
        "frames_since_ball": np.nan_to_num(features[:, fi["frames_since_ball"]], nan=0.0),
        "n_tracked": np.nan_to_num(features[:, fi["n_total_tracked"]], nan=0.0),
        "mean_player_speed": np.nan_to_num(
            (features[:, fi["home_mean_speed"]] + features[:, fi["away_mean_speed"]]) / 2, nan=0.0),
        "ball_speed_est": features[:, fi["ball_speed"]],
    }
    dr = features[:, fi["dist_ratio"]]
    m = 1.0 - 2.0 * dr # positive favors home
    sig["margin"] = np.where(np.isnan(m), 0.0, np.clip(m, -1, 1))
    # out-of-bounds from visible positions
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    sig["oob"] = ball_visible & ((bx < -0.005) | (bx > 1.005) | (by < -0.005) | (by > 1.005))
    # estimate stoppage pace from team speeds
    hs = np.nan_to_num(features[:, fi["home_mean_speed"]], nan=0.0)
    aws = np.nan_to_num(features[:, fi["away_mean_speed"]], nan=0.0)
    comb = uniform_filter1d((hs + aws) / 2, size=int(2 * FPS), mode="nearest")
    sig["freeze"] = np.clip((1.0 - comb / 1.0), 0.0, 1.0)
    # find set-piece windows
    from pipeline2.data_layer import build_setpiece_windows
    sig["setpiece_windows"] = build_setpiece_windows(None, None, sig)
    return sig

if __name__ == "__main__":
    import time
    t0 = time.time()
    from pipeline2.data_layer import load_core, build_signals
    core = load_core()
    sig = build_signals(core)
    print(f"signals loaded in {time.time() - t0:.1f} s")

    from pipeline.smart_tagging_prior import build_smart_tagging_prior
    st_labels, st_conf, _, _ = build_smart_tagging_prior()
    s = build_affinity(sig, smart_tagging_prior=(st_labels, st_conf))
    p_d = build_dead_evidence(sig)
    print(f"affinity: mean {s.mean():+.3f}  |s|>0.3 on {100 * np.mean(np.abs(s) > 0.3):.1f}% of frames")
    print(f"p_D: mean {p_d.mean():.3f}  >0.5 on {100 * np.mean(p_d > 0.5):.1f}% of frames")
    print(f"took {time.time() - t0:.1f} s total")