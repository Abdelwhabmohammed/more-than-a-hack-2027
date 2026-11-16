"""principled segmentation: replaces attempt-1's hand-weighted peak hunting"""

import os
import sys
from collections import deque
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
# frame ranges for each half
HALVES = half_video_blocks(GAME)
H, A, D = 0, 1, 2
N_STATES = 3
# segmentation priors
K_S = 5.0 # affinity sharpness
EPS_P = 0.02 # minimum state probability
LAM = 10.0 # default boundary cost
# direct team switches cost more than switches through D
LAM_TT = 60.0 # direct H/A switch cost
LAM_TD = 30.0 # switch through D cost
MIN_DWELL = {H: 15, A: 15, D: 75} # minimum duration in frames
MAX_SEG = int(45 * FPS) # maximum segment duration
SWITCH_COST = 3.0    # viterbi switch cost

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def emissions(s, p_d):
    """Per-frame negative log-likelihoods, shape (3, N): rows H, A, D"""
    ph = _sigmoid(K_S * s)
    pa = _sigmoid(-K_S * s)
    alive_mass = np.clip(1.0 - p_d, EPS_P, 1.0)
    post = np.vstack([ph * alive_mass, pa * alive_mass, p_d])
    post = post / post.sum(axis=0, keepdims=True)
    return -np.log(post + EPS_P)

# dynamic-programming segmenter
def dp_partition(neglog, first, last, lam=None, min_dwell=None, max_seg=MAX_SEG):
    """optimal piecewise-constant labelling of [first, last] (inclusive)"""
    if min_dwell is None:
        min_dwell = MIN_DWELL
    if lam is None:
        lam = np.full((N_STATES, N_STATES), LAM)
        np.fill_diagonal(lam, 0.0)
    elif np.isscalar(lam):
        lam = np.full((N_STATES, N_STATES), float(lam))
        np.fill_diagonal(lam, 0.0)
    else:
        lam = np.asarray(lam, dtype=np.float64)
    n_win = last - first + 1
    prefix = np.zeros((N_STATES, n_win + 1))
    for k in range(N_STATES):
        prefix[k, 1:] = np.cumsum(neglog[k, first:last + 1])

    INF = np.inf
    # best cost for each label and prefix
    cost = np.full((N_STATES, n_win + 1), INF)
    back_i = np.full((N_STATES, n_win + 1), -1, dtype=np.int32)
    back_l = np.full((N_STATES, n_win + 1), -1, dtype=np.int8)
    for k in range(N_STATES):
        cost[k, 0] = 0.0

    # candidate starts for each state
    deques = [deque() for _ in range(N_STATES)]
    for j in range(1, n_win + 1):
        # add valid starts and keep the cheapest first
        for k in range(N_STATES):
            i_new = j - min_dwell[k]
            if i_new >= 0:
                best_prev = INF
                best_prev_k = -1
                for kp in range(N_STATES):
                    c = cost[kp, i_new] + lam[kp, k]
                    if c < best_prev:
                        best_prev, best_prev_k = c, kp
                if best_prev < INF:
                    v_new = best_prev - prefix[k, i_new]
                    dq = deques[k]
                    while dq and dq[-1][1] >= v_new:
                        dq.pop()
                    dq.append((i_new, v_new, best_prev_k))
                # remove starts that exceed the maximum duration
                i_old = j - max_seg - 1
                while deques[k] and deques[k][0][0] <= i_old:
                    deques[k].popleft()
        best_v, best_i, best_k, best_pk = INF, -1, -1, -1
        for k in range(N_STATES):
            if deques[k]:
                i, v, pk = deques[k][0]
                v_total = v + prefix[k, j]
                if v_total < best_v:
                    best_v, best_i, best_k, best_pk = v_total, i, k, pk
        for k in range(N_STATES):
            cost[k, j] = best_v if k == best_k else INF
        back_i[best_k, j] = best_i
        back_l[best_k, j] = best_pk

    # trace the best state sequence
    final_k = int(np.argmin(cost[:, n_win]))
    segs = []
    j, k = n_win, final_k
    while j > 0:
        i, kp = int(back_i[k, j]), int(back_l[k, j])
        segs.append((first + i, first + j, k))
        j, k = i, kp
    segs.reverse()
    return segs

# viterbi comparator
def viterbi_decode(neglog, first, last, switch_cost=SWITCH_COST):
    """first-order 3-state Viterbi over [first, last]. Returns (start, stop, label) segments"""
    n_win = last - first + 1
    delta = neglog[:, first].copy()
    back = np.zeros((n_win, N_STATES), dtype=np.int8)
    trans = np.zeros((N_STATES, N_STATES))
    for k in range(N_STATES):
        for m in range(N_STATES):
            trans[k, m] = 0.0 if k == m else switch_cost
    for t in range(1, n_win):
        cand = delta[:, None] + trans          # transition costs
        best_from = np.argmin(cand, axis=0)
        delta = cand[best_from, np.arange(N_STATES)] + neglog[:, first + t]
        back[t] = best_from
    states = np.zeros(n_win, dtype=np.int8)
    states[-1] = int(np.argmin(delta))
    for t in range(n_win - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    segs = []
    start = 0
    for t in range(1, n_win + 1):
        if t == n_win or states[t] != states[t - 1]:
            segs.append((first + start, first + t, int(states[start])))
            start = t
    return segs

# label both halves
def segment_match(s, p_d, method="dp", lam=LAM, min_dwell=None):
    """Segment both halves. Returns a list of (start, stop, label) over the full frame range"""
    neglog = emissions(s, p_d)
    if method == "dp":
        lam_matrix = np.full((N_STATES, N_STATES), float(lam))
        np.fill_diagonal(lam_matrix, 0.0)
    segs = []
    for _, ff, lf in HALVES:
        half_segs = (dp_partition(neglog, ff, lf, lam=lam_matrix, min_dwell=min_dwell)
                     if method == "dp" else viterbi_decode(neglog, ff, lf))
        segs.extend(half_segs)
    return segs

def segments_to_labels(segs, n=N_FRAMES):
    labels = np.full(n, "X", dtype="<U1")
    for start, stop, lab in segs:
        labels[start:stop] = ("H", "A", "D")[lab]
    return labels

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="demo on public-window-like slice")
    args = parser.parse_args()
    t0 = time.time()
    from pipeline2.data_layer import load_core, build_signals
    from pipeline2.evidence import build_affinity, build_dead_evidence
    from pipeline.smart_tagging_prior import build_smart_tagging_prior

    core = load_core()
    sig = build_signals(core)
    st_labels, st_conf, _, _ = build_smart_tagging_prior()
    s = build_affinity(sig, smart_tagging_prior=(st_labels, st_conf))
    p_d = build_dead_evidence(sig)
    print(f"evidence ready in {time.time() - t0:.1f} s")

    for method in ("dp", "viterbi"):
        t1 = time.time()
        segs = segment_match(s, p_d, method=method)
        labels = segments_to_labels(segs)
        runs = [(a, b, l) for (a, b, l) in segs]
        n_poss = sum(1 for _, _, l in segs if l in (H, A))
        n_dead = sum(1 for _, _, l in segs if l == D)
        print(f"\n[{method}] {len(segs)} segments in {time.time() - t1:.1f} s: "
              f"{n_poss} possessions, {n_dead} dead stretches")
        dur = [(b - a) / FPS for a, b, l in segs if l in (H, A)]
        if dur:
            print(f"  possession duration: mean {np.mean(dur):.1f} s  "
                  f"median {np.median(dur):.1f} s  min {np.min(dur):.1f}  max {np.max(dur):.1f}")
        dd = [(b - a) / FPS for a, b, l in segs if l == D]
        if dd:
            print(f"  dead duration:        mean {np.mean(dd):.1f} s  "
                  f"median {np.median(dd):.1f} s  min {np.min(dd):.1f}  max {np.max(dd):.1f}")
