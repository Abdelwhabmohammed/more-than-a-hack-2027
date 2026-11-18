"""V4 evidence: pressing-aware team affinity.

Lives in pipeline3 because it belongs to the pressing experiment only; the shared
infrastructure (data layer, DP segmenter, post-processing) stays in pipeline2 and
is imported from there.

    cd more-than-a-hack-2027
    python -m pipeline3.evidence
"""

import os
import sys
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES
from pipeline2.evidence import (
    W_MARGIN, W_CARRY, W_ST_PRIOR, AFFINITY_SMOOTH,
    build_affinity,
)

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]

# v4 priors
PRESS_DISCOUNT = 0.45 # pressure discount
TTI_VETO = 0.5 # interception veto
CCW_EVIDENCE_CUT = 0.5 # evidence cut without a carry
CARRY_GATE = 0.2 # carry threshold for the counter press gate

def _persist(margin, obs, n, decay_s=4.0):
    """Forward-fill observed margin through gaps, decaying over ~4 s."""
    out = np.zeros(n)
    last = 0.0
    miss = 0
    for f in range(n):
        if obs[f]:
            last = margin[f]
            miss = 0
            out[f] = last
        else:
            miss += 1
            out[f] = last * np.exp(-miss / (decay_s * FPS))
    return out

def build_affinity_v4(sig, smart_tagging_prior=None, tti_veto=TTI_VETO):
    """Pressing-aware affinity (variant V4) -- build_affinity plus pressing fixes.

    The raw nearest-player margin measures the *presser* as much as the
    possessor, and Barcelona press high with a man-oriented system, so during
    Athletic possessions the margin drifts home.  Three corrections, all with
    weights stated as priors (ablated in development: the TTI score as an
    additive vote fragmented the match, so it enters only as a veto):

      pressure discount -- margin evidence toward a team is discounted while
                           that team is the one swarming the ball.
      counter-press gate-- inside 5.5 s after a team lost a live ball, evidence
                           toward that team is halved unless it shows a carry
                           (the six-second rule: convergence is not control).
      TTI veto          -- where the interception-race score firmly disagrees
                           with the margin (margin says H, the ball's future
                           path is reachable first by Athletic), the margin is
                           damped proportionally.  A presser behind a moving
                           ball never wins it; the intended receiver does.
    """
    base = build_affinity(sig, smart_tagging_prior=smart_tagging_prior, use_carry=True, use_st=True)
    n = len(base)
    press_h = np.nan_to_num(sig.get("press_home", np.zeros(n)))
    press_a = np.nan_to_num(sig.get("press_away", np.zeros(n)))
    ccw_h = sig.get("ccw_home", np.zeros(n, dtype=bool))
    ccw_a = sig.get("ccw_away", np.zeros(n, dtype=bool))

    # rebuild the proximity margin
    obs = sig["has_ball_obs"] | sig["margin_filled_mask"]
    m = _persist(np.clip(np.nan_to_num(sig["margin"], nan=0.0), -1, 1), obs, n)

    # discount evidence for the pressing team
    press_dir = np.where(m > 0, press_h, 0.0) - np.where(m < 0, press_a, 0.0)
    m = m * (1.0 - PRESS_DISCOUNT * np.abs(press_dir))

    # use interception time to check the margin
    if tti_veto > 0 and "tti_score" in sig:
        tti = np.nan_to_num(sig["tti_score"], nan=0.0)
        disagree = (np.sign(m) != np.sign(tti)) & (np.abs(tti) > 0.3) & (np.abs(m) > 0.1)
        m = m * (1.0 - tti_veto * disagree * np.minimum(np.abs(tti), 1.0))
    carry_diff = np.clip(sig["carry_home"] - sig["carry_away"], -1, 1)
    s = W_MARGIN * m + W_CARRY * carry_diff
    if smart_tagging_prior is not None:
        st_labels, st_conf = smart_tagging_prior
        quiet = ~obs | (np.abs(m) < 0.15)
        st = np.zeros(n)
        st[(st_labels == 0) & quiet] = +st_conf[(st_labels == 0) & quiet]
        st[(st_labels == 1) & quiet] = -st_conf[(st_labels == 1) & quiet]
        s += W_ST_PRIOR * np.clip(st, -1.0, 1.0)
    s = np.tanh(s)

    # require a carry during counter pressing
    weak_home_carry = np.nan_to_num(sig["carry_home"], nan=0.0) < CARRY_GATE
    weak_away_carry = np.nan_to_num(sig["carry_away"], nan=0.0) < CARRY_GATE
    s[ccw_h & (s > 0) & weak_home_carry] *= CCW_EVIDENCE_CUT
    s[ccw_a & (s < 0) & weak_away_carry] *= CCW_EVIDENCE_CUT
    s = median_filter(s, size=5, mode="nearest")
    s = uniform_filter1d(s, size=AFFINITY_SMOOTH, mode="nearest")
    return s

if __name__ == "__main__":
    import time
    t0 = time.time()
    from pipeline2.data_layer import load_core, build_signals
    from pipeline3.pressing import compute_pressing
    from pipeline.smart_tagging_prior import build_smart_tagging_prior

    sig = build_signals(load_core())
    sig = compute_pressing(sig)
    st_labels, st_conf, _, _ = build_smart_tagging_prior()
    s = build_affinity_v4(sig, smart_tagging_prior=(st_labels, st_conf))
    print(f"v4 affinity in {time.time() - t0:.1f} s: mean {s.mean():+.3f}, "
          f"|s|>0.3 on {100 * np.mean(np.abs(s) > 0.3):.1f}% of frames")
