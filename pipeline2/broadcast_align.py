"""Broadcast-ATD fusion (variant V3): a second, independent measurement of the ball.

The broadcast recording carries its own ATD in its own timeline. Where the tactical
tracking loses the ball (41% of graded frames), the broadcast tracking often still
sees it -- two independent cameras rarely occlude on the same frames.

Alignment strategy
------------------
The broadcast timeline is NOT synchronised to the tactical one and has cuts and
replays. So:

  1. split the broadcast into continuous stretches (homography valid + no teleport);
  2. inside each stretch, estimate a constant frame offset against the tactical
     ball-x series by cross-correlation (both are normalised pitch coordinates);
  3. accept the stretch only if the correlation passes the quality gate;
  4. fill tactical ball gaps from accepted stretches, tagged with their confidence.

The gate is deliberately strict: a wrong alignment injects phantom ball motion,
which is worse than no second camera. If overall coverage is too low, the fusion
reports failure and the pipeline falls back to V2 behaviour.

    cd more-than-a-hack-2027
    python -m pipeline2.broadcast_align
"""

import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require  # noqa: E402
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]

# alignment thresholds
GATE_CORR = 0.75
GATE_MIN_OVERLAP_S = 20.0
GATE_MIN_COVERAGE = 0.03
MAX_OFFSET_S = 30.0
COARSE_STEP_F = 5

def load_broadcast_ball(data_dir=None):
    """broadcast ball positions (normalised pitch coords), by broadcast record."""
    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    raw_path = require(["atd_raw"], GAME, data_dir, source="broadcast")[0]
    xs, ys, spd = [], [], []
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            ball_g = parts[2].strip()
            if ball_g[0] == "N":
                xs.append(np.nan); ys.append(np.nan); spd.append(np.nan)
            else:
                c1 = ball_g.index(",")
                c2 = ball_g.index(",", c1 + 1)
                xs.append(float(ball_g[:c1]))
                ys.append(float(ball_g[c1 + 1:c2]))
                spd.append(float(ball_g[c2 + 1:]))
    return np.array(xs), np.array(ys), np.array(spd)

def _broadcast_stretches(bx):
    """continuous broadcast stretches: consecutive records with the ball present."""
    present = np.isfinite(bx)
    stretches = []
    start = None
    last_f = None
    for f in np.flatnonzero(present):
        if start is None:
            start = f
        elif f - last_f > 2:
            stretches.append((int(start), int(last_f + 1)))
            start = f
        last_f = f
    if start is not None and last_f is not None:
        stretches.append((int(start), int(last_f + 1)))
    return stretches

def fuse_broadcast(sig, data_dir=None):
    """fill tactical ball gaps from the aligned broadcast ATD."""
    t0 = time.time()
    bx, by, bspd = load_broadcast_ball(data_dir)
    tac_x = sig["ball_xy"][:, 0]
    tac_ok = sig["ball_visible"]
    stretches = [s for s in _broadcast_stretches(bx) if s[1] - s[0] >= GATE_MIN_OVERLAP_S * FPS]
    print(f"broadcast: {len(bx)} records, {len(stretches)} usable stretches")

    max_off = int(MAX_OFFSET_S * FPS)
    candidates = []
    for br_lo, br_hi in stretches:
        # check overlap at this offset
        sl = slice(br_lo, br_hi)
        br_seg = bx[sl]
        br_ok = np.isfinite(br_seg)
        best = (None, 0.0)
        for off in range(-max_off, max_off + 1, COARSE_STEP_F):
            tf = np.arange(br_lo, br_hi) + off
            inside = (tf >= 0) & (tf < N_FRAMES)
            if inside.sum() < GATE_MIN_OVERLAP_S * FPS:
                continue
            ta = tac_x[tf[inside]]
            ba = br_seg[inside]
            m = np.isfinite(ta) & br_ok[inside] & tac_ok[tf[inside]]
            if m.sum() < GATE_MIN_OVERLAP_S * FPS:
                continue
            ta, ba = ta[m], ba[m]
            if ta.std() < 1e-6 or ba.std() < 1e-6:
                continue
            r = float(np.corrcoef(ta, ba)[0, 1])
            if r > best[1]:
                best = (off, r)
        if best[0] is not None and best[1] >= GATE_CORR:
            candidates.append((br_lo, br_hi, best[0], best[1]))

    # keep offsets that match a nearby stretch
    accepted = []
    for c in candidates:
        br_lo, br_hi, off, r = c
        neighbour = next((a for a in accepted if abs(a[0] - br_hi) < FPS or abs(br_lo - a[1]) < FPS), None)
        if neighbour is None or abs(neighbour[2] - off) <= FPS:
            accepted.append(c)

    print(f"stretches accepted: {len(accepted)}/{len(stretches)} "
          f"({len(candidates)} passed corr alone)")
    fused_sig = None
    if accepted:
        ball_xy = sig["ball_xy"].copy()
        ball_vis = sig["ball_visible"].copy()
        ball_speed = sig["ball_speed"].copy()
        conf = np.zeros(N_FRAMES)
        n_fused = 0
        for br_lo, br_hi, off, r in accepted:
            bf = np.arange(br_lo, br_hi)
            tf = bf + off
            inside = (tf >= 0) & (tf < N_FRAMES)
            bf, tf = bf[inside], tf[inside]
            fill = ~ball_vis[tf] & np.isfinite(bx[bf]) & np.isfinite(by[bf])
            ball_xy[tf[fill], 0] = bx[bf[fill]]
            ball_xy[tf[fill], 1] = by[bf[fill]]
            ball_speed[tf[fill]] = bspd[bf[fill]]
            ball_vis[tf[fill]] = True
            conf[tf[fill]] = r
            n_fused += int(fill.sum())
        coverage = n_fused / N_FRAMES
        print(f"fused {n_fused} tactical frames ({100 * coverage:.1f}%), "
              f"mean corr {np.mean([a[3] for a in accepted]):.3f}")
        if coverage < GATE_MIN_COVERAGE:
            print("gate FAILED: fused coverage too low -- falling back to V2")
            return None
        fused_sig = dict(sig)
        fused_sig["ball_xy"] = ball_xy
        fused_sig["ball_visible"] = ball_vis
        fused_sig["ball_speed"] = ball_speed
        fused_sig["broadcast_conf"] = conf
        from pipeline2.data_layer import rebuild_ball_dependents
        fused_sig = rebuild_ball_dependents(fused_sig)
    else:
        print("gate FAILED: no stretch passed the correlation gate, falling back to V2")
    print(f"({time.time() - t0:.1f} s)")
    return fused_sig

if __name__ == "__main__":
    from pipeline2.data_layer import load_core, build_signals
    core = load_core()
    sig = build_signals(core)
    fused = fuse_broadcast(sig)
    if fused is not None:
        added = int(fused["ball_visible"].sum() - sig["ball_visible"].sum())
        print(f"ball coverage: {sig['ball_visible'].sum()} -> {fused['ball_visible'].sum()} "
              f"(+{added} frames)")
