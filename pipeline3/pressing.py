"""Pressing-aware possession evidence (V4).

Barcelona’s high press biases nearest-player proximity toward the presser. This
module corrects that by estimating who can realistically intercept the ball,
discounting pressure on the ball, handling counter-press windows, and using
defensive-line height as context.

Calibration is self-supervised or literature-based; no ground truth is used.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1

V_MAX = 8.0  # sprint speed, m/s
TAU_REACT = 0.3  # reaction delay, s
R_CTRL = 0.7  # control radius, m
TTI_MAX = 2.0  # interception horizon, s
TTI_GRID = np.round(np.arange(0.1, TTI_MAX + 1e-9, 0.1), 3)
TTI_STRIDE = 5  # frames between calculations
K_MIN, K_MAX = 0.2, 3.0  # friction bounds, 1/s

PRESS_DIST_M = 3.0 # pressing distance in metres
PRESS_WINDOW = 15 # pressure smoothing window
CCW_SECONDS = 5.5 # counter press window in seconds

# ball friction
def calibrate_ball_friction(sig):
    """Fit the exponential decay of ball speed between visible-frame pairs.

    A rolling ball obeys v(t+dt) = v(t) * exp(-k dt); k is fit by regressing
    log-speed ratios against the time gap over consecutive visible pairs
    (teleports excluded).  Purely internal: uses only observed ball frames.
    """
    vis = sig["ball_visible"]
    spd = np.nan_to_num(sig["ball_speed"], nan=0.0)
    # use fast ball pairs to estimate friction
    deltas, lograt = [], []
    for gap in range(2, 11):
        m = vis[gap:] & vis[:-gap] & (spd[gap:] > 6.0) & (spd[:-gap] > 6.0)
        if m.sum() < 50:
            continue
        v1 = spd[:-gap][m]
        v2 = spd[gap:][m]
        # ignore accelerations
        decay = v2 < v1
        if decay.sum() < 50:
            continue
        deltas.append(np.full(int(decay.sum()), gap / FPS))
        lograt.append(np.log(v2[decay] / v1[decay]))
    if not deltas:
        return 1.0
    d = np.concatenate(deltas)
    lr = np.concatenate(lograt)
    slope = np.polyfit(d, lr, 1)[0] # negative friction
    k = float(np.clip(-slope, K_MIN, K_MAX))
    return k

# time to interception
def tti_team_scores(sig, k=None, stride=TTI_STRIDE):
    """TTI of each team's best interceptor of the ball's future path.

    For frames on `stride`, for each horizon t: ball position integrates its
    velocity under friction, players hold velocity; a player intercepts when
    within reach(t) = V_MAX * max(0, t - TAU_REACT) + R_CTRL.  The team TTI is
    the min over its players; the score tanh((TTI_A - TTI_H)/scale) in [-1, +1]
    replaces proximity in the affinity (a presser behind a moving ball never
    wins it; the receiver downfield does).
    """
    if k is None:
        k = calibrate_ball_friction(sig)
    players = sig["_players"] # player coordinates
    team = sig["_home_mask"]
    pv = sig["player_v"] # player velocity
    ball = sig["ball_xy"] * np.array([LENGTH, WIDTH]) # metres
    bvis = sig["ball_visible"]
    bv = np.nan_to_num(sig["ball_v"], nan=0.0) # ball velocity in m/s

    idx = np.arange(0, N_FRAMES, stride)
    tti_h = np.full(len(idx), TTI_MAX + 1.0)
    tti_a = np.full(len(idx), TTI_MAX + 1.0)
    home_mask_frame = team[None, :] # team mask

    for t in TTI_GRID:
        # future ball position
        s_t = (1.0 - np.exp(-k * t)) / k if k > 0 else t
        bt = ball[idx] + bv[idx] * s_t # future ball positions
        pt = players[idx] * np.array([LENGTH, WIDTH]) + pv[idx] * t # future player positions
        d = np.sqrt(((pt - bt[:, None, :]) ** 2).sum(-1)) # player distances
        reach = V_MAX * max(0.0, t - TAU_REACT) + R_CTRL
        ok = np.isfinite(d) & (d <= reach)
        ok[~bvis[idx]] = False
        if ok.any():
            ok_h = ok & home_mask_frame
            ok_a = ok & ~home_mask_frame
            newly_h = ok_h.any(1) & (tti_h >= TTI_MAX + 1.0)
            newly_a = ok_a.any(1) & (tti_a >= TTI_MAX + 1.0)
            tti_h[newly_h] = t
            tti_a[newly_a] = t

    # fill scores between sampled frames
    full_h = np.interp(np.arange(N_FRAMES), idx, tti_h)
    full_a = np.interp(np.arange(N_FRAMES), idx, tti_a)
    full_h[~bvis] = TTI_MAX
    full_a[~bvis] = TTI_MAX
    score = np.tanh((full_a - full_h) / 0.7) # positive favors home
    score[~bvis] = 0.0
    return {"tti_home": full_h, "tti_away": full_a, "tti_score": score, "ball_k": k}

# pressure and defensive line
def pressure_signals(sig):
    """Per team: how hard it presses the ball (count within 3 m, closing speed)."""
    from scipy.ndimage import uniform_filter1d

    players = sig["_players"]
    home_mask = sig["_home_mask"]
    ball = sig["ball_xy"]
    bvis = sig["ball_visible"]
    with np.errstate(invalid="ignore"):
        dx = (players[:, :, 0] - ball[:, 0:1]) * LENGTH
        dy = (players[:, :, 1] - ball[:, 1:2]) * WIDTH
        dist = np.sqrt(dx * dx + dy * dy)
    near = bvis[:, None] & np.isfinite(dist) & (dist < PRESS_DIST_M)
    press_h = uniform_filter1d((near & home_mask[None, :]).astype(np.float32), size=PRESS_WINDOW, mode="nearest").sum(1)
    press_a = uniform_filter1d((near & ~home_mask[None, :]).astype(np.float32), size=PRESS_WINDOW, mode="nearest").sum(1)

    # closing speed of the nearest player
    d_home = np.nanmin(np.where(home_mask[None, :], dist, np.nan), axis=1)
    d_away = np.nanmin(np.where(~home_mask[None, :], dist, np.nan), axis=1)
    close_h = np.zeros(N_FRAMES)
    close_a = np.zeros(N_FRAMES)
    with np.errstate(invalid="ignore"):
        close_h[2:-1] = (d_home[1:-2] - d_home[3:]) * FPS / 2.0 # positive means closing
        close_a[2:-1] = (d_away[1:-2] - d_away[3:]) * FPS / 2.0
    close_h = uniform_filter1d(np.nan_to_num(close_h), size=PRESS_WINDOW, mode="nearest")
    close_a = uniform_filter1d(np.nan_to_num(close_a), size=PRESS_WINDOW, mode="nearest")

    # defensive line position
    xs = players[:, :, 0]
    line_h = np.full(N_FRAMES, np.nan)
    line_a = np.full(N_FRAMES, np.nan)
    hx = np.where(home_mask[None, :] & np.isfinite(xs), xs, np.nan)
    ax = np.where(~home_mask[None, :] & np.isfinite(xs), xs, np.nan)
    # sort player positions
    hx_s = np.where(np.isfinite(hx), hx, 2.0)
    hx_s.sort(axis=1)
    line_h = hx_s[:, 1] # defends near x zero
    ax_s = np.where(np.isfinite(ax), ax, -1.0)
    ax_s.sort(axis=1)
    line_a = 1.0 - ax_s[:, -2] # defends near x one
    line_h = uniform_filter1d(np.clip(line_h, 0, 1), size=2 * FPS, mode="nearest")
    line_a = uniform_filter1d(np.clip(line_a, 0, 1), size=2 * FPS, mode="nearest")

    return {
        "press_home": np.clip(press_h, 0, 4) / 4.0,
        "press_away": np.clip(press_a, 0, 4) / 4.0,
        "closing_home": np.clip(close_h, 0, 5) / 5.0,
        "closing_away": np.clip(close_a, 0, 5) / 5.0,
        "defline_home": line_h,
        "defline_away": line_a,
    }

# counter press windows
def counterpress_windows(sig):
    """Boolean per team: inside CCW_SECONDS after losing a live ball.

    A turnover is a sustained flip of the smoothed raw team evidence (margin +
    carry, 2 s smoothing) while the ball is alive; the window then marks the
    pressing burst in which flips back to the losing team require carry
    evidence, not proximity.  With ~a few hundred turnovers a match and a 5.5 s
    window, a healthy detector covers tens of percent of live play.
    """
    from scipy.ndimage import uniform_filter1d

    margin = sig["margin"]
    carry_diff = sig["carry_home"] - sig["carry_away"]
    raw = np.tanh(1.2 * np.nan_to_num(margin) + 1.2 * carry_diff)
    raw_s = uniform_filter1d(raw, size=2 * FPS, mode="nearest")
    live = sig["ball_visible"] & (np.nan_to_num(sig["ball_speed"], nan=0) > 0.5)
    est = np.where(raw_s > 0.1, HOME, np.where(raw_s < -0.1, AWAY, -1))
    ccw_h = np.zeros(N_FRAMES, dtype=bool)
    ccw_a = np.zeros(N_FRAMES, dtype=bool)
    win = int(CCW_SECONDS * FPS)

    current = -1
    for f in range(N_FRAMES):
        # keep the last team through ambiguous stretches
        if est[f] in (HOME, AWAY) and est[f] != current:
            # check for a live ball near the turnover
            if current in (HOME, AWAY) and live[max(0, f - 50):f + 50].any():
                target = ccw_h if current == HOME else ccw_a
                target[f:f + win] = True
            current = est[f]
    return {"ccw_home": ccw_h, "ccw_away": ccw_a}

# run all pressing signals
def compute_pressing(sig, with_tti=True):
    """All pressing-aware extras, merged into the signals dict."""
    out = dict(sig)
    extras = pressure_signals(sig)
    extras.update(counterpress_windows(sig))
    if with_tti:
        extras.update(tti_team_scores(sig))
    out.update(extras)
    return out

if __name__ == "__main__":
    import time
    from pipeline2.data_layer import load_core, build_signals
    t0 = time.time()
    sig = build_signals(load_core())
    k = calibrate_ball_friction(sig)
    print(f"ball friction k = {k:.2f} 1/s  (halving speed in {np.log(2)/k:.1f} s)  "
          f"[{time.time()-t0:.0f}s]")
    t0 = time.time()
    tti = tti_team_scores(sig, k=k)
    print(f"TTI computed in {time.time()-t0:.0f}s: "
          f"home sooner on {100*np.mean(tti['tti_score']>0.2):.0f}% of frames, "
          f"away on {100*np.mean(tti['tti_score']<-0.2):.0f}%")
    t0 = time.time()
    pr = pressure_signals(sig)
    ccw = counterpress_windows(sig)
    print(f"pressure/lines in {time.time()-t0:.0f}s: mean press H {pr['press_home'].mean():.2f} "
          f"A {pr['press_away'].mean():.2f}; line H {np.nanmean(pr['defline_home']):.2f} "
          f"A {np.nanmean(pr['defline_away']):.2f}")
    print(f"counter-press windows cover: H {100*ccw['ccw_home'].mean():.0f}% "
          f"A {100*ccw['ccw_away'].mean():.0f}% of frames")
