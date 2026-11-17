"""Degrade the clean Bundesliga tracking until it looks like the challenge's.

The DFL tracking is provider-corrected: every player every frame, the ball always
there.  The challenge tracking is automatic: players drop out, and the ball is
missing on 41% of graded frames, mostly in long stretches.  A model tuned on
clean data would expect a ball that always exists, so every training match is
passed through this module first, with fixed seeds so runs are reproducible.

Only published facts and the challenge *input* shape the distortion -- never its
ground truth or video:

  independent   ball gaps fall anywhere, whatever the state of the game
  stoppage      most dead-ball stretches lose the ball, the rest of the 41% anywhere
  matched_speed gaps start more often at low player speed, until the missing share
                in every speed bin and the gap lengths match the challenge's
  matched_dead  like matched_speed, and a dead-ball stretch loses the ball as often
                as the challenge loses it while the players stand still

On top of the gaps, two models of how the *seen* ball misbehaves, both fitted to
challenge-input statistics only:

  ball drift    the challenge ball is smooth (median acceleration 6 m/s^2) yet never
                quite still (slowest decile still 1 m/s) -- the signature of a
                smoothed position with a slowly drifting error.  Modelled as AR(1)
                noise per axis (spread sigma, time constant tau) followed by a
                centred moving average, (window, sigma, tau) chosen on a grid.
  dead ball     the DFL ball freezes exactly where it went out and jumps to the
                restart spot -- a corrected-data convention, not what a camera
                sees.  Every stoppage is re-enacted: the ball rolls out with its
                last velocity, disappears (or is carried to the restart spot),
                lies at the spot for a while before the restart, and stays unseen
                briefly after it while the thrower moves away.

The fitted numbers live in JSON caches under pipeline4/_cache/ and are recomputed
from the local data by anyone holding it.
"""

import itertools
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in (HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if sub not in sys.path:
        sys.path.insert(0, sub)

FPS = 25
BALL_MISSING = 0.41  # share of graded frames with no ball
PLAYER_VISIBLE = 20.5 / 22  # mean tracked objects on graded frames
PLAYER_GAP_MEAN_S = 4.0
GAP_MEDIAN_S = 10.0  # median length of ball gaps
GAP_SIGMA = 1.0
GAP_MIN_S, GAP_MAX_S = 0.2, 240.0
STOPPAGE_HIDE = 0.8  # chance a stoppage hides the ball
STOPPAGE_PAD_S = 2.0
IID_NOISE_M = 0.3  # fallback ball position noise
SCENARIOS = ("independent", "stoppage")
MATCHED = ("matched_speed", "matched_dead")
EDGE_WANDER_S = 1.0  # gap edges can shift around stoppages
CALIBRATION_STEPS = 10
NOISE_GRID_M = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
FIT_MATCH = "J03WMX"  # match used to fit the simulator

CACHE = os.environ.get("LP4_CACHE", os.path.join(HERE, "_cache"))
DFL_DIR = os.environ.get("LP4_DFL_DATA", os.path.join(CACHE, "external_data", "dfl"))
PROFILE_JSON = os.path.join(CACHE, "challenge_profile.json")
DRIFT_JSON = os.path.join(CACHE, "ball_drift.json")
DEAD_JSON = os.path.join(CACHE, "dead_ball.json")
IID_JSON = os.path.join(CACHE, "iid_noise.json")

SPEED_BIN_EDGES = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0])  # speed bins in meters per second
STANDING_MPS = 0.8  # speed threshold for standing players

def run_edges(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# challenge input profile

def speed_bin(speed_2s):
    """Bin index of the 2 s mean player speed; NaN counts as slowest."""
    return np.digitize(np.nan_to_num(speed_2s, nan=0.0), SPEED_BIN_EDGES)

def speed_2s(ps):
    """Mean player speed over a centred 2 s window, as features.base_signals has it."""
    ok = np.isfinite(ps)
    count = ok.sum(axis=1)
    mean = np.where(count > 0, np.where(ok, ps, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    return pd.Series(mean).rolling(2 * FPS, center=True, min_periods=1).mean().values

def seen_ball_speed(bx, by, length, width):
    """Ball speed as the features see it (median of 5), on frames with a ball."""
    step = np.hypot(np.diff(bx.astype(np.float64)) * length,
                    np.diff(by.astype(np.float64)) * width) * FPS
    speed = pd.Series(np.concatenate(([np.nan], step))).rolling(
        5, center=True, min_periods=1).median().values
    return speed[np.isfinite(bx)]

def profile(rebuild=False, data_dir=None):
    """How the challenge tracking behaves, measured on its input only:
    miss rate per speed bin, gap lengths and their start bins, ball-speed
    percentiles, and the miss rate while the players stand still."""
    if not rebuild and os.path.isfile(PROFILE_JSON):
        with open(PROFILE_JSON, encoding="utf-8") as handle:
            return json.load(handle)
    import pipeline4.features as feats
    table = feats.challenge_table(data_dir=data_dir)
    graded = table["half"].values > 0
    seen = table["ball_seen"].values
    bins = speed_bin(table["speed_mean_2s"].values)
    miss_rate = np.array([(~seen[graded & (bins == k)]).mean()
                           if (graded & (bins == k)).any() else np.nan
                           for k in range(len(SPEED_BIN_EDGES) + 1)])
    starts, ends = run_edges(~seen & graded)
    speed = table["ball_speed"].values[graded & seen]
    standing = graded & (np.nan_to_num(table["speed_mean_2s"].values, nan=0.0) < STANDING_MPS)
    out = {"miss_rate": miss_rate.tolist(),
           "gap_frames": (ends - starts).astype(int).tolist(),
           "gap_start_bin": bins[starts].astype(int).tolist(),
           "ball_speed": np.nanpercentile(speed, [10, 50, 90]).tolist(),
           "standing_miss": float((~seen[standing]).mean())}
    os.makedirs(CACHE, exist_ok=True)
    with open(PROFILE_JSON, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    return out


# gap generation

def alternating_mask(rng, n, mean_visible, draw_gap):
    """True/False mask: visible stretches (exponential) alternating with gaps."""
    missing = np.zeros(n, dtype=bool)
    t = int(rng.exponential(mean_visible))
    while t < n:
        length = draw_gap()
        missing[t:t + length] = True
        t += length + max(1, int(rng.exponential(mean_visible)))
    return missing

def draw_lognormal_gap(rng):
    seconds = np.clip(rng.lognormal(np.log(GAP_MEDIAN_S), GAP_SIGMA), GAP_MIN_S, GAP_MAX_S)
    return max(1, int(round(seconds * FPS)))

def fill_to_share(rng, n, graded, base=None, target=BALL_MISSING):
    """Add independent gaps to `base` until about `target` of graded frames miss."""
    base = np.zeros(n, dtype=bool) if base is None else base
    have = base[graded].mean()
    if have >= target:
        return base
    need = (target - have) / (1 - have)
    lo, hi = 1.0, 1e7
    for _ in range(25):  # find the visible stretch length for the target share
        mid = np.sqrt(lo * hi)
        state = rng.get_state()
        extra = alternating_mask(rng, n, mid, lambda: draw_lognormal_gap(rng))
        rng.set_state(state)
        added = (extra & ~base)[graded].sum() / max((~base)[graded].sum(), 1)
        if added > need:
            lo = mid
        else:
            hi = mid
    extra = alternating_mask(rng, n, np.sqrt(lo * hi), lambda: draw_lognormal_gap(rng))
    return base | extra

def hazard_fill(rng, bins, graded, base, target, lengths):
    """Add gaps so each speed bin misses about its `target` share.

    In each bin a gap starts with its own hazard and its length is drawn from the
    challenge gaps that started in that bin; the hazards are re-calibrated until
    every bin matches (same random draws on every iteration, via a saved state)."""
    n_bins = len(target)

    def share(mask):
        return np.array([mask[graded & (bins == k)].mean() if (graded & (bins == k)).any() else 0.0
                         for k in range(n_bins)])

    def odds(x):
        x = np.clip(x, 1e-3, 0.999)
        return x / (1 - x)

    have = share(base)
    mean_len = np.array([np.mean(lengths[k]) for k in range(n_bins)])
    need = np.clip((target - have) / np.maximum(1 - have, 1e-6), 0, 0.98)
    hazard = need / (1 - need) / mean_len
    state = rng.get_state()
    for step in range(CALIBRATION_STEPS + 1):
        rng.set_state(state)
        u = rng.rand(len(bins))
        pick = rng.rand(len(bins))
        missing = base.copy()
        h = hazard[bins]
        t, n = 0, len(bins)
        while t < n:
            if graded[t] and not missing[t] and u[t] < h[t]:
                choices = lengths[bins[t]]
                length = int(choices[int(pick[t] * len(choices))])
                missing[t:t + length] = True
                t += length
            else:
                t += 1
        missing &= graded
        if step == CALIBRATION_STEPS:
            return missing
        hazard *= np.clip(odds(target) / odds(share(missing)), 0.25, 4.0)
        hazard[target <= have] = 0.0


# ball position noise

def iid_noise_m():
    """The iid-noise level (m) that reproduces the challenge ball-speed percentiles."""
    if os.path.isfile(IID_JSON):
        with open(IID_JSON, encoding="utf-8") as handle:
            return json.load(handle)["sigma_m"]
    import pipeline4.dfl_data as dfl_data
    target = np.array(profile()["ball_speed"])
    data = dfl_data.load(FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    rng = np.random.RandomState(0)
    errors = []
    for sigma in NOISE_GRID_M:
        bx = data["bx"] + rng.normal(0, sigma / length, len(data["bx"]))
        by = data["by"] + rng.normal(0, sigma / width, len(data["by"]))
        got = np.nanpercentile(seen_ball_speed(bx, by, length, width), [10, 50, 90])
        errors.append(float(np.sum(np.log((got + 0.1) / (target + 0.1)) ** 2)))
    best = NOISE_GRID_M[int(np.argmin(errors))]
    with open(IID_JSON, "w", encoding="utf-8") as handle:
        json.dump({"sigma_m": best, "grid": list(NOISE_GRID_M), "errors": errors}, handle, indent=1)
    return best

def ar1(rng, n, sigma, tau):
    """AR(1) noise with stationary spread sigma and time constant tau seconds."""
    rho = np.exp(-1.0 / (tau * FPS))
    shocks = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + shocks[t]
    return out


DRIFT_GRID = {"window": [1, 5, 9, 13, 17, 21, 25, 31, 41],
              "sigma": [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8],
              "tau": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]}
DRIFT_FIT_FRAMES = 40000  # enough frames for a fast parameter search

def apply_drift(bx, by, length, width, rng, params):
    """The ball as the challenge tracking would see it (normalised in and out)."""
    n = len(bx)
    x = bx.astype(np.float64) + ar1(rng, n, params["sigma"], params["tau"]) / length
    y = by.astype(np.float64) + ar1(rng, n, params["sigma"], params["tau"]) / width
    window = int(params["window"])
    if window > 1:
        x = pd.Series(x).rolling(window, center=True, min_periods=1).mean().values
        y = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
    keep = np.isfinite(bx)
    return (np.where(keep, x, np.nan).astype(np.float32),
            np.where(keep, y, np.nan).astype(np.float32))

def nearest_player_distance(bx, by, px, py, length, width):
    dist = np.sqrt(((px - bx[:, None]) * length) ** 2 + ((py - by[:, None]) * width) ** 2)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.where(np.isfinite(dist), dist, np.nan), axis=1)

def motion_signature(bx, by, length, width, graded, px, py):
    """Percentiles describing how a tracked ball moves and where it sits relative
    to the players (a position error shows up as the ball reading a bit off a
    player it is actually at)."""
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    vx, vy = np.diff(x) * FPS, np.diff(y) * FPS
    speed = np.hypot(vx, vy)
    acc = np.hypot(np.diff(vx), np.diff(vy)) * FPS
    k = 4
    before = np.hypot((x[k:-k] - x[:-2 * k]) * FPS / k, (y[k:-k] - y[:-2 * k]) * FPS / k)
    after = np.hypot((x[2 * k:] - x[k:-k]) * FPS / k, (y[2 * k:] - y[k:-k]) * FPS / k)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.degrees(np.arccos(np.clip(
            ((x[k:-k] - x[:-2 * k]) * (x[2 * k:] - x[k:-k]) +
             (y[k:-k] - y[:-2 * k]) * (y[2 * k:] - y[k:-k])) * FPS ** 2 / k ** 2 /
            (before * after), -1, 1)))
    turn = np.where((before > MOVING) & (after > MOVING), turn, np.nan)
    dspeed = np.abs(after - before)
    g1, g2, gk = graded[1:], graded[2:], graded[k:-k]
    with np.errstate(invalid="ignore"):
        d_min = nearest_player_distance(bx, by, px, py, length, width)
    return np.concatenate([np.nanpercentile(speed[g1], [10, 50, 90]),
                           np.nanpercentile(acc[g2], [50, 90, 99]),
                           np.nanpercentile(turn[gk], [50, 90]),
                           np.nanpercentile(dspeed[gk], [50, 90]),
                           np.nanpercentile(d_min[graded], [10, 25, 50])])


MOVING = 1.0

def drift_params():
    """(window, sigma, tau) fitted so the degraded external ball moves like the
    challenge's: speed, acceleration, turn and speed-change percentiles."""
    if os.path.isfile(DRIFT_JSON):
        with open(DRIFT_JSON, encoding="utf-8") as handle:
            return json.load(handle)["params"]
    import pipeline4.dfl_data as dfl_data
    import pipeline4.features as feats
    px, py, ps, bx, by, team, half, length, width = feats.challenge_arrays()
    target = motion_signature(bx, by, length, width, half > 0, px, py)
    data = dfl_data.load(FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    bx, by = data["bx"][:DRIFT_FIT_FRAMES], data["by"][:DRIFT_FIT_FRAMES]
    graded = data["half"][:DRIFT_FIT_FRAMES] > 0
    px, py = data["px"][:DRIFT_FIT_FRAMES], data["py"][:DRIFT_FIT_FRAMES]
    ranked = []
    for window, sigma, tau in itertools.product(*DRIFT_GRID.values()):
        params = {"window": window, "sigma": sigma, "tau": tau}
        x, y = apply_drift(bx, by, length, width, np.random.RandomState(0), params)
        got = motion_signature(x, y, length, width, graded, px, py)
        error = float(np.mean(np.log((got + 0.5) / (target + 0.5)) ** 2))
        ranked.append((error, params))
    ranked.sort(key=lambda r: r[0])
    payload = {"params": ranked[0][1], "error": ranked[0][0],
               "runner_up": [{"params": p, "error": e} for e, p in ranked[1:6]]}
    with open(DRIFT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return ranked[0][1]


# dead ball model

DEAD_GRID = {"roll_s": [0.5], "placed_s": [1.0, 4.0], "edge_factor": [0.0],
             "carried": [0.35, 0.5], "pickup_s": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
             "inside_factor": [0.0, 1.0]}
ROLL_TAU_S = 1.0  # roll out decay time
EDGE_M = 2.0  # distance from a line
STILL_MPS = 0.5
BACK_GAP_S = 2.0

def distance_to_line(bx, by, length, width):
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    return np.minimum.reduce([np.abs(x), np.abs(length - x), np.abs(y), np.abs(width - y)])

def apply_dead_model(bx, by, label, half, length, width, rng, params):
    """Re-enact every stoppage; returns (bx, by, hidden) with the replaced ball
    and the frames where it is not seen."""
    bx, by = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    n = len(bx)
    hidden = np.zeros(n, dtype=bool)
    decay = np.exp(-1.0 / (ROLL_TAU_S * FPS))
    for s, e in zip(*run_edges((label == 2) & (half > 0))):
        if e >= n or not np.isfinite(bx[e]):
            hidden[s:e] = True
            continue
        span = e - s
        roll = min(int(rng.exponential(params["roll_s"]) * FPS) if params["roll_s"] > 0 else 0, span)
        if roll and s >= 5 and np.isfinite(bx[s - 5]):
            vx, vy = (bx[s - 1] - bx[s - 5]) / 4, (by[s - 1] - by[s - 5]) / 4
            k = np.arange(1, roll + 1)
            travel = decay * (1 - decay ** k) / (1 - decay)
            bx[s:s + roll], by[s:s + roll] = bx[s - 1] + vx * travel, by[s - 1] + vy * travel
        else:
            roll = 0
        by_line = distance_to_line(bx[e:e + 1], by[e:e + 1], length, width)[0] < EDGE_M
        mean = params["placed_s"] * (params["edge_factor"] if by_line else 1.0)
        placed = min(int(rng.exponential(mean) * FPS) if mean > 0 else 0, span - roll)
        carried = rng.rand() < params.get("carried", 0.0)
        middle = e - placed - (s + roll)
        if carried and middle > 0:
            # the tracker follows the ball to the restart spot
            x0 = bx[s + roll - 1] if roll else bx[s]
            y0 = by[s + roll - 1] if roll else by[s]
            f = np.arange(1, middle + 1) / middle
            bx[s + roll:e - placed] = x0 + (bx[e] - x0) * f
            by[s + roll:e - placed] = y0 + (by[e] - y0) * f
            bx[e - placed:e], by[e - placed:e] = bx[e], by[e]
        else:
            bx[s + roll:e], by[s + roll:e] = bx[e], by[e]
            hidden[s + roll:e - placed] = True
        late = params.get("pickup_s", 0.0) * (1.0 if by_line else params.get("inside_factor", 1.0))
        if late > 0:
            # the ball reappears after the thrower moves away
            hidden[e:min(e + int(rng.exponential(late) * FPS), n)] = True
    return bx.astype(np.float32), by.astype(np.float32), hidden

def dead_signature(px, py, ps, bx, by, half, length, width):
    """Statistics of the seen ball around stoppages; no labels."""
    graded = half > 0
    seen = np.isfinite(bx)
    standing = speed_2s(ps) < STANDING_MPS
    outside = seen & ((bx < -0.5 / length) | (bx > 1 + 0.5 / length) |
                      (by < -0.5 / width) | (by > 1 + 0.5 / width))
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
    still = seen & (pd.Series(step).rolling(5, center=True, min_periods=1).median().values < STILL_MPS)
    line = distance_to_line(bx, by, length, width) < EDGE_M
    starts, ends = run_edges(~seen & graded)
    back = ends[((ends - starts) >= BACK_GAP_S * FPS) & (ends < len(bx))]
    back = back[seen[back]]
    return np.array([outside[graded & seen].mean(), line[graded & seen].mean(),
                     still[graded & seen].mean(), still[graded & seen & standing].mean(),
                     seen[graded & standing].mean(), line[back].mean(),
                     still[back].mean(), standing[back].mean()])

def dead_params():
    if os.path.isfile(DEAD_JSON):
        with open(DEAD_JSON, encoding="utf-8") as handle:
            return json.load(handle)["params"]
    import pipeline4.dfl_data as dfl_data
    import pipeline4.features as feats
    px, py, ps, bx, by, team, half, length, width = feats.challenge_arrays()
    target = dead_signature(px, py, ps, bx, by, half, length, width)
    data = dfl_data.load(FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    ranked = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for values in itertools.product(*DEAD_GRID.values()):
            params = dict(zip(DEAD_GRID, values))
            got = []
            for scenario in MATCHED:
                out = distort(data, scenario, 2027, ball="drift", dead=params)
                got.append(dead_signature(out["px"], out["py"], out["ps"], out["bx"], out["by"],
                                          out["half"], length, width))
            got = np.mean(got, axis=0)
            error = float(np.mean(np.log((got + 0.02) / (target + 0.02)) ** 2))
            ranked.append((error, params))
    ranked.sort(key=lambda r: r[0])
    with open(DEAD_JSON, "w", encoding="utf-8") as handle:
        json.dump({"params": ranked[0][1], "error": ranked[0][0]}, handle, indent=1)
    return ranked[0][1]


# entry point

def _apply_ball_noise(out, rng, missing, sigma, ball, length, width):
    """Noise on the seen ball, then the gaps.  The iid draws happen in both ball
    modes so the players and gaps of a seed are the same whichever is chosen."""
    n = len(out["bx"])
    clean_x, clean_y = out["bx"], out["by"]
    out["bx"] = clean_x + rng.normal(0, sigma / length, n).astype(np.float32)
    out["by"] = clean_y + rng.normal(0, sigma / width, n).astype(np.float32)
    if ball == "drift":
        out["bx"], out["by"] = apply_drift(clean_x, clean_y, length, width,
                                           np.random.RandomState(rng.randint(2 ** 31)),
                                           drift_params())
    elif ball != "iid":
        raise ValueError(ball)
    out["bx"][missing] = np.nan
    out["by"][missing] = np.nan
    return out

def _distort_matched(out, scenario, rng, graded, length, width, ball="iid", hidden=None):
    """The ball part of the matched scenarios; players were dropped already."""
    prof = profile()
    n = len(out["bx"])
    bins = speed_bin(speed_2s(out["ps"]))
    pool = np.array(prof["gap_frames"])
    start_bins = np.array(prof["gap_start_bin"])
    lengths = []
    for k in range(len(prof["miss_rate"])):
        own = pool[start_bins == k]
        lengths.append(own if len(own) >= 20 else pool)
    target = np.nan_to_num(np.array(prof["miss_rate"]), nan=BALL_MISSING)
    base = np.zeros(n, dtype=bool) if hidden is None else hidden.copy()
    if scenario == "matched_dead":
        share = float(prof["standing_miss"])
        sd = EDGE_WANDER_S * FPS
        for s, e in zip(*run_edges((out["label"] == 2) & graded)):
            if rng.rand() < share:
                a = int(round(s + rng.normal(0, sd)))
                b = int(round(e + rng.normal(0, sd)))
                if b > a:
                    base[max(a, 0):min(b, n)] = True
    missing = hazard_fill(rng, bins, graded, base & graded, target, lengths)
    return _apply_ball_noise(out, rng, missing, iid_noise_m(), ball, length, width)

def distort(data, scenario, seed, ball="iid", dead=None):
    """A degraded copy of an external match (layout of dfl_data.load).

    ball: "iid" (independent noise) or "drift" (the fitted drift model).
    dead: None keeps the DFL ball frozen through stoppages; "real" (or a dict of
    parameters) re-enacts them, with its own random draws whose hidden frames
    count towards the missing ball."""
    if scenario not in SCENARIOS + MATCHED:
        raise ValueError(scenario)
    rng = np.random.RandomState(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    n, n_players = out["px"].shape
    graded = out["half"] > 0
    length, width = float(out["length"]), float(out["width"])

    # player tracking averages about 20 of 22 players
    on_pitch = np.isfinite(out["px"])
    mean_on = PLAYER_GAP_MEAN_S * FPS * PLAYER_VISIBLE / (1 - PLAYER_VISIBLE)
    for c in range(n_players):
        drop = alternating_mask(rng, n, mean_on,
                                lambda: max(1, int(rng.exponential(PLAYER_GAP_MEAN_S * FPS))))
        drop &= on_pitch[:, c]
        out["px"][drop, c] = np.nan
        out["py"][drop, c] = np.nan
        out["ps"][drop, c] = np.nan

    hidden = np.zeros(n, dtype=bool)
    if dead is not None:
        params = dead_params() if dead == "real" else dead
        out["bx"], out["by"], hidden = apply_dead_model(
            out["bx"], out["by"], out["label"], out["half"], length, width,
            np.random.RandomState(seed + 7919), params)

    if scenario in MATCHED:
        return _distort_matched(out, scenario, rng, graded, length, width, ball, hidden)

    base = hidden.copy()
    if scenario == "stoppage":
        pad = int(STOPPAGE_PAD_S * FPS)
        for s, e in zip(*run_edges(out["label"] == 2)):
            if rng.rand() < STOPPAGE_HIDE:
                base[max(s - rng.randint(0, pad + 1), 0):min(e + rng.randint(0, pad + 1), n)] = True
    missing = fill_to_share(rng, n, graded, base)
    return _apply_ball_noise(out, rng, missing, IID_NOISE_M, ball, length, width)
