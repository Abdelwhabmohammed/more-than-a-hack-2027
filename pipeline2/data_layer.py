"""Dense ATD parsing + derived ball/team signals for the attempt-2 pipeline.

Loads the tactical ATD once into dense arrays (cached to disk), then computes the
signals the evidence layer needs:

  * per-team nearest-player distance to the ball (metres)
  * ball velocity vector with teleport-jump invalidation
  * ball-to-nearest-player relative speed, per team
  * carry score per team (ball glued to a player of that team, moving with it)
  * physics-aware ball gap filling (short gaps only; long gaps stay unknown and
    become dead-ball evidence instead of fake positions)
  * out-of-play geometry (ball beyond the pitch bounds)
  * player freeze detection (whistle / stoppage walking pace)
  * data-driven set-piece dead intervals: for each of the 59 Smart Tagging set
    pieces, the dead window is measured from the tracking (last live frame before
    the restart tag, first live frame after it), with fixed-duration fallbacks only
    where the tracking cannot answer.

All thresholds are football priors stated inline in seconds/metres -- none of them
is fitted to the public ground truth.

    cd more-than-a-hack-2027
    python -m pipeline2.data_layer
"""

import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import uniform_filter1d

# empty frames are expected
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
LENGTH, WIDTH = PITCH
HOME, AWAY = 0, 1
CACHE_DIR = os.path.join(HERE, "_cache")
CORE_CACHE = os.path.join(CACHE_DIR, "core_tactical.npz")

# football priors
BALL_JUMP_M = 8.0 # teleport threshold
FILL_SHORT_GAP = 12 # short-gap fill limit
FILL_MEDIUM_GAP = 75 # medium-gap fill limit
CARRY_DIST_M = 1.8  # max ball-to-player distance
CARRY_RELSPEED = 2.5 # max relative speed (m/s)
CARRY_WINDOW = 25 # carry window (~1 s)
OOB_MARGIN = 0.005 # out-of-bounds margin
FREEZE_SPEED = 1.0 # stoppage speed threshold
FREEZE_TIME = 2.0 # stoppage duration
SETPIECE_SEARCH_BACK_S = 40.0 # search before the tag
SETPIECE_SEARCH_FWD_S = 30.0 # search after the tag
LIVE_BALL_SPEED = 2.0 # live-play speed threshold (m/s)
# confirm live play by ball movement
LIVE_CONFIRM_FRAMES = 10 # live-play confirmation frames

# fallback durations for set pieces
SETPIECE_FALLBACK_S = {
    "kick off (start)": 15.0,
    "kick off (after goal)": 35.0,
    "goal kick": 15.0,
    "corner kick": 12.0,
    "free kick": 12.0,
    "throw-in": 8.0,
    "default": 12.0,
}

# parse core data

def team_of_each_track(meta_path):
    """array over track slots: 0 = home, 1 = away (same convention as starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)

def _parse_core(raw_path, meta_path, n_slots_expect=None):
    """stream the ATD txt into dense arrays."""
    team_of_slot = team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    players = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((N_FRAMES, 2), np.nan, dtype=np.float64)
    ball_speed = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_visible = np.zeros(N_FRAMES, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1
            if vf < 0 or vf >= N_FRAMES:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N" or slot >= n_slots:
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                players[vf, slot, 0] = float(group[:c1])
                players[vf, slot, 1] = float(group[c1 + 1:c2])
                player_speed[vf, slot] = float(group[c2 + 1:])
            ball = parts[2].strip()
            if ball[0] != "N":
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                ball_xy[vf, 0] = float(ball[:c1])
                ball_xy[vf, 1] = float(ball[c1 + 1:c2])
                ball_speed[vf] = float(ball[c2 + 1:])
                ball_visible[vf] = True
    return dict(players=players, player_speed=player_speed, ball_xy=ball_xy,
                ball_speed=ball_speed, ball_visible=ball_visible,
                team_of_slot=team_of_slot)

def load_core(data_dir=None, use_cache=True):
    """Dense tactical ATD arrays, parsed once and cached to npz."""
    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    if use_cache and os.path.isfile(CORE_CACHE):
        cache_mtime = os.path.getmtime(CORE_CACHE)
        if cache_mtime > max(os.path.getmtime(meta_path), os.path.getmtime(raw_path)):
            with np.load(CORE_CACHE) as z:
                return {k: z[k] for k in z.files}
    core = _parse_core(raw_path, meta_path)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(CORE_CACHE, **core)
    return core

# build derived signals
def _ball_velocity(ball_xy, ball_visible):
    """Ball velocity vector in m/s, centred differences, teleports invalidated."""
    n = len(ball_xy)
    v = np.full((n, 2), np.nan, dtype=np.float64)
    both = ball_visible[1:] & ball_visible[:-1]
    d = (ball_xy[1:] - ball_xy[:-1]) * np.array([LENGTH, WIDTH])   # convert to metres
    jump = np.sqrt((d ** 2).sum(axis=1))
    ok = both & (jump < BALL_JUMP_M)
    # use adjacent frames when available
    fwd = np.full((n, 2), np.nan)
    bwd = np.full((n, 2), np.nan)
    fwd[:-1][ok] = d[ok] * FPS
    bwd[1:][ok] = d[ok] * FPS
    with np.errstate(invalid="ignore"):
        v = (fwd + bwd) / 2.0
        v[np.isnan(fwd) & ~np.isnan(bwd)] = bwd[np.isnan(fwd) & ~np.isnan(bwd)]
        v[np.isnan(bwd) & ~np.isnan(fwd)] = fwd[np.isnan(bwd) & ~np.isnan(fwd)]
    return uniform_filter1d(np.nan_to_num(v, nan=0.0), size=5, axis=0, mode="nearest") * \
        np.isfinite(v[:, 0:1])

def _gap_fill(ball_xy, ball_visible):
    """Physics-aware gap filling"""
    n = len(ball_xy)
    fill_xy = ball_xy.copy()
    fill_conf = np.where(ball_visible, 1.0, 0.0).astype(np.float32)
    frames_since = np.full(n, 0, dtype=np.int32)
    vis_idx = np.flatnonzero(ball_visible)

    # count frames since the ball was last seen
    since = 0
    for f in range(n):
        if ball_visible[f]:
            since = 0
        else:
            since += 1
        frames_since[f] = since
    if len(vis_idx) >= 2:
        for i in range(len(vis_idx) - 1):
            a, b = vis_idx[i], vis_idx[i + 1]
            gap = b - a - 1
            if gap <= 0 or gap > FILL_MEDIUM_GAP:
                continue
            x0, y0 = ball_xy[a]
            x1, y1 = ball_xy[b]
            if gap <= FILL_SHORT_GAP:
                # estimate velocity from nearby visible frames
                v0 = np.array([0.0, 0.0])
                v1 = np.array([0.0, 0.0])
                if a - 1 >= 0 and ball_visible[a - 1]:
                    step = ball_xy[a] - ball_xy[a - 1]
                    if np.hypot(*(step * np.array([LENGTH, WIDTH]))) < BALL_JUMP_M:
                        v0 = step
                if b + 1 < n and ball_visible[b + 1]:
                    step = ball_xy[b + 1] - ball_xy[b]
                    if np.hypot(*(step * np.array([LENGTH, WIDTH]))) < BALL_JUMP_M:
                        v1 = step
                for j in range(1, gap + 1):
                    w = j / (gap + 1)          # interpolation weight
                    t = j                     # frames from the start
                    cv0 = np.array([x0, y0]) + v0 * t
                    cv1 = np.array([x1, y1]) - v1 * (gap + 1 - t)
                    pos = (1.0 - w) * cv0 + w * cv1
                    fill_xy[a + j] = pos
                    fill_conf[a + j] = 0.75 * (1.0 - 0.5 * gap / FILL_SHORT_GAP)
            else:
                for j in range(1, gap + 1):
                    w = j / (gap + 1)
                    fill_xy[a + j] = (1.0 - w) * np.array([x0, y0]) + w * np.array([x1, y1])
                    fill_conf[a + j] = 0.30

    return fill_xy, fill_conf, frames_since

def build_signals(core, smart_tagging_path=None, data_dir=None):
    """All derived signals the evidence layer consumes."""
    players = core["players"]
    player_speed = core["player_speed"]
    ball_xy = core["ball_xy"]
    ball_speed = core["ball_speed"].astype(np.float64)
    ball_visible = core["ball_visible"]
    team_of_slot = core["team_of_slot"]
    n = N_FRAMES
    home_mask = team_of_slot == HOME
    sig = {}
    sig["ball_visible"] = ball_visible
    sig["ball_xy"] = ball_xy
    sig["ball_speed"] = ball_speed
    # ball-independent signals
    mean_player_speed = np.nanmean(player_speed, axis=1)
    sig["mean_player_speed"] = mean_player_speed
    smooth_speed = uniform_filter1d(np.nan_to_num(mean_player_speed, nan=0.0), size=int(FREEZE_TIME * FPS), mode="nearest")
    sig["freeze"] = np.clip((FREEZE_SPEED - smooth_speed) / FREEZE_SPEED, 0.0, 1.0) \
        if smooth_speed.max() > 0 else np.zeros(n)
    tracked = np.sum(np.isfinite(players[:, :, 0]), axis=1)
    sig["n_tracked"] = tracked
    sig["n_home_tracked"] = np.sum(np.isfinite(players[:, :, 0]) & home_mask[None, :], axis=1)
    sig["n_away_tracked"] = np.sum(np.isfinite(players[:, :, 0]) & ~home_mask[None, :], axis=1)

    # data for rebuilding ball-dependent signals
    sig["_players"] = players
    sig["_home_mask"] = home_mask

    # shared player velocity
    pv = np.full_like(players, np.nan)
    pv[1:-1] = (players[2:] - players[:-2]) * 0.5 * FPS * np.array(
        [LENGTH, WIDTH], dtype=np.float32)
    sig["player_v"] = pv

    # ball-dependent signals
    _recompute_ball_dependents(sig)
    sig["setpiece_windows"] = build_setpiece_windows(smart_tagging_path, data_dir, sig)
    return sig

def _recompute_ball_dependents(sig):
    """(Re)compute every signal that reads the ball arrays"""
    players = sig["_players"]
    home_mask = sig["_home_mask"]
    ball_xy = sig["ball_xy"]
    ball_visible = sig["ball_visible"]
    n = len(ball_xy)

    # nearest-player distances
    with np.errstate(invalid="ignore"):
        dx = (players[:, :, 0] - ball_xy[:, 0:1]) * LENGTH
        dy = (players[:, :, 1] - ball_xy[:, 1:2]) * WIDTH
        dists = np.sqrt(dx * dx + dy * dy)
    d_home = np.nanmin(np.where(home_mask[None, :], dists, np.nan), axis=1)
    d_away = np.nanmin(np.where(~home_mask[None, :], dists, np.nan), axis=1)
    sig["d_home"], sig["d_away"] = d_home, d_away
    denom = d_home + d_away
    margin = np.zeros(n)
    has_ball = np.isfinite(denom) & (denom > 1e-6)
    margin[has_ball] = (d_away[has_ball] - d_home[has_ball]) / denom[has_ball]
    sig["margin"] = margin                     # positive favors home
    sig["has_ball_obs"] = has_ball

    # skip untracked players
    finite_dists = np.where(np.isfinite(dists), dists, np.inf)
    d_home_masked = np.where(home_mask[None, :], finite_dists, np.inf)
    d_away_masked = np.where(~home_mask[None, :], finite_dists, np.inf)
    near_home_slot = np.argmin(d_home_masked, axis=1)
    near_away_slot = np.argmin(d_away_masked, axis=1)
    finite_home = np.isfinite(np.take_along_axis(
        dists, near_home_slot[:, None], axis=1)[:, 0])
    finite_away = np.isfinite(np.take_along_axis(
        dists, near_away_slot[:, None], axis=1)[:, 0])

    # ball velocity
    sig["ball_v"] = _ball_velocity(ball_xy, ball_visible)
    ball_speed_mag = np.hypot(sig["ball_v"][:, 0], sig["ball_v"][:, 1])
    sig["ball_speed_est"] = np.where(ball_visible, ball_speed_mag, np.nan)
    # use ATD speed when available
    sig["ball_speed_raw"] = sig["ball_speed"]
    sig["ball_speed"] = np.where(np.isfinite(sig["ball_speed_raw"]), sig["ball_speed_raw"], sig["ball_speed_est"])

    # nearest-player velocity
    pv = sig.get("player_v")
    if pv is None:
        pv = np.full_like(players, np.nan)
        pv[1:-1] = (players[2:] - players[:-2]) * 0.5 * FPS * np.array(
            [LENGTH, WIDTH], dtype=np.float32)
        sig["player_v"] = pv
    rows = np.arange(n)
    v_near_h = pv[rows, near_home_slot]
    v_near_a = pv[rows, near_away_slot]
    rel_h = np.full(n, np.nan)
    rel_a = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        rel_h[finite_home] = np.hypot(
            *(sig["ball_v"][finite_home] - v_near_h[finite_home]).T)
        rel_a[finite_away] = np.hypot(
            *(sig["ball_v"][finite_away] - v_near_a[finite_away]).T)
    sig["rel_speed_home"] = rel_h
    sig["rel_speed_away"] = rel_a

    # carry score by team
    carry_h_raw = has_ball & (d_home < CARRY_DIST_M) & \
        (np.nan_to_num(rel_h, nan=99.0) < CARRY_RELSPEED)
    carry_a_raw = has_ball & (d_away < CARRY_DIST_M) & \
        (np.nan_to_num(rel_a, nan=99.0) < CARRY_RELSPEED)
    sig["carry_home"] = uniform_filter1d(
        carry_h_raw.astype(np.float32), size=CARRY_WINDOW, mode="nearest")
    sig["carry_away"] = uniform_filter1d(
        carry_a_raw.astype(np.float32), size=CARRY_WINDOW, mode="nearest")

    # fill short ball gaps
    fill_xy, fill_conf, frames_since = _gap_fill(ball_xy, ball_visible)
    sig["fill_xy"], sig["fill_conf"], sig["frames_since_ball"] = fill_xy, fill_conf, frames_since

    # estimate affinity from reliable fills
    fill_ok = (~ball_visible) & (fill_conf >= 0.5)
    if fill_ok.any():
        with np.errstate(invalid="ignore"):
            fdx = (players[:, :, 0] - fill_xy[:, 0:1]) * LENGTH
            fdy = (players[:, :, 1] - fill_xy[:, 1:2]) * WIDTH
            fdists = np.sqrt(fdx * fdx + fdy * fdy)
        fd_home = np.nanmin(np.where(home_mask[None, :], fdists, np.nan), axis=1)
        fd_away = np.nanmin(np.where(~home_mask[None, :], fdists, np.nan), axis=1)
        fdenom = fd_home + fd_away
        fhas = np.isfinite(fdenom) & (fdenom > 1e-6) & fill_ok
        margin[fhas] = (fd_away[fhas] - fd_home[fhas]) / fdenom[fhas]
        sig["margin_filled_mask"] = fhas
    else:
        sig["margin_filled_mask"] = fill_ok

    # out-of-play check
    geom = np.where(ball_visible[:, None], ball_xy, fill_xy)
    geom_ok = ball_visible | (fill_conf > 0.3)
    x, y = geom[:, 0], geom[:, 1]
    sig["oob"] = geom_ok & ((x < -OOB_MARGIN) | (x > 1 + OOB_MARGIN) |
                            (y < -OOB_MARGIN) | (y > 1 + OOB_MARGIN))
    sig["near_edge"] = geom_ok & ~sig["oob"] & (
        (x < 2 * OOB_MARGIN) | (x > 1 - 2 * OOB_MARGIN) |
        (y < 2 * OOB_MARGIN) | (y > 1 - 2 * OOB_MARGIN))

def rebuild_ball_dependents(sig):
    """Public wrapper: recompute ball-dependent signals after a ball swap."""
    _recompute_ball_dependents(sig)
    return sig

# find set-piece windows
def _load_setpieces(data_dir=None):
    """(start_frame, end_frame, type, team) for every SET PIECES / GOALS instance."""
    from pipeline.smart_tagging_prior import _load_smart_tagging, TEAM_MAP, H, A
    events = _load_smart_tagging(data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR)
    out = []
    for ev in events:
        code = ev["code"].strip().lower()
        if code not in ("set pieces", "goals"):
            continue
        sp_type = (ev["tags"].get("Type") or ev["tags"].get("type") or "default").strip().lower()
        team_raw = (ev["tags"].get("Team") or "").strip().lower()
        team = TEAM_MAP.get(team_raw, -1)
        out.append((ev["start_frame"], ev["end_frame"], sp_type, team))
    return out

def _is_live(sig, f):
    """Heuristic live-play test at frame f: ball actually in play"""
    if f < 0 or f >= len(sig["ball_visible"]):
        return False
    if not sig["ball_visible"][f] or sig["oob"][f]:
        return False
    speed = sig["ball_speed"][f]
    if not np.isfinite(speed) or not (speed > LIVE_BALL_SPEED):
        speed_est = sig["ball_speed_est"][f]
        if not np.isfinite(speed_est) or not (speed_est > LIVE_BALL_SPEED):
            return False
    return True

def _live_streak(sig, f, need=LIVE_CONFIRM_FRAMES):
    """True when the ball is live here and stays live for `need` of the next frames."""
    seen = 0
    for g in range(f, min(len(sig["ball_visible"]), f + 2 * need)):
        if _is_live(sig, g):
            seen += 1
            if seen >= need:
                return True
        elif g > f and not sig["ball_visible"][g]:
            seen = 0
    return False

def build_setpiece_windows(smart_tagging_path, data_dir, sig):
    """measure each set piece's dead window from the tracking; fall back to priors"""
    n = N_FRAMES
    setpieces = _load_setpieces(data_dir)
    windows = []
    back = int(SETPIECE_SEARCH_BACK_S * FPS)
    fwd = int(SETPIECE_SEARCH_FWD_S * FPS)

    for sf, ef, sp_type, team in setpieces:
        # last live frame before the tag
        dead_start = None
        for f in range(min(n - 1, sf), max(0, sf - back), -1):
            if _is_live(sig, f):
                dead_start = f + 1
                break
        # first sustained live frame after the tag
        dead_end = None
        search_end = min(n, max(ef + 1, sf + fwd))
        for f in range(max(0, sf), search_end):
            if _live_streak(sig, f):
                dead_end = f
                break

        source = "data"
        if dead_start is None or dead_end is None or dead_end <= dead_start:
            # use the fallback duration
            dur = int(SETPIECE_FALLBACK_S.get(sp_type, SETPIECE_FALLBACK_S["default"]) * FPS)
            dead_start = max(0, sf - dur)
            dead_end = max(dead_start + 1, sf)
            source = "prior"
        windows.append({"start": int(dead_start), "end": int(dead_end),
                        "type": sp_type, "team": int(team), "source": source})

    windows.extend(_goal_celebration_windows(setpieces, windows, n))
    return windows

def _goal_celebration_windows(setpieces, sp_windows, n):
    """Dead from each goal until the following kickoff restart"""
    goals = sorted(sf for sf, _, code, _ in setpieces if code == "goals")
    kickoffs = sorted(w["end"] for w in sp_windows
                      if w["type"] == "kick off (after goal)")
    out = []
    for gf in goals:
        ko = next((k for k in kickoffs if k > gf), None)
        if ko is None:
            continue
        out.append({"start": int(gf), "end": int(min(n, ko)), "type": "goal celebration", "team": -1, "source": "data"})
    return out

if __name__ == "__main__":
    t0 = time.time()
    print("loading core ATD (cached after first run)")
    core = load_core()
    print(f"core arrays in {time.time() - t0:.1f} s "
          f"(ball visible {core['ball_visible'].sum()} frames)")

    t0 = time.time()
    sig = build_signals(core)
    print(f"  signals in {time.time() - t0:.1f} s")
    w = sig["setpiece_windows"]
    data_driven = sum(1 for x in w if x["source"] == "data")
    print(f"\nset-piece windows: {len(w)} total, {data_driven} data-driven, "
          f"{len(w) - data_driven} prior fallbacks")
    for x in w[:8]:
        print(f"  {x['type']:<22s} frames {x['start']:>6d}..{x['end']:<6d} "
              f"({(x['end'] - x['start']) / FPS:5.1f} s) [{x['source']}] team={x['team']}")
    for key in ("margin", "carry_home", "carry_away", "oob", "freeze"):
        arr = sig[key]
        print(f"{key:<14s} mean={np.nanmean(arr):+.3f}  nonzero={100 * np.mean(np.nan_to_num(arr) != 0):5.1f}%")