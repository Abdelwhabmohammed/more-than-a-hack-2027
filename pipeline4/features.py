"""The per-frame signal table: one row per video frame, one column per signal.

One function builds the table for **any** match in the ATD layout -- the challenge
match and the external Bundesliga training matches go through it unchanged, so a
parameter learned on one means the same thing on the other:

    import pipeline4.features as feats
    table = feats.table_for(px, py, ps, bx, by, team, half, length, width)
    table = feats.challenge_table()            # the challenge match, cached

Three layers of signals, each answering something the previous one cannot:

  layer 1 -- where the ball is and who is near it
      visibility and position of the ball, its speed, each team's nearest-player
      distance, tracked-object counts, mean player speed now and over 2 s, and
      the gap structure (length, position, nearest team just before / after)

  layer 2 -- time context and shape
      player speed over 1/4/8 s and past-vs-next 2 s, how long since each team
      was last within 1.5 m of a seen ball (and until it next is -- possession
      passes on controlled contact, so these carry the team through passes and
      short gaps), ball-visibility timing, team spread/centroid oriented so
      "forward" is the direction the home team attacks, counts near the ball

  layer 3 -- touches and control
      a *touch* is a seen ball that turns by > 25 deg or changes speed by
      > 2.5 m/s (measured over 4 frames either side) with a player within 1.5 m,
      one per burst at the closest approach; *control* is the same player slot
      staying within 2 m of the ball frame after frame.  Per team: time since /
      until its last / next touch, touch counts over 2 s, longest control runs.

Distances are metres, times are frames (25 per second), capped at CAP frames.
Every signal is a home/away pair or symmetric, and `table_for(..., team=swap_sides(team))`
gives the same match seen from the other side -- training uses both so the model
stays blind to which team is home.

The challenge table additionally carries Smart Tagging columns (tag_home, tag_away,
setpiece), which external matches do not have; they are excluded from the model
columns for that reason.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import games
import starter.baseline
import starter.load_smart_tagging as tagging

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if sub not in sys.path:
        sys.path.insert(0, sub)

# constants
FPS = 25
CAP = 1500  # frame cap for since and until signals
OUTSIDE_MARGIN_M = 0.5  # margin for balls outside the pitch
BALL_SPEED_WINDOW = 5  # median window for ball speed
SPEED_2S_WINDOW = 2 * FPS
GAP_CONTEXT = FPS  # frames around a gap to check nearby teams
CLOSE_M = 1.5  # distance for player contact with the ball
NEAR_M, AROUND_M = 3.0, 8.0  # player count distances around the ball
KICK_SPAN = 4  # frames on either side for turn and speed change
MOVING_MPS = 1.0  # minimum speed for ball direction
TURN_DEG = 25.0  # turn angle threshold for a kick
DSPEED_MPS = 2.5
TOUCH_M = 1.5
CONTROL_M = 2.0
TOUCH_WINDOW = 2 * FPS

# table columns used for bookkeeping, not model input
NON_SIGNALS = ("frame", "half", "nearest_team", "ball_x", "ball_y",
               "tag_home", "tag_away", "setpiece")
SPEED_COLUMNS = ("speed_mean", "speed_mean_2s", "speed_1s", "speed_4s", "speed_8s",
                 "speed_past_2s", "speed_next_2s", "speed_change", "speed_top3", "speed_sd")

# smart tagging codes from overview
# attacking phases belong to the team with the ball
# defending phases belong to the team without it
ATTACK_CODES = ("build up", "progression", "creating chances", "attacking transition",
                "long ball", "ball in final third", "ball in the box",
                "players in the box", "goals")
DEFEND_CODES = ("defending in attacking third", "defending in middle third",
                "defending in defensive third", "defensive transition")

GAME = games.GAME
PITCH = games.GAMES[GAME]["pitch"]
N_FRAMES = games.source_frames("tactical", GAME)
CHALLENGE_CACHE = os.environ.get("LP4_CHALLENGE_TABLE",
                                  os.path.join(HERE, "_cache", "feature_table.npz"))


# small time helpers

def run_edges(mask):
    """(starts, ends) of the true-runs of a boolean array, ends exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)

def mean_window(values, window, center=True):
    return pd.Series(values).rolling(window, center=center, min_periods=1).mean().values

def mean_next(values, window):
    """Mean over the next `window` frames, current frame included."""
    return pd.Series(values[::-1]).rolling(window, min_periods=1).mean().values[::-1]

def frames_since(mask, cap=CAP):
    """Frames since mask was last true (0 on true frames), capped."""
    n = len(mask)
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -10 ** 9))
    return np.minimum(np.arange(n) - last, cap).astype(np.float32)

def frames_until(mask, cap=CAP):
    return frames_since(mask[::-1], cap)[::-1]

def value_last(mask, values, fill=-1):
    """values at the most recent frame where mask is true (fill before the first)."""
    n = len(mask)
    index = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    return np.where(index >= 0, values[np.maximum(index, 0)], fill)

def value_next(mask, values, fill=-1):
    return value_last(mask[::-1], values[::-1], fill)[::-1]

def shifted(values, k):
    """values[t + k], NaN outside the array."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out

def run_of_key(key):
    """For an integer key per frame (-1 = none): (run start, run length) per frame."""
    n = len(key)
    fresh = np.ones(n, dtype=bool)
    fresh[1:] = key[1:] != key[:-1]
    start = np.maximum.accumulate(np.where(fresh, np.arange(n), 0))
    ends = np.flatnonzero(np.concatenate((fresh[1:], [True])))
    by_start = np.zeros(n, dtype=np.int64)
    starts = np.flatnonzero(fresh)
    by_start[starts] = ends - starts + 1
    return start, by_start[start]

def majority_team(values):
    values = values[values >= 0]
    return int(np.bincount(values, minlength=2).argmax()) if len(values) >= 5 else -1


# layer 1 ball position and nearby players

def base_signals(px, py, ps, bx, by, team, half, length, width):
    n = len(bx)
    home, away = team == 0, team == 1
    out = {"frame": np.arange(n), "half": np.asarray(half).astype(np.int8)}

    seen = np.isfinite(bx) & np.isfinite(by)
    out["ball_seen"] = seen
    out["ball_x"] = np.where(seen, bx * length, np.nan)
    out["ball_y"] = np.where(seen, by * width, np.nan)
    mx, my = OUTSIDE_MARGIN_M / length, OUTSIDE_MARGIN_M / width
    out["ball_outside"] = seen & ((bx < -mx) | (bx > 1 + mx) | (by < -my) | (by > 1 + my))

    step = np.full(n, np.nan)
    both = seen[1:] & seen[:-1]
    step[1:][both] = np.hypot(np.diff(bx.astype(np.float64)) * length,
                              np.diff(by.astype(np.float64)) * width)[both] * FPS
    out["ball_speed"] = pd.Series(step).rolling(BALL_SPEED_WINDOW, center=True,
                                                 min_periods=1).median().values

    dist = np.sqrt(((px - bx[:, None]) * length) ** 2 + ((py - by[:, None]) * width) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    per_home = dist[:, home].min(axis=1)
    per_away = dist[:, away].min(axis=1)
    out["d_home"] = np.where(np.isfinite(per_home), per_home, np.nan)
    out["d_away"] = np.where(np.isfinite(per_away), per_away, np.nan)
    out["nearest_team"] = np.where(np.isfinite(dist.min(axis=1)),
                                   team[dist.argmin(axis=1)], -1).astype(np.int8)

    occupied = np.isfinite(px) & np.isfinite(py)
    out["n_home"] = (occupied & home).sum(axis=1).astype(np.int8)
    out["n_away"] = (occupied & away).sum(axis=1).astype(np.int8)
    usable = np.isfinite(ps)
    count = usable.sum(axis=1)
    out["speed_mean"] = np.where(count > 0,
                                 np.where(usable, ps, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    out["speed_mean_2s"] = mean_window(out["speed_mean"], SPEED_2S_WINDOW)

    # gap length, position, and nearest team before and after
    gap_len = np.zeros(n, dtype=np.int32)
    gap_pos = np.zeros(n, dtype=np.int32)
    before = np.full(n, -1, dtype=np.int8)
    after = np.full(n, -1, dtype=np.int8)
    nearest = out["nearest_team"]
    for s, e in zip(*run_edges(~seen)):
        gap_len[s:e] = e - s
        gap_pos[s:e] = np.arange(e - s)
        before[s:e] = majority_team(nearest[max(s - GAP_CONTEXT, 0):s])
        after[s:e] = majority_team(nearest[e:e + GAP_CONTEXT])
    out["gap_len"], out["gap_pos"] = gap_len, gap_pos
    out["team_before_gap"], out["team_after_gap"] = before, after
    return out


# layer 2 time context and team shape

def context_signals(out, px, py, ps, team, length, width):
    n = len(out["ball_seen"])
    home, away = team == 0, team == 1
    seen = out["ball_seen"]
    ball_x, ball_y = out["ball_x"], out["ball_y"]
    x, y = px * length, py * width

    # forward points toward the goal home attacks
    with np.errstate(invalid="ignore"):
        cx_home = np.nanmean(np.where(home, x, np.nan), axis=1)
        cx_away = np.nanmean(np.where(away, x, np.nan), axis=1)
        sx_home = np.nanstd(np.where(home, x, np.nan), axis=1)
        sx_away = np.nanstd(np.where(away, x, np.nan), axis=1)
        sy_home = np.nanstd(np.where(home, y, np.nan), axis=1)
        sy_away = np.nanstd(np.where(away, y, np.nan), axis=1)
    direction = np.ones(n)
    for h in (1, 2):
        rows = out["half"] == h
        if rows.any():
            direction[rows] = 1.0 if np.nanmedian((cx_away - cx_home)[rows]) >= 0 else -1.0
    out["ball_fwd_home"] = (ball_x - length / 2) * direction
    out["ball_wide"] = np.abs(ball_y - width / 2)
    out["ball_to_touch"] = np.minimum(ball_y, width - ball_y)
    out["ball_to_goalline"] = np.minimum(ball_x, length - ball_x)
    out["centroid_fwd_home"] = (cx_home - length / 2) * direction
    out["centroid_fwd_away"] = (cx_away - length / 2) * direction
    out["centroid_gap"] = (cx_away - cx_home) * direction
    out["spread_x_home"], out["spread_x_away"] = sx_home, sx_away
    out["spread_y_home"], out["spread_y_away"] = sy_home, sy_away

    # counts around the ball and nearest team margin
    dist = np.sqrt((x - ball_x[:, None]) ** 2 + (y - ball_y[:, None]) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    for name, radius in (("near", NEAR_M), ("around", AROUND_M)):
        out["{}_home".format(name)] = (dist[:, home] < radius).sum(axis=1).astype(np.int8)
        out["{}_away".format(name)] = (dist[:, away] < radius).sum(axis=1).astype(np.int8)
    d_home = np.nan_to_num(out["d_home"], nan=np.inf)
    d_away = np.nan_to_num(out["d_away"], nan=np.inf)
    out["d_diff"] = np.clip(np.where(np.isfinite(d_home) & np.isfinite(d_away),
                                     d_away - d_home, np.nan), -15, 15)
    out["d_min"] = np.where(seen, np.minimum(d_home, d_away), np.nan)

    # last and next team in contact with the ball
    contact_home = seen & (d_home < CLOSE_M) & (d_home <= d_away)
    contact_away = seen & (d_away < CLOSE_M) & (d_away < d_home)
    out["since_close_home"], out["since_close_away"] = frames_since(contact_home), frames_since(contact_away)
    out["until_close_home"], out["until_close_away"] = frames_until(contact_home), frames_until(contact_away)
    contact_team = np.where(contact_home, 0, np.where(contact_away, 1, -1))
    any_contact = contact_team >= 0
    out["last_close_team"] = value_last(any_contact, contact_team).astype(np.int8)
    out["next_close_team"] = value_next(any_contact, contact_team).astype(np.int8)
    nearest = out["nearest_team"]
    for name, window in (("1s", FPS), ("3s", 3 * FPS)):
        h = np.where(seen, (nearest == 0).astype(float), np.nan)
        out["home_near_past_" + name] = pd.Series(h).rolling(window, min_periods=1).mean().values
        out["home_near_next_" + name] = mean_next(h, window)

    # speed windows, past vs next, top speeds, and effort spread
    speed = out["speed_mean"]
    out["speed_1s"] = mean_window(speed, FPS)
    out["speed_4s"] = mean_window(speed, 4 * FPS)
    out["speed_8s"] = mean_window(speed, 8 * FPS)
    out["speed_past_2s"] = mean_window(speed, 2 * FPS, center=False)
    out["speed_next_2s"] = mean_next(speed, 2 * FPS)
    out["speed_change"] = out["speed_next_2s"] - out["speed_past_2s"]
    ok = np.isfinite(ps)
    top = np.sort(np.where(ok, ps, -1.0), axis=1)[:, -3:]
    top = np.where(top >= 0, top, np.nan)
    with np.errstate(invalid="ignore"):
        out["speed_top3"] = mean_window(np.nanmean(top, axis=1), 2 * FPS)
        out["speed_sd"] = mean_window(np.nanstd(np.where(ok, ps, np.nan), axis=1), 2 * FPS)
    ball_speed = out["ball_speed"]
    out["ball_speed_1s"] = mean_window(ball_speed, FPS)
    out["ball_speed_max_1s"] = pd.Series(ball_speed).rolling(
        FPS, center=True, min_periods=1).max().values

    # ball visibility timing
    out["since_missing"], out["until_missing"] = frames_since(~seen), frames_until(~seen)
    out["since_seen"], out["until_seen"] = frames_since(seen), frames_until(seen)
    out["since_outside"], out["until_outside"] = frames_since(out["ball_outside"]), frames_until(out["ball_outside"])
    out["seen_share_4s"] = mean_window(seen.astype(float), 4 * FPS)
    return out


# layer 3 touches and control

def touch_signals(px, py, bx, by, team, length, width):
    n = len(bx)
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    k = KICK_SPAN
    vbx, vby = (x - shifted(x, -k)) * FPS / k, (y - shifted(y, -k)) * FPS / k
    vax, vay = (shifted(x, k) - x) * FPS / k, (shifted(y, k) - y) * FPS / k
    before, after = np.hypot(vbx, vby), np.hypot(vax, vay)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (vbx * vax + vby * vay) / (before * after)
    turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    turn = np.where((before > MOVING_MPS) & (after > MOVING_MPS), turn, np.nan)
    dspeed = after - before

    dist = np.sqrt(((px - bx[:, None]) * length) ** 2 + ((py - by[:, None]) * width) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(axis=1)
    d_min = dist[np.arange(n), nearest]
    near_team = np.where(np.isfinite(d_min), team[nearest], -1)

    kick = (np.nan_to_num(turn) > TURN_DEG) | (np.abs(np.nan_to_num(dspeed)) > DSPEED_MPS)
    candidate = kick & (d_min < TOUCH_M)
    touch = np.zeros(n, dtype=bool)
    edges = np.diff(np.concatenate(([0], candidate.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        touch[s + int(np.argmin(d_min[s:e]))] = True  # one touch per burst at closest approach
    touch_team = np.where(touch, near_team, -1)
    touch_home, touch_away = touch & (touch_team == 0), touch & (touch_team == 1)

    out = {"ball_turn": turn, "ball_dspeed": dspeed, "touch": touch.astype(np.int8)}
    out["since_touch_home"], out["since_touch_away"] = frames_since(touch_home), frames_since(touch_away)
    out["until_touch_home"], out["until_touch_away"] = frames_until(touch_home), frames_until(touch_away)
    out["last_touch_team"] = value_last(touch, touch_team).astype(np.int8)
    out["next_touch_team"] = value_next(touch, touch_team).astype(np.int8)
    for name, mask in (("home", touch_home), ("away", touch_away)):
        m = mask.astype(float)
        out["touches_{}_past_2s".format(name)] = pd.Series(m).rolling(
            TOUCH_WINDOW, min_periods=1).sum().values
        out["touches_{}_next_2s".format(name)] = pd.Series(m[::-1]).rolling(
            TOUCH_WINDOW, min_periods=1).sum().values[::-1]

    # control when one player slot stays close on consecutive frames
    controlling = np.where(d_min < CONTROL_M, nearest, -1)
    _, streak = run_of_key(controlling)
    streak = np.where(controlling >= 0, streak, 0)
    control_team = np.where(controlling >= 0, team[np.maximum(controlling, 0)], -1)
    out["control_len"] = np.minimum(streak, CAP).astype(np.float32)
    out["control_team"] = control_team.astype(np.int8)
    for name, side in (("home", 0), ("away", 1)):
        own = np.where(control_team == side, streak, 0).astype(float)
        out["control_{}_past_2s".format(name)] = pd.Series(own).rolling(
            TOUCH_WINDOW, min_periods=1).max().values
        out["control_{}_next_2s".format(name)] = pd.Series(own[::-1]).rolling(
            TOUCH_WINDOW, min_periods=1).max().values[::-1]
    return out


# build the table

def table_for(px, py, ps, bx, by, team, half, length, width):
    """All three signal layers for any match in the ATD layout."""
    import warnings
    with warnings.catch_warnings():  # empty team slices can produce nan
        warnings.simplefilter("ignore", RuntimeWarning)
        out = base_signals(px, py, ps, bx, by, team, half, length, width)
        out = context_signals(out, px, py, ps, team, length, width)
        out.update(touch_signals(px, py, bx, by, team, length, width))
    return out

def model_columns(table):
    return [c for c in table.columns if c not in NON_SIGNALS]

def swap_sides(team):
    """Home slots become away and vice versa; anything else stays."""
    return np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(team.dtype)

def scale_speeds(table):
    """Speed columns as ratios to the match's own median -- "slower than usual"
    must mean the same thing whatever the match and the tracking system."""
    out = table.copy()
    median = float(np.nanmedian(table["speed_mean"].values[table["half"].values > 0]))
    for column in SPEED_COLUMNS:
        out[column] = table[column] / median
    return out


# challenge match

def side_of_club(name):
    """0 / 1 / None for a club name as written in the tagging or metadata."""
    lowered = (name or "").strip().lower()
    if "barcel" in lowered:
        return 0
    if "atletic" in lowered or "athletic" in lowered:
        return 1
    return None

def read_atd(raw_path, n_frames, cache_path, rebuild=False):
    """Stream the raw ATD text into arrays indexed by video frame (frame_id - 1).

    Line format: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>"; an observation that
    starts with N is missing."""
    if not rebuild and os.path.isfile(cache_path):
        stamp = (os.path.getsize(raw_path), int(os.path.getmtime(raw_path)))
        with np.load(cache_path) as cached:
            if tuple(cached["stamp"]) == stamp:
                print("  using cache {}".format(cache_path))
                return {k: cached[k] for k in cached.files if k != "stamp"}
    px = py = ps = None
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    started = time.time()
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                raise ValueError("frame_id {} outside 1..{}".format(i + 1, n_frames))
            groups = [g for g in parts[1].split(";") if g]
            if px is None:
                px = np.full((n_frames, len(groups)), np.nan, dtype=np.float32)
                py = np.full_like(px, np.nan)
                ps = np.full_like(px, np.nan)
            for slot, group in enumerate(groups):
                if group[0] == "N":
                    continue
                fields = group.split(",")
                px[i, slot] = float(fields[0])
                py[i, slot] = float(fields[1])
                if len(fields) > 2:
                    ps[i, slot] = float(fields[2])
            ball = parts[2].strip()
            if ball and ball[0] != "N":
                fields = ball.split(",")
                bx[i] = float(fields[0])
                by[i] = float(fields[1])
    print("  parsed ATD in {:.0f}s".format(time.time() - started))
    data = dict(px=px, py=py, ps=ps, bx=bx, by=by)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    np.savez_compressed(cache_path, stamp=stamp, **data)
    return data

def challenge_arrays(data_dir=None):
    """(px, py, ps, bx, by, team, half, length, width) for the challenge match."""
    import starter.baseline
    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir)
    atd = read_atd(raw, N_FRAMES, os.path.join(HERE, "_cache", "atd_tactical.npz"))
    half = np.zeros(N_FRAMES, dtype=np.int8)
    for h, first, last in games.half_video_blocks(GAME):
        half[first:last + 1] = h
    return (atd["px"], atd["py"], atd["ps"], atd["bx"], atd["by"],
            starter.baseline.team_of_each_track(meta), half, PITCH[0], PITCH[1])

def tagging_columns(data_dir=None):
    """Smart Tagging as per-frame counts; only the challenge match has it."""
    def add(target, a, b):
        target[a] += 1
        target[b] -= 1

    out = {}
    tag_home = np.zeros(N_FRAMES + 1, dtype=np.int32)
    tag_away = np.zeros(N_FRAMES + 1, dtype=np.int32)
    setpiece = np.zeros(N_FRAMES + 1, dtype=np.int32)
    for event in tagging.load(GAME, data_dir):
        code = (event["code"] or "").strip().lower()
        a, b = max(event["start_frame"], 0), min(event["end_frame"], N_FRAMES)
        if b <= a:
            continue
        if code == "set pieces":
            add(setpiece, a, b)
        side = side_of_club(event["tags"].get("Team"))
        if side is None:
            continue
        if code in DEFEND_CODES:
            side = 1 - side
        elif code not in ATTACK_CODES and code != "set pieces":
            continue
        add(tag_home if side == 0 else tag_away, a, b)
    out["tag_home"] = np.cumsum(tag_home)[:N_FRAMES].astype(np.int8)
    out["tag_away"] = np.cumsum(tag_away)[:N_FRAMES].astype(np.int8)
    out["setpiece"] = np.cumsum(setpiece)[:N_FRAMES] > 0
    return out

def challenge_table(rebuild=False, data_dir=None, swap=False):
    """The challenge match's full table; swap=True reads it from the other side."""
    path = CHALLENGE_CACHE.replace(".npz", "_swap.npz") if swap else CHALLENGE_CACHE
    if rebuild or not os.path.isfile(path):
        px, py, ps, bx, by, team, half, length, width = challenge_arrays(data_dir)
        columns = table_for(px, py, ps, bx, by,
                            swap_sides(team) if swap else team, half, length, width)
        columns.update(tagging_columns(data_dir))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        return pd.DataFrame({name: saved[name] for name in saved.files})
