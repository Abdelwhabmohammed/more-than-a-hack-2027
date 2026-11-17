"""Shared machinery for the pipeline4 stages (A: frame+change, B: stacker, C: GRU).

Everything is trained on the seven open DFL/IDSSE matches (CC BY 4.0) degraded to
look like the challenge tracking, over all four degradation scenarios with the
fitted ball-drift and dead-ball models.

    This module sets up sys.path and cache redirection on import.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import evaluation.possession_metrics as pm
import pipeline4.dfl_data as dfl_data
import pipeline4.features as feats
import pipeline4.noise as noise

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in (HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if sub not in sys.path:
        sys.path.insert(0, sub)

CACHE = os.environ.get("LP4_CACHE", os.path.join(HERE, "_cache"))
MODEL_DIR = os.path.join(HERE, "model")

# external matches and the split
TUNE = ("J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ")
HOLDOUT = ("J03WN1", "J03WR9")
ALL_MATCHES = sorted(dfl_data.MATCH_FILES)
SEED = 20271117

# model settings
BALL_MODE = "drift"  # fitted drift noise with smoothing
DEAD_MODE = "real"  # realistic stoppage replay
SCENARIOS = noise.MATCHED + noise.SCENARIOS  # all scenarios used for training
JUDGE_SCENARIOS = noise.MATCHED  # scenarios used to judge decoder and stacker

CHANGE_RADIUS = 6  # frames around a direct team change
FRAME_ALPHA = 0.5  # frame weight based on spell length
FRAME_MIN_LEN = 5
CONTROLLED_MIN = 50  # minimum frames for a controlled team run
STRIDE = 3  # use every third frame to reduce near duplicates
N_TREES = 400
SECOND_HALF_KICKOFF = 113742  # second half starts here after dead frames

LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05,
              "num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "seed": SEED, "num_threads": 0}
CHANGE_PARAMS = dict(LGB_PARAMS, objective="binary")
CHANGE_PARAMS.pop("num_class")

COST_GRID = {"team_a": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40],
             "team_b": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
             "enter_dead": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60, 90],
             "exit_dead": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60, 90],
             "dead_bias": [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]}
ROUNDS = 3

try:
    from numba import njit
except ImportError:
    def njit(*_, **__):
        def wrap(f):
            return f
        return wrap


# labels

def controlled(truth, min_control=CONTROLLED_MIN):
    """Short team runs straight after the other team's run go to that team -- the
    challenge's controlled-possession definition (a touch is not a possession)."""
    out = truth.copy()
    change = np.ones(len(out), dtype=bool)
    change[1:] = out[1:] != out[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(out))
    current = None
    for a, b in zip(starts, ends):
        code = truth[a]
        if code in (pm.H, pm.A):
            if current is not None and code != current and b - a < min_control:
                out[a:b] = current
            else:
                current = code
        else:
            current = None
    return out

def change_target(truth, radius=CHANGE_RADIUS):
    """1 within `radius` frames of a direct H <-> A flip in the truth."""
    flip = np.zeros(len(truth), dtype=bool)
    a, b = truth[:-1], truth[1:]
    at = np.flatnonzero(((a == pm.H) & (b == pm.A)) | ((a == pm.A) & (b == pm.H))) + 1
    for t in at:
        flip[max(t - radius, 0):t + radius + 1] = True
    return flip.astype(np.int8)

def spell_lengths(truth):
    """Length of the H/A/D stretch each frame belongs to."""
    change = np.ones(len(truth), dtype=bool)
    change[1:] = truth[1:] != truth[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(truth))
    lengths = np.empty(len(truth), dtype=np.float64)
    for a, b in zip(starts, ends):
        lengths[a:b] = b - a
    return lengths

def spell_weights(truth, alpha=FRAME_ALPHA, min_len=FRAME_MIN_LEN):
    """(1 / spell length) ** alpha, floored, normalised to mean 1 -- short
    possessions count without swamping the loss."""
    w = np.maximum(spell_lengths(truth), min_len).astype(np.float64) ** -alpha
    return w / w.mean()


# data

def item(match_id, scenario, swap=False):
    """(feature table, truth, half runs) of a degraded external match, cached."""
    path = os.path.join(CACHE, "item_{}_{}{}.npz".format(
        match_id, scenario, "_swap" if swap else ""))
    if not os.path.isfile(path):
        data = dfl_data.load(match_id)
        data = noise.distort(data, scenario, SEED + ALL_MATCHES.index(match_id),
                             ball=BALL_MODE, dead=DEAD_MODE)
        team = 1 - data["team"] if swap else data["team"]
        columns = feats.table_for(data["px"], data["py"], data["ps"], data["bx"], data["by"],
                                  team, data["half"], float(data["length"]), float(data["width"]))
        label = data["label"].copy()
        if swap:
            label = np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)
        columns["truth"] = np.where(data["half"] > 0, label, pm.X).astype(np.int8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        table = pd.DataFrame({k: saved[k] for k in saved.files if k != "truth"})
        truth = saved["truth"]
    return table, truth, half_runs(table["half"].values)

def feature_names(table):
    return feats.model_columns(table)

def rows(matches, scenarios=SCENARIOS):
    """Features / frame labels / change labels / normalised weights, concatenated."""
    xs, ys, cs, ws = [], [], [], []
    for m in matches:
        for s in scenarios:
            for swap in (False, True):
                table, truth, _ = item(m, s, swap)
                keep = np.flatnonzero(truth != pm.X)[::STRIDE]
                xs.append(table.iloc[keep])
                ys.append(truth[keep])
                cs.append(change_target(truth)[keep])
                ws.append(spell_weights(truth)[keep])
    names = feature_names(xs[0])
    return (pd.concat(xs, ignore_index=True)[names], np.concatenate(ys),
            np.concatenate(cs), np.concatenate(ws), names)


# models

def fit_frame(matches, n_trees=N_TREES, scenarios=SCENARIOS):
    x, y, _, w, _ = rows(matches, scenarios)
    import lightgbm as lgb
    return lgb.train(LGB_PARAMS, lgb.Dataset(x, y, weight=w), num_boost_round=n_trees)

def fit_change(matches, n_trees=N_TREES, scenarios=SCENARIOS):
    x, _, c, _, _ = rows(matches, scenarios)
    import lightgbm as lgb
    return lgb.train(CHANGE_PARAMS, lgb.Dataset(x, c), num_boost_round=n_trees)

def predict_probs(frame, change, table, table_swap):
    """(n, 3) H/A/D and (n,) change probabilities, both sides averaged."""
    names = feature_names(table)
    p = frame.predict(table[names].astype(np.float32))
    q = frame.predict(table_swap[names].astype(np.float32))
    prob = np.clip((p + q[:, [1, 0, 2]]) / 2, 1e-4, 1.0)
    if change is not None:
        c = (change.predict(table[names].astype(np.float32)) +
             change.predict(table_swap[names].astype(np.float32))) / 2
        chg = np.clip(c, 1e-4, 1 - 1e-4)
    else:
        chg = np.full(len(prob), 0.5)
    return prob, chg


# decoder with time based team switch costs

@njit(cache=True)
def viterbi_tv(log_e, team_cost, enter_dead, exit_dead):
    """Most likely H/A/D sequence; changing team between t-1 and t costs team_cost[t]."""
    n = log_e.shape[0]
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_e[0].copy()
    new = np.empty(3)
    cost = np.zeros((3, 3))
    for t in range(1, n):
        cost[0, 1] = team_cost[t]
        cost[1, 0] = team_cost[t]
        cost[0, 2] = enter_dead
        cost[1, 2] = enter_dead
        cost[2, 0] = exit_dead
        cost[2, 1] = exit_dead
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                v = score[i] - cost[i, j]
                if v > best:
                    best, arg = v, i
            new[j] = best + log_e[t, j]
            back[t, j] = arg
        for j in range(3):
            score[j] = new[j]
    out = np.zeros(n, dtype=np.int8)
    k = int(np.argmin(score))
    for t in range(n - 1, -1, -1):
        out[t] = k
        k = back[t, k]
    return out

def half_runs(half):
    """(first, last) inclusive frame spans of each half in a half-code array."""
    edges = np.diff(np.concatenate(([0], (np.asarray(half) > 0).astype(np.int8), [0])))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1),
                                                 np.flatnonzero(edges == -1))]

def decode(prob, change, half, d):
    log_e = np.log(prob)
    log_e[:, 2] += d["dead_bias"]
    team_cost = np.maximum(0.0, d["team_a"] - d["team_b"] * np.log(change / (1 - change)))
    codes = np.full(len(prob), pm.X, dtype=np.int8)
    for first, last in half_runs(half):
        codes[first:last + 1] = viterbi_tv(np.ascontiguousarray(log_e[first:last + 1]),
                                           np.ascontiguousarray(team_cost[first:last + 1]),
                                           float(d["enter_dead"]), float(d["exit_dead"]))
    return codes

def metrics(codes, truth, runs):
    cm3 = pm.scored_confusion(pm.confusion(truth, codes))
    spells = pm.spell_scores(truth, codes, runs)
    return {"spellF1": spells["mean_f1"], "macroF1": pm.macro_f1(cm3)}

def summary(cases, d, judge="controlled"):
    """Mean metrics over cases; judged on controlled-rewritten truth by default."""
    vals = []
    for p, c, h, g, r in cases:
        codes = decode(p, c, h, d)
        target = controlled(g) if judge == "controlled" else g
        vals.append(metrics(codes, target, r))
    return {k: float(np.mean([v[k] for v in vals])) for k in ("spellF1", "macroF1")}

def decoder_search(cases, judge="controlled", rounds=ROUNDS, grid=None):
    best = {"team_a": 6.0, "team_b": 0.0, "enter_dead": 90.0, "exit_dead": 10.0, "dead_bias": 1.0}
    best_score = summary(cases, best, judge)["spellF1"]
    print("    decoder start {:.4f}".format(best_score), flush=True)
    for round_number in range(1, rounds + 1):
        changed = False
        for name, values in (grid or COST_GRID).items():
            for value in values:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: float(value)})
                s = summary(cases, trial, judge)["spellF1"]
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            print("    round {} {:<11} -> {:<6} spellF1 {:.4f}".format(
                round_number, name, best[name], best_score), flush=True)
        if not changed:
            break
    return best, best_score


# challenge match

def challenge_tables(rebuild=False, data_dir=None):
    """The challenge match's feature table, both sides (cached)."""
    table = feats.challenge_table(rebuild=rebuild, data_dir=data_dir)
    table_swap = feats.challenge_table(rebuild=rebuild, data_dir=data_dir, swap=True)
    return table, table_swap

def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path

def possession_count(labels, fps=25):
    """(count, mean seconds) of H/A spells inside the halves."""
    spells, current = [], None
    for label in labels:
        if label in ("H", "A"):
            if label == current:
                spells[-1] += 1
            else:
                spells.append(1)
                current = label
        else:
            current = None
    return len(spells), (float(np.mean(spells)) / fps if spells else 0.0)

def save_params(path, **params):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1, default=float)
