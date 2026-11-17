"""Stage B: a second-stage stacker that reads Stage A's out-of-fold probabilities
as a sequence, plus long-context signals, and re-decodes.

    cd more-than-a-hack-2027
    python -m pipeline4.train_stack             # requires Stage A first
    python -m pipeline4.train_stack --smoke

Design:

  stacker inputs per frame:
    * Stage-A probabilities (H / A / D) and the change probability;
    * rolling mean / std of each over +-25, +-75, +-250 frames (~1 s / 3 s / 10 s);
    * long-context signals computed from the v3 table columns: touch chains per
      team, still-ball-at-line run lengths, team-centroid velocities, ball
      progress across tracking gaps;
    * decoded run lengths: how long the current Stage-A label has been running.
  target: the controlled-possession-rewritten truth; the stacker is trained on
  the Stage-A out-of-fold rows of the five TUNE matches, and the decoder prices
  are re-searched on those out-of-fold cases, validated on the two HOLDOUT
  matches (Stage-A models trained on all five TUNE matches predict them).

Nothing here reads the challenge match.
"""

import argparse
import datetime
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import pipeline4.core as core
from pipeline4 import core
from pipeline4.train_stage1 import OUT
import evaluation.possession_metrics as pm

def rolling(values, half, k):
    """Rolling mean and std over +-k frames, per half, NaN-safe."""
    out_m = np.full(len(values), np.nan, dtype=np.float64)
    out_s = np.full(len(values), np.nan, dtype=np.float64)
    for first, last in v1_blocks(half):
        v = pd.Series(values[first:last + 1]).astype(np.float64)
        m = v.rolling(2 * k + 1, center=True, min_periods=k // 2).mean()
        s = v.rolling(2 * k + 1, center=True, min_periods=k // 2).std()
        out_m[first:last + 1] = m.values
        out_s[first:last + 1] = s.values
    return out_m, out_s

def run_lengths(codes, half):
    """How long the current label has been running (frames), per half."""
    out = np.zeros(len(codes), dtype=np.float32)
    for first, last in v1_blocks(half):
        run = 1
        out[first] = 1
        for t in range(first + 1, last + 1):
            run = run + 1 if codes[t] == codes[t - 1] else 1
            out[t] = run
    return out

def still_at_line_runs(table, half):
    """Runs of seen-and-still ball at the touch/goal line, per team."""
    get = lambda name: table[name].values.astype(np.float32)
    seen = get("ball_seen") > 0
    ball_speed = np.nan_to_num(get("ball_speed"), nan=99.0)
    still = seen & (ball_speed < 1.0)
    at_line = still & ((np.nan_to_num(get("ball_to_touch"), nan=99.0) < 1.5) |
                       (np.nan_to_num(get("ball_to_goalline"), nan=99.0) < 1.5))
    cols = {}
    for axis, forward in (("until", False), ("for", True)):
        cols["x_line_" + axis] = same_run(at_line.astype(np.int8), half, forward)
    return cols

def same_run(values, half, forward):
    """frames the value has been the same up to t (or will stay from t, forward)."""
    out = np.zeros(len(values), dtype=np.float32)
    for first, last in v1_blocks(half):
        v = values[first:last + 1]
        n = len(v)
        run = np.zeros(n, dtype=np.float32)
        if forward:
            acc = 1
            for t in range(n - 1, -1, -1):
                acc = acc + 1 if t < n - 1 and v[t] == v[t + 1] else 1
                run[t] = acc
        else:
            acc = 1
            for t in range(n):
                acc = acc + 1 if t > 0 and v[t] == v[t - 1] else 1
                run[t] = acc
        out[first:last + 1] = run
    return out

def touch_chains(table, half):
    """consecutive-touch counts per team up to t / from t."""
    cols = {}
    for name, forward in (("last_touch_team", False), ("next_touch_team", True)):
        values = table[name].values.astype(np.float32)
        for team, tag in ((0, "home"), (1, "away")):
            cols["x_chain_{}_{}".format(tag, "next" if forward else "past")] = \
                same_run(np.where(values == team, team, -1).astype(np.int8), half, forward)
    return cols

def gap_jump(table, half):
    """how far the ball moved across a tracking gap, and the implied speed."""
    get = lambda name: table[name].values.astype(np.float32)
    seen = get("ball_seen") > 0
    fwd = np.where(seen, get("ball_fwd_home"), np.nan)
    wide = np.where(seen, get("ball_wide"), np.nan)
    jump = np.zeros(len(fwd), dtype=np.float32)
    for first, last in v1_blocks(half):
        f, w = pd.Series(fwd[first:last + 1]), pd.Series(wide[first:last + 1])
        jump[first:last + 1] = np.nan_to_num(np.hypot(f.bfill().values - f.ffill().values,
                                                      w.bfill().values - w.ffill().values))
    return {"x_gap_jump": jump,
            "x_gap_speed": jump / np.maximum(np.nan_to_num(get("gap_len")), 1.0)}

def stacker_table(table, prob, chg, decoded, half):
    """the stage-b feature table for one case."""
    cols = {}
    for j, name in enumerate(("p_h", "p_a", "p_d")):
        cols[name] = prob[:, j]
        for k in (25, 75, 250):
            m, s = rolling(prob[:, j], half, k)
            cols["{}_m{}".format(name, k)] = m
            cols["{}_s{}".format(name, k)] = s
    cols["p_chg"] = chg
    cols["x_runlen"] = run_lengths(decoded, half)
    cols.update(still_at_line_runs(table, half))
    cols.update(touch_chains(table, half))
    cols.update(gap_jump(table, half))
    return pd.DataFrame(cols)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-trees", type=int, default=400)
    args = ap.parse_args()
    n_trees = 40 if args.smoke else args.n_trees

    with open(os.path.join(core.CACHE, "stage1_oof.pkl"), "rb") as handle:
        oof_raw = pickle.load(handle)
    oof = {tuple(k.split("|")): v for k, v in oof_raw.items()}
    t0 = time.time()

    # use stage one models trained on tune matches to predict holdout matches
    frame_t = core.fit_frame(list(core.TUNE))
    change_t = core.fit_change(list(core.TUNE))

    # step 1 build stacker rows from out of fold cases
    print("1. building stacker rows from the Stage-A out-of-fold cases ...", flush=True)
    xs, ys, ws = [], [], []
    decoder_cases, holdout_cases = [], []
    for (mh, s), (p, c, half, truth, runs) in sorted(oof.items()):
        decoded = core.decode(p, c, half, json.load(open(os.path.join(OUT, "params_stage1.json"), encoding="utf-8"))["decoder"])
        table, _, _ = core.item(mh, s)
        st = stacker_table(table, p, c, decoded, half)
        keep = np.flatnonzero(truth != pm.X)[::core.STRIDE]
        xs.append(st.iloc[keep])
        ys.append(core.controlled(truth)[keep])
        ws.append(core.spell_weights(truth)[keep])
        decoder_cases.append((st, c, half, truth, runs))
    x = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    print("   {} rows x {} stacker features".format(len(x), x.shape[1]), flush=True)

    # step 2 fit the stacker and search its decoder on out of fold cases
    print("2. fitting the stacker and searching its decoder ...", flush=True)
    import lightgbm as lgb
    stack = lgb.train(core.LGB_PARAMS, lgb.Dataset(x, y, weight=w), num_boost_round=n_trees)
    s_cases = []
    for st, c, half, truth, runs in decoder_cases:
        p = stack.predict(st[st.columns].astype(np.float32))
        p = np.clip(p / p.sum(axis=1, keepdims=True), 1e-4, 1.0)
        s_cases.append((p, c, half, truth, runs))
    decoder, cv_spell = core.decoder_search(s_cases)
    print("   stacker decoder {} -> OOF spellF1 {:.4f}".format(decoder, cv_spell), flush=True)

    # step 3 validate stage one models and the stacker on holdout matches
    print("3. validating on the HOLDOUT matches ...", flush=True)
    h_cases = []
    for m in core.HOLDOUT:
        for s in core.JUDGE_SCENARIOS:
            table, truth, runs = core.item(m, s)
            swap_table = core.item(m, s, True)[0]
            p1, c1 = core.predict_probs(frame_t, change_t, table, swap_table)
            d1 = core.decode(p1, c1, half := table["half"].values,
                             json.load(open(os.path.join(OUT, "params_stage1.json"),
                                            encoding="utf-8"))["decoder"])
            st = stacker_table(table, p1, c1, d1, half)
            p2 = stack.predict(st[st.columns].astype(np.float32))
            p2 = np.clip(p2 / p2.sum(axis=1, keepdims=True), 1e-4, 1.0)
            h_cases.append((p2, c1, half, truth, runs))
    holdout = core.summary(h_cases, decoder)
    print("   HOLDOUT spellF1 {:.4f} / macroF1 {:.4f}".format(
        holdout["spellF1"], holdout["macroF1"]), flush=True)

    # step 4 save the model
    stack_path = os.path.join(OUT, "lp4_stack.txt")
    stack.save_model(stack_path)
    core.save_params(os.path.join(OUT, "params_stack.json"),
                     variant="Let's Play pipeline4 stage 2: stacker over Stage-A OOF "
                             "probabilities + long-context signals",
                     stack_features=list(x.columns),
                     model_files=[os.path.relpath(stack_path, core.ROOT)],
                     decoder=decoder,
                     n_trees=n_trees,
                     cv_spellF1=float(cv_spell),
                     holdout=holdout,
                     stage1_params="model/params_stage1.json",
                     trained_on="Stage-A out-of-fold rows of {} (open DFL / IDSSE, CC BY 4.0)".format(
                         ", ".join(core.TUNE)),
                     written=datetime.datetime.now().replace(microsecond=0).isoformat())
    print("   saved {} (+ params_stack.json)  [{:.0f} s total]".format(
        os.path.relpath(stack_path, core.ROOT), time.time() - t0), flush=True)

v1_blocks = core.half_runs

if __name__ == "__main__":
    main()