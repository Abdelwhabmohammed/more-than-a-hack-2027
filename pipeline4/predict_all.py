"""Predict BAR-ATH with every trained pipeline4 candidate and write CSVs.

Candidates (whichever artifacts exist):
  stage1 frame + change models -> Lets_Play_p4_s1.csv
  stack Stage-B stacker over Stage-A outputs -> Lets_Play_p4_stack.csv
  gru GRU ensemble with Stage-A + stacker -> Lets_Play_p4_full.csv

Also prints the one-shot public-window score per candidate (correctness check
only -- candidate selection uses the DFL holdout numbers recorded at training).

    cd more-than-a-hack-2027
    export MTAH_DATA_DIR=/path/to/dataset
    python -m pipeline4.predict_all
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pipeline4.core as core
from pipeline4 import core
from pipeline4.train_stack import stacker_table
from pipeline4.train_stage1 import OUT

RESULTS = os.path.join(core.ROOT, "evaluation", "results")

def load_booster(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)

def write(labels, name):
    out = os.path.join(RESULTS, name)
    core.write_submission(out, labels)
    problem = pm.validate_submission(pm.read_label_csv(out), feats.N_FRAMES)
    print("  written {}  ({})".format(os.path.relpath(out, core.ROOT), problem or "valid"))
    return out

def score_once(path):
    """One-shot public-window read (correctness check, never a selector)."""
    gt = pm.label_codes(pm.read_label_csv(os.path.join(core.ROOT, "evaluation", "gt",
                                                       "BAR-ATH_possession_gt_public.csv")))
    pred = pm.label_codes(pm.read_label_csv(path))
    r = pm.evaluate(gt, pred)
    print("  public check: spellF1 {:.4f}  macroF1 {:.4f}  poss {} (gt 17)  dur {:.1f} s".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"]))
    return r

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true", help="print the one-shot public check")
    ap.add_argument("--which", nargs="*", default=("stage1", "stack", "gru"),
                    help="candidates to produce")
    args = ap.parse_args()
    t0 = time.time()

    s1_path = os.path.join(OUT, "params_stage1.json")
    if not os.path.isfile(s1_path):
        raise SystemExit("no Stage-A artifacts: run python -m pipeline4.train_stage1 first")
    params1 = json.load(open(s1_path, encoding="utf-8"))
    frame = load_booster(os.path.join(core.ROOT, params1["model_files"][0]))
    change = load_booster(os.path.join(core.ROOT, params1["model_files"][1]))
    print("challenge feature tables (cached after the first build) ...", flush=True)
    table, table_swap = core.challenge_tables()
    half = table["half"].values
    p1, c1 = core.predict_probs(frame, change, table, table_swap)

    # second half is graded from 110808 but kicks off at 113742 so it is all dead
    before = (half == 2) & (table["frame"].values < core.SECOND_HALF_KICKOFF)
    p1[before] = [1e-4, 1e-4, 1.0]
    c1[before] = 1e-4

    d1 = core.decode(p1, c1, half, params1["decoder"])
    if "stage1" in args.which:
        print("stage1 candidate:")
        write(np.array(pm.LABELS)[d1], "Lets_Play_p4_s1.csv")

    stack_params_path = os.path.join(OUT, "params_stack.json")
    if "stack" in args.which and os.path.isfile(stack_params_path):
        print("stack candidate:")
        ps = json.load(open(stack_params_path, encoding="utf-8"))
        stack = load_booster(os.path.join(core.ROOT, ps["model_files"][0]))
        st = stacker_table(table, p1, c1, d1, half)
        p2 = stack.predict(st[st.columns].astype(np.float32))
        p2 = np.clip(p2 / p2.sum(axis=1, keepdims=True), 1e-4, 1.0)
        d2 = core.decode(p2, c1, half, ps["decoder"])
        write(np.array(pm.LABELS)[d2], "Lets_Play_p4_stack.csv")

    gru_params_path = os.path.join(OUT, "params_gru.json")
    if "gru" in args.which and os.path.isfile(gru_params_path):
        print("gru / ensemble candidate:")
        pg = json.load(open(gru_params_path, encoding="utf-8"))
        import torch
        from pipeline4.train_gru import gru_probs, build_sequences
        saved = torch.load(os.path.join(core.ROOT, pg["model_file"]),
                           map_location="cpu", weights_only=False)
        import torch.nn as nn
        model = nn.GRU(saved["n_in"], saved["hidden"], num_layers=saved["layers"],
                       bidirectional=True, batch_first=True)
        head = nn.Linear(2 * saved["hidden"], 3)
        model.load_state_dict(saved["state"])
        head.load_state_dict(saved["head"])
        # standardise the challenge table with the GRU training scaler
        names = saved["feature_names"]
        mu = np.array(saved["scaler_mu"], dtype=np.float32)
        sd = np.array(saved["scaler_sd"], dtype=np.float32)
        seq = np.nan_to_num((table[names].values.astype(np.float32) - mu) / sd)[None]
        pg_prob = gru_probs(model, head, [seq[0]])[0]
        # average with stacker probabilities when available, otherwise use stage 1
        if "stack" in args.which and os.path.isfile(stack_params_path):
            ps = json.load(open(stack_params_path, encoding="utf-8"))
            stack = load_booster(os.path.join(core.ROOT, ps["model_files"][0]))
            st = stacker_table(table, p1, c1, d1, half)
            p2 = stack.predict(st[st.columns].astype(np.float32))
            p2 = np.clip(p2 / p2.sum(axis=1, keepdims=True), 1e-4, 1.0)
            base = p2
            dec = ps["decoder"]
        else:
            base = p1
            dec = params1["decoder"]
        p3 = np.clip((base + pg_prob) / 2, 1e-4, 1.0)
        d3 = core.decode(p3, c1, half, dec)
        write(np.array(pm.LABELS)[d3], "Lets_Play_p4_full.csv")

    if args.score:
        for name in sorted(os.listdir(RESULTS)):
            if name.startswith("Lets_Play_p4"):
                print(name + ":")
                score_once(os.path.join(RESULTS, name))
    print("[{:.0f} s]".format(time.time() - t0), flush=True)


import possession_metrics as pm  # noqa: E402
import pipeline4.features as feats  # noqa: E402

if __name__ == "__main__":
    main()
