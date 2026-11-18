"""Stage A: the v4w-equivalent frame + change models, with out-of-fold probabilities.

    cd more-than-a-hack-2027
    python -m pipeline4.train_stage1 # full protocol
    python -m pipeline4.train_stage1 --smoke # tiny settings, to test the plumbing

Protocol:

  1. v3 feature tables (touch/control signals) for the seven DFL matches, all four
     degradation scenarios, both sides, with the drift ball model and the
     realistic dead-ball model (cached; the slow first pass);
  2. leave-one-match-out over the five TUNE matches: for each fold, frame model
     (spell-length-weighted) + change model trained on the other four, then
     out-of-fold probabilities on the held match -- saved for the Stage-B stacker;
  3. decoder prices searched on the TUNE out-of-fold cases (matched scenarios,
     controlled-possession-rewritten truth), validated on the two HOLDOUT matches;
  4. final frame + change models refit on all seven matches.

Nothing here reads the challenge match.
"""

import argparse
import datetime
import json
import os
import pickle
import time
import numpy as np
import pipeline4.core as core
from pipeline4 import core

OUT = os.path.join(core.MODEL_DIR)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="tiny settings, plumbing test only")
    ap.add_argument("--skip-features", action="store_true", help="reuse cached tables")
    ap.add_argument("--n-trees", type=int, default=core.N_TREES)
    args = ap.parse_args()
    n_trees = 40 if args.smoke else args.n_trees
    scenarios = core.JUDGE_SCENARIOS if args.smoke else core.SCENARIOS

    os.makedirs(core.CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # step 1 feature tables
    if not args.skip_features:
        print("1. feature tables: 7 matches x {} scenarios x both sides ...".format(
            len(scenarios)), flush=True)
        for m in core.ALL_MATCHES:
            t1 = time.time()
            for s in scenarios:
                core.item(m, s, False)
                core.item(m, s, True)
            print("   {} ready in {:.0f} s".format(m, time.time() - t1), flush=True)

    # step 2 leave one match out to get out of fold probabilities
    print("2. leave-one-match-out over {} ...".format(", ".join(core.TUNE)), flush=True)
    oof = {}
    for held in core.TUNE:
        t1 = time.time()
        train = [m for m in core.TUNE if m != held]
        frame = core.fit_frame(train, n_trees=n_trees, scenarios=scenarios)
        change = core.fit_change(train, n_trees=n_trees, scenarios=scenarios)
        for s in core.JUDGE_SCENARIOS:
            table, truth, runs = core.item(held, s)
            swap_table = core.item(held, s, True)[0]
            p, c = core.predict_probs(frame, change, table, swap_table)
            oof[(held, s)] = (p, c, table["half"].values, truth, runs)
        print("   fold {} done in {:.0f} s".format(held, time.time() - t1), flush=True)
    with open(os.path.join(core.CACHE, "stage1_oof.pkl"), "wb") as handle:
        pickle.dump({ "{}|{}".format(k[0], k[1]): v for k, v in oof.items() }, handle)

    # step 3 search decoder prices on tune out of fold data and validate on holdout
    print("3. decoder search on the out-of-fold cases ...", flush=True)
    oof_cases = [v for (mh, s), v in oof.items() if s in core.JUDGE_SCENARIOS]
    decoder, cv_spell = core.decoder_search(oof_cases)
    print("   decoder {} -> TUNE-OOF spellF1 {:.4f}".format(decoder, cv_spell), flush=True)

    print("   validating on the HOLDOUT matches {} (final TUNE models)...".format(
        ", ".join(core.HOLDOUT)), flush=True)
    frame_t = core.fit_frame(list(core.TUNE), n_trees=n_trees, scenarios=scenarios)
    change_t = core.fit_change(list(core.TUNE), n_trees=n_trees, scenarios=scenarios)
    holdout_cases = []
    for m in core.HOLDOUT:
        for s in core.JUDGE_SCENARIOS:
            table, truth, runs = core.item(m, s)
            swap_table = core.item(m, s, True)[0]
            p, c = core.predict_probs(frame_t, change_t, table, swap_table)
            holdout_cases.append((p, c, table["half"].values, truth, runs))
    holdout = core.summary(holdout_cases, decoder)
    print("   HOLDOUT spellF1 {:.4f} / macroF1 {:.4f}".format(
        holdout["spellF1"], holdout["macroF1"]), flush=True)

    # step 4 final models on all seven matches
    print("4. final models on all 7 matches ...", flush=True)
    frame_f = core.fit_frame(core.ALL_MATCHES, n_trees=n_trees, scenarios=scenarios)
    change_f = core.fit_change(core.ALL_MATCHES, n_trees=n_trees, scenarios=scenarios)
    frame_path = os.path.join(OUT, "lp4_frame.txt")
    change_path = os.path.join(OUT, "lp4_change.txt")
    frame_f.save_model(frame_path)
    change_f.save_model(change_path)
    names = core.feature_names(core.item(core.TUNE[0], core.JUDGE_SCENARIOS[0])[0])
    core.save_params(
        os.path.join(OUT, "params_stage1.json"),
        variant="Let's Play pipeline4 stage 1: LightGBM frame + change on IDSSE/DFL, ",
        features=names,
        model_files=[os.path.relpath(frame_path, core.ROOT), os.path.relpath(change_path, core.ROOT)],
        decoder=decoder,
        n_trees=n_trees,
        scenarios=list(scenarios),
        judge_scenarios=list(core.JUDGE_SCENARIOS),
        cv_spellF1=float(cv_spell),
        holdout=holdout,
        trained_on="open DFL / IDSSE matches {} (CC BY 4.0), scenarios {}, both sides, "
                   "ball model drift, dead model real".format(", ".join(core.ALL_MATCHES), ", ".join(scenarios)),
        decoder_tuned_on="leave-one-match-out probabilities on {} (matched scenarios), "
                         "controlled-possession-rewritten truth".format(", ".join(core.TUNE)),
        written=datetime.datetime.now().replace(microsecond=0).isoformat())
    print("   saved {} + {} (+ params_stage1.json)  [{:.0f} s total]".format(
        os.path.relpath(frame_path, core.ROOT), os.path.relpath(change_path, core.ROOT),
        time.time() - t0), flush=True)

if __name__ == "__main__":
    main()