"""Train the shipped DFL frame model (frame classifier + change detector).

This is the model behind the current `Lets_Play.csv`: LightGBM on the seven open
DFL/IDSSE matches (CC BY 4.0), degraded by the fitted simulator over the two
scenarios matched to the challenge input, both sides, spell-length weights, and
a time-varying Viterbi whose prices are tuned on the two HOLDOUT matches with
the organisers' own scorer code.

    cd more-than-a-hack-2027
    python -m pipeline4.train_dfl # full protocol
    python -m pipeline4.train_dfl --skip-features # reuse cached feature tables
"""

import argparse
import datetime
import json
import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import pipeline4.core as core

MODEL_DIR = core.MODEL_DIR
SCENARIOS = ("matched_speed", "matched_dead")  # the two matched scenarios

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-features", action="store_true", help="reuse cached tables")
    ap.add_argument("--n-trees", type=int, default=core.N_TREES)
    args = ap.parse_args()
    n_trees = args.n_trees
    os.makedirs(core.CACHE, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()

    if not args.skip_features:
        print("1. feature tables: 7 matches x {} scenarios x both sides ...".format(len(SCENARIOS)),
              flush=True)
        for m in core.ALL_MATCHES:
            t1 = time.time()
            for s in SCENARIOS:
                core.item(m, s, False)
                core.item(m, s, True)
            print("   {} ready in {:.0f} s".format(m, time.time() - t1), flush=True)

    print("2. fitting frame + change models on {} ...".format(", ".join(core.TUNE)), flush=True)
    t1 = time.time()
    frame = core.fit_frame(list(core.TUNE), n_trees=n_trees, scenarios=SCENARIOS)
    change = core.fit_change(list(core.TUNE), n_trees=n_trees, scenarios=SCENARIOS)
    print("   trained in {:.0f} s".format(time.time() - t1), flush=True)

    print("3. decoder search on the HOLDOUT matches {} ...".format(", ".join(core.HOLDOUT)),
          flush=True)
    cases = []
    for m in core.HOLDOUT:
        for s in SCENARIOS:
            table, truth, runs = core.item(m, s)
            swap_table = core.item(m, s, True)[0]
            p, c = core.predict_probs(frame, change, table, swap_table)
            cases.append((p, c, table["half"].values, truth, runs))
    decoder, holdout_spell = core.decoder_search(cases)
    print("   decoder {} -> holdout spellF1 {:.4f}".format(decoder, holdout_spell), flush=True)

    print("4. final models on all 7 matches ...", flush=True)
    frame_final = core.fit_frame(core.ALL_MATCHES, n_trees=n_trees, scenarios=SCENARIOS)
    change_final = core.fit_change(core.ALL_MATCHES, n_trees=n_trees, scenarios=SCENARIOS)
    frame_path = os.path.join(MODEL_DIR, "lp_dfl_frame.txt")
    change_path = os.path.join(MODEL_DIR, "lp_dfl_change.txt")
    frame_final.save_model(frame_path)
    change_final.save_model(change_path)
    names = core.feature_names(core.item(core.TUNE[0], SCENARIOS[0])[0])
    core.save_params(
        os.path.join(MODEL_DIR, "params_dfl.json"),
        variant="Let's Play pipeline4: LightGBM frame model + change detector on IDSSE/DFL; ",
        features=names,
        model_files=[os.path.relpath(frame_path, ROOT), os.path.relpath(change_path, ROOT)],
        decoder=decoder,
        n_trees=n_trees,
        scenarios=list(SCENARIOS),
        trained_on="open DFL / IDSSE matches {} (CC BY 4.0), scenarios {}, both sides".format(
            ", ".join(core.ALL_MATCHES), ", ".join(SCENARIOS)),
        decoder_tuned_on="the two HOLDOUT DFL matches ({}), challenge scorer code".format(
            ", ".join(core.HOLDOUT)),
        holdout_spellF1=float(holdout_spell),
        written=datetime.datetime.now().replace(microsecond=0).isoformat())
    print("   saved {} + {} (+ params_dfl.json)  [{:.0f} s total]".format(
        os.path.relpath(frame_path, ROOT), os.path.relpath(change_path, ROOT), time.time() - t0),
        flush=True)

if __name__ == "__main__":
    main()