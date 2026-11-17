"""Predict BAR-ATH with the shipped DFL model (frame + change) and write a CSV.

    cd more-than-a-hack-2027
    export MTAH_DATA_DIR=/path/to/dataset
    python -m pipeline4.predict                       # -> evaluation/results/Lets_Play_dfl.csv
    python -m pipeline4.predict --score               # also validate + run the official scorer
"""

import argparse
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

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "Lets_Play_dfl.csv"))
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    with open(os.path.join(core.MODEL_DIR, "params_dfl.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    import lightgbm as lgb
    frame = lgb.Booster(model_file=os.path.join(ROOT, params["model_files"][0]))
    change = lgb.Booster(model_file=os.path.join(ROOT, params["model_files"][1]))

    table, table_swap = core.challenge_tables()
    missing = [c for c in params["features"] if c not in table.columns]
    if missing:
        raise SystemExit("feature table lacks columns the model needs: {}".format(missing[:5]))
    prob, chg = core.predict_probs(frame, change, table, table_swap, names=list(params["features"]))

    # second half is graded from 110808 but kicks off at 113742 so it is all dead
    before_kickoff = (table["half"].values == 2) & \
        (table["frame"].values < core.SECOND_HALF_KICKOFF)
    prob[before_kickoff] = [1e-4, 1e-4, 1.0]
    chg[before_kickoff] = 1e-4

    labels = np.array(pm.LABELS)[core.decode(prob, chg, table["half"].values, params["decoder"])]
    core.write_submission(args.out, labels)
    problem = pm.validate_submission(pm.read_label_csv(args.out), feats.N_FRAMES)
    print("written {}  ({})  [{:.0f} s]".format(
        os.path.relpath(args.out, ROOT), problem or "valid submission", time.time() - t0))
    graded = labels != "X"
    count, mean_s = core.possession_count(labels)
    print("graded frames: H {:.0%}  A {:.0%}  D {:.0%}   possessions {} (mean {:.1f} s)".format(
        (labels[graded] == "H").mean(), (labels[graded] == "A").mean(),
        (labels[graded] == "D").mean(), count, mean_s))

    if args.score:
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), args.out], check=False)
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")], check=False)

import possession_metrics as pm
import pipeline4.features as feats

if __name__ == "__main__":
    main()
