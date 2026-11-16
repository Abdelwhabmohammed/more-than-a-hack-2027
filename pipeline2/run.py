"""Attempt-2 end-to-end runner.

    cd more-than-a-hack-2027
    export MTAH_DATA_DIR=/path/to/dataset        # Windows: $env:MTAH_DATA_DIR="..."

    python -m pipeline2.run                      # default variant (v2)
    python -m pipeline2.run --variant v1         # attempt-1 signals + new segmenter
    python -m pipeline2.run --variant v2         # full data layer + new segmenter
    python -m pipeline2.run --variant v3         # v2 + broadcast fusion
    python -m pipeline2.run --variant v2 --score # also validate + score
    python -m pipeline2.run --lam 30 --method dp # override segmentation knobs
"""

import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR  # noqa: E402
N_FRAMES = GAMES[GAME]["n_video_frames"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
VARIANTS = ("v1", "v2", "v3")   # v4 is in pipeline3

def write_csv(path, labels):
    """write a frame,label submission CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path

def build_v2_signals(data_dir):
    from pipeline2.data_layer import load_core, build_signals
    core = load_core(data_dir=data_dir)
    return build_signals(core)

def build_v1_signals(data_dir):
    from pipeline.features import extract_features
    from pipeline2.evidence import signals_from_v1_features
    features, feature_names, ball_visible, ball_xy = extract_features(data_dir=data_dir)
    return signals_from_v1_features(features, feature_names, ball_visible, ball_xy)

def run_variant(variant="v2", data_dir=None, output_path=None, lam=30.0, method="dp",
                run_scorer=False, min_poss_s=0.8, min_dead_s=3.0):
    """build the label series for one variant and write the submission CSV."""
    from pipeline2.evidence import build_affinity, build_dead_evidence
    from pipeline2.segmentation import segment_match
    from pipeline2.postprocess2 import (clean_segments, control_gated_flips, segments_to_labels, label_stats)
    from pipeline.smart_tagging_prior import build_smart_tagging_prior

    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    t_total = time.time()
    print(f"variant {variant} | method={method} lam={lam}")

    # build signals
    print("[1/4] signals")
    t0 = time.time()
    if variant == "v1":
        sig = build_v1_signals(data_dir)
    else:
        sig = build_v2_signals(data_dir)
    print(f"      {time.time() - t0:.1f} s")

    # build evidence
    print("[2/4] evidence (affinity + dead)")
    t0 = time.time()
    st_labels, st_conf, _, _ = build_smart_tagging_prior(data_dir=data_dir)
    st_prior = (st_labels, st_conf) if variant != "v1" else None
    s = build_affinity(sig, smart_tagging_prior=st_prior)
    p_d = build_dead_evidence(sig)
    print(f"      {time.time() - t0:.1f} s")

    # fuse broadcast data for v3
    if variant == "v3":
        print("[3/4] broadcast fusion")
        t0 = time.time()
        from pipeline2.broadcast_align import fuse_broadcast
        fused = fuse_broadcast(sig, data_dir=data_dir)
        if fused is not None:
            sig = fused
            # rebuild evidence
            s = build_affinity(sig, smart_tagging_prior=st_prior)
            p_d = build_dead_evidence(sig)
            print(f"      fused ({time.time() - t0:.1f} s)")
        else:
            print("      alignment gate FAILED  falling back to v2 behaviour")

    # segment and clean labels
    print(f"[4/4] segmentation ({method}, lam={lam})")
    t0 = time.time()
    segs = segment_match(s, p_d, method=method, lam=lam)
    segs = clean_segments(segs, min_poss_s=min_poss_s, min_dead_s=min_dead_s)
    labels = segments_to_labels(segs, N_FRAMES)
    print(f"      {time.time() - t0:.1f} s")

    # report and write results
    label_stats(labels)
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, f"Lets_Play_{variant}.csv")
    write_csv(output_path, labels)
    print(f"wrote {output_path}  (total {time.time() - t_total:.1f} s)")

    if run_scorer:
        import subprocess
        val = os.path.join(ROOT, "evaluation", "validate_submission.py")
        print("\nvalidate ")
        subprocess.run([sys.executable, val, output_path], check=False)
        print("score")
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")], check=False)
    return labels

def main():
    ap = argparse.ArgumentParser(description="attempt-2 possession pipeline")
    ap.add_argument("variant", default="v2", choices=VARIANTS)
    ap.add_argument("data-dir", default=None)
    ap.add_argument("output", "-o", default=None)
    ap.add_argument("lam", type=float, default=30.0, help="DP boundary cost (nats); 30 = prior-calibrated default")
    ap.add_argument("method", default="dp", choices=("dp", "viterbi"))
    ap.add_argument("min-poss", type=float, default=0.8)
    ap.add_argument("min-dead", type=float, default=3.0)
    ap.add_argument("score", action="store_true")
    args = ap.parse_args()
    run_variant(variant=args.variant, data_dir=args.data_dir, output_path=args.output,
                lam=args.lam, method=args.method, run_scorer=args.score,
                min_poss_s=args.min_poss, min_dead_s=args.min_dead)

if __name__ == "__main__":
    main()