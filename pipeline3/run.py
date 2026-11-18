"""Pipeline 3 end-to-end runner: the pressing-aware variant (V4).

Reuses the shared infrastructure (data layer, DP segmenter, post-processing)
from pipeline2 and adds the pressing-aware evidence.  See docs/V4_pressing_aware.md
for the experiment's outcome: investigated, not shipped.

    cd more-than-a-hack-2027
    export MTAH_DATA_DIR=/path/to/dataset

    python -m pipeline3.run  # write evaluation/results/Lets_Play_v4.csv
    python -m pipeline3.run --score # also validate + run the official scorer
    python -m pipeline3.run --tti-veto 0 # ablation: discount + counter-press only
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR

N_FRAMES = GAMES[GAME]["n_video_frames"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

def run_v4(data_dir=None, output_path=None, lam=30.0, tti_veto=0.5, min_poss_s=0.8, min_dead_s=3.0, run_scorer=False):
    from pipeline2.data_layer import load_core, build_signals
    from pipeline2.segmentation import segment_match
    from pipeline2.postprocess2 import clean_segments, segments_to_labels, label_stats
    from pipeline2.run import write_csv
    from pipeline3.pressing import compute_pressing
    from pipeline3.evidence import build_affinity_v4
    from pipeline2.evidence import build_dead_evidence
    from pipeline.smart_tagging_prior import build_smart_tagging_prior

    data_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    t_total = time.time()
    print("=== variant v4 (pressing-aware) | lam=%.0f tti_veto=%.1f ===" % (lam, tti_veto))

    print("[1/3] signals + pressing extras")
    t0 = time.time()
    sig = build_signals(load_core(data_dir=data_dir), data_dir=data_dir)
    sig = compute_pressing(sig)
    print(f"      {time.time() - t0:.1f} s")

    print("[2/3] evidence + segmentation")
    t0 = time.time()
    st_labels, st_conf, _, _ = build_smart_tagging_prior(data_dir=data_dir)
    s = build_affinity_v4(sig, smart_tagging_prior=(st_labels, st_conf), tti_veto=tti_veto)
    p_d = build_dead_evidence(sig)
    segs = segment_match(s, p_d, method="dp", lam=lam)
    segs = clean_segments(segs, min_poss_s=min_poss_s, min_dead_s=min_dead_s)
    labels = segments_to_labels(segs, N_FRAMES)
    print(f"      {time.time() - t0:.1f} s")

    print("[3/3] report + write")
    label_stats(labels)
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "Lets_Play_v4.csv")
    write_csv(output_path, labels)
    print(f"wrote {output_path}  (total {time.time() - t_total:.1f} s)")

    if run_scorer:
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), output_path], check=False)
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")], check=False)
    return labels

def main():
    ap = argparse.ArgumentParser(description="pressing-aware possession pipeline (V4)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--tti-veto", type=float, default=0.5, help="0 disables the TTI veto (discount + counter-press only)")
    ap.add_argument("--min-poss", type=float, default=0.8)
    ap.add_argument("--min-dead", type=float, default=3.0)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    run_v4(data_dir=args.data_dir, output_path=args.output, lam=args.lam,
           tti_veto=args.tti_veto, min_poss_s=args.min_poss,
           min_dead_s=args.min_dead, run_scorer=args.score)

if __name__ == "__main__":
    main()