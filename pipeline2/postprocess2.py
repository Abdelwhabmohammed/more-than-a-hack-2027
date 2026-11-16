"""Segment-level cleanup for the attempt-2 pipeline"""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES

FPS = GAMES[GAME]["fps"]
H, A, D = 0, 1, 2

# football prior durations in seconds
MIN_POSS_S = 0.8
MIN_DEAD_S = 3.0

def _merge_adjacent(segs):
    out = []
    for a, b, l in segs:
        if out and out[-1][2] == l:
            out[-1] = (out[-1][0], b, l)
        else:
            out.append((a, b, l))
    return out

def clean_segments(segs, min_poss_s=MIN_POSS_S, min_dead_s=MIN_DEAD_S):
    """Light cleanup. segs: [(start, stop, label)]. Returns cleaned segments."""
    segs = _merge_adjacent(segs)
    min_poss = int(min_poss_s * FPS)
    min_dead = int(min_dead_s * FPS)
    for _ in range(6):
        changed = False
        # merge short possessions
        i = 0
        while i < len(segs):
            a, b, l = segs[i]
            if l in (H, A) and (b - a) < min_poss and len(segs) > 1:
                prev_l = segs[i - 1][2] if i > 0 else None
                next_l = segs[i + 1][2] if i + 1 < len(segs) else None
                if prev_l == next_l is not None:
                    target = i - 1
                elif prev_l is None:
                    target = i + 1
                elif next_l is None:
                    target = i - 1
                else:
                    prev_len = segs[i - 1][1] - segs[i - 1][0]
                    next_len = segs[i + 1][1] - segs[i + 1][0]
                    target = i - 1 if prev_len >= next_len else i + 1
                ta, tb, tl = segs[target]
                segs[target] = (min(ta, a), max(tb, b), tl)
                del segs[i]
                changed = True
                continue
            i += 1
        # merge short dead stretches
        i = 1
        while i < len(segs) - 1:
            a, b, l = segs[i]
            if l == D and (b - a) < min_dead and \
                    segs[i - 1][2] == segs[i + 1][2] and segs[i - 1][2] in (H, A):
                segs[i - 1] = (segs[i - 1][0], segs[i + 1][1], segs[i - 1][2])
                del segs[i:i + 2]
                changed = True
                continue
            i += 1

        segs = _merge_adjacent(segs)
        if not changed:
            break
    return segs

def _causal_control(sig, team, dist_m=1.8, rel_mps=2.5, sustain=3, lookback=5):
    """causal per-frame 'this team controls the ball' indicator"""
    d = sig["d_home"] if team == H else sig["d_away"]
    rel = sig["rel_speed_home"] if team == H else sig["rel_speed_away"]
    near = sig["ball_visible"] & np.isfinite(d) & (d < dist_m) & \
        (np.nan_to_num(rel, nan=99.0) < rel_mps)
    # count control frames
    cnt = np.convolve(near.astype(np.int32), np.ones(lookback, dtype=np.int32),
                      mode="full")[:len(near)]
    # trailing count for each frame
    return cnt >= sustain

def control_gated_flips(segs, sig, snap_back_s=1.0, snap_fwd_s=2.0,
                        revert_ctrl_frac=0.0, revert_max_s=3.0):
    """make possession changes control-gated; revert flips nobody controlled"""
    out = [list(s) for s in segs]
    n = sig["ball_visible"].shape[0]
    ctrl = {H: _causal_control(sig, H), A: _causal_control(sig, A)}
    revert_max = int(revert_max_s * FPS)
    i = 1
    while i < len(out):
        pa, pb, pl = out[i - 1]
        ca, cb, cl = out[i]
        if cl in (H, A) and pl in (H, A):
            back = max(0, ca - int(snap_back_s * FPS))
            fwd = min(n, ca + int(snap_fwd_s * FPS))
            snap = next((f for f in range(back, fwd) if ctrl[cl][f]), None)
            if snap is not None and snap != ca:
                # move the boundary
                new_start = max(pa + 1, min(snap, cb - 1))
                if pa < new_start < cb:
                    out[i - 1] = (pa, new_start, pl)
                    out[i] = (new_start, cb, cl)
            else:
                # check for control inside the segment
                seg_ctrl = ctrl[cl][ca:cb]
                ctrl_frac = float(seg_ctrl.mean()) if len(seg_ctrl) else 0.0
                if ctrl_frac < revert_ctrl_frac and (cb - ca) < revert_max:
                    if i + 1 < len(out) and out[i + 1][2] == pl:
                        out[i - 1] = (pa, out[i + 1][1], pl)
                        del out[i:i + 2]
                    else:
                        out[i - 1] = (pa, cb, pl)
                        del out[i]
                    continue
        i += 1

    # remove empty segments and merge neighbors
    out = [s for s in out if s[1] > s[0]]
    return [tuple(s) for s in _merge_adjacent(out)]

def segments_to_labels(segs, n):
    labels = np.full(n, "X", dtype="<U1")
    for start, stop, lab in segs:
        labels[start:stop] = ("H", "A", "D")[lab]
    return labels

def label_stats(labels, halves=None):
    """Quick structural report: label counts and segment structure."""
    if halves is None:
        from games import half_video_blocks
        halves = half_video_blocks(GAME)
    for lab in ("H", "A", "D", "X"):
        print(f"  {lab}: {int(np.sum(labels == lab))} frames")
    # count possessions by half
    for _, ff, lf in halves:
        window = labels[ff:lf + 1]
        changes = int(np.sum(window[1:] != window[:-1])) + 1
        poss = 0
        prev = None
        for v in window:
            if v in ("H", "A") and v != prev:
                poss += 1
            prev = v
        print(f"  half {ff}-{lf}: {changes} segments, {poss} possessions")

if __name__ == "__main__":
    print("module import check: ok")
