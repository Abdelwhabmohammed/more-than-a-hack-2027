import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "official" / "more-than-a-hack-2027"
STARTER = OFFICIAL / "starter"

os.environ["MTAH_DATA_DIR"] = str(ROOT / "data" / "raw")
for p in (OFFICIAL, STARTER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

N_FRAMES = 199319
PITCH_L, PITCH_W = 105.0, 68.0   # ASSUMPTION: verify against metadata

# Physical parameters (NOT tuned on ground truth)
CONTROL_RADIUS_M = 3.0
MARGIN_M = 0.5
MAX_GAP = 75        # 3 s at 25 fps
MIN_SPELL = 12      # ~0.5 s
OUT_TOL = 0.02      # ball this far outside [0,1] => dead ball

H, A, D, UNK = 0, 1, 2, -1


def _runs(a):
    cut = np.flatnonzero(np.diff(a)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(a)]
    return starts, ends


def raw_possession(df, track_teams):
    ids = sorted(int(c.split("_")[0]) for c in df.columns
                 if c[0].isdigit() and c.endswith("_x"))
    assert len(ids) == len(track_teams) == 46, (len(ids), len(track_teams))

    X = df[[f"{i}_x" for i in ids]].to_numpy(dtype=float)
    Y = df[[f"{i}_y" for i in ids]].to_numpy(dtype=float)
    bx = df["ball_x"].to_numpy(dtype=float)
    by = df["ball_y"].to_numpy(dtype=float)

    dx = (X - bx[:, None]) * PITCH_L
    dy = (Y - by[:, None]) * PITCH_W
    dist = np.sqrt(dx ** 2 + dy ** 2)
    dist[np.isnan(dist)] = np.inf

    team = np.asarray(track_teams)
    d_home = dist[:, team == 0].min(axis=1)
    d_away = dist[:, team == 1].min(axis=1)

    codes = np.full(len(df), UNK, dtype=int)
    home_ok = (d_home <= CONTROL_RADIUS_M) & (d_home < d_away - MARGIN_M)
    away_ok = (d_away <= CONTROL_RADIUS_M) & (d_away < d_home - MARGIN_M)
    codes[home_ok] = H
    codes[away_ok] = A

    ball_ok = ~np.isnan(bx) & ~np.isnan(by)
    outside = ball_ok & ((bx < -OUT_TOL) | (bx > 1 + OUT_TOL) |
                         (by < -OUT_TOL) | (by > 1 + OUT_TOL))
    codes[outside] = D
    return codes


def fill_gaps(codes, max_gap=MAX_GAP):
    codes = codes.copy()
    n = len(codes)
    starts, ends = _runs(codes)
    for s, e in zip(starts, ends):
        if codes[s] != UNK:
            continue
        left = codes[s - 1] if s > 0 else None
        right = codes[e] if e < n else None
        length = e - s
        if length > max_gap or left not in (H, A) or right not in (H, A):
            codes[s:e] = D
        elif left == right:
            codes[s:e] = left
        else:
            mid = s + length // 2
            codes[s:mid] = left
            codes[mid:e] = right
    return codes


def remove_short_spells(codes, min_len=MIN_SPELL, passes=2):
    codes = codes.copy()
    for _ in range(passes):
        starts, ends = _runs(codes)
        for k, (s, e) in enumerate(zip(starts, ends)):
            if codes[s] in (H, A) and (e - s) < min_len:
                if k > 0:
                    codes[s:e] = codes[starts[k - 1]]
                elif k + 1 < len(starts):
                    codes[s:e] = codes[starts[k + 1]]
                else:
                    codes[s:e] = D
    return codes


def to_labels(codes):
    return np.array(["H", "A", "D"])[codes]


def generate_v2_labels(df, track_teams):
    codes = raw_possession(df, track_teams)
    codes = fill_gaps(codes)
    codes = remove_short_spells(codes)
    assert set(np.unique(codes)).issubset({H, A, D})
    return to_labels(codes)


def main():
    from games import asset_paths
    from kloppy import metrica
    from baseline import team_of_each_track

    paths = asset_paths()
    dataset = metrica.load_tracking_epts(
        meta_data=paths["atd_meta"], raw_data=paths["atd_raw"]
    )
    df = dataset.to_df()
    assert len(df) == N_FRAMES, len(df)
    assert (df["frame_id"].to_numpy() == np.arange(1, N_FRAMES + 1)).all()

    track_teams = team_of_each_track(paths["atd_meta"])
    labels = to_labels_wrapper(df, track_teams)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "team_submission_v2.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(
        out, index=False, encoding="utf-8"
    )
    print("Saved:", out)
    print(pd.Series(labels).value_counts())

    from evaluation import validate_submission
    validate_submission(out)


def to_labels_wrapper(df, track_teams):
    return generate_v2_labels(df, track_teams)


if __name__ == "__main__":
    main()