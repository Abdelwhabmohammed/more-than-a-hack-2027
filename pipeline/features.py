import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require

# ATD parsing constants
FIELD_SEP = ":"
GROUP_SEP = ";"
PITCH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
HOME, AWAY = 0, 1

# team assignment from ATD metadata
def team_of_each_track(meta_path):
    """Array over track slots: 0 = home, 1 = away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


# feature extraction

# Feature name constants - keep in sync with the extraction loop below.
FEATURE_NAMES = [
    # ball proximity (NaN when ball invisible)
    "nearest_home_dist",
    "nearest_away_dist",
    "dist_ratio",
    "home_within_5m",
    "away_within_5m",
    "home_within_10m",
    "away_within_10m",
    "ball_speed",
    "ball_x",
    "ball_y",

    # team shape
    "home_centroid_x",
    "home_centroid_y",
    "away_centroid_x",
    "away_centroid_y",
    "home_spread_x", # std of x positions, metres
    "home_spread_y",
    "away_spread_x",
    "away_spread_y",
    "n_home_tracked",
    "n_away_tracked",
    "n_total_tracked",

    # territorial
    "home_frac_attacking",
    "away_frac_attacking",
    "centroid_closer_to_ball",

    # velocity / dynamics
    "home_mean_speed",
    "away_mean_speed",

    # missing data
    "ball_visible",
    "frames_since_ball",
]
N_FEATURES = len(FEATURE_NAMES)

def _parse_group(group):
    # parse 'x,y,speed' into (x, y, speed) floats, or None if missing.
    if group[0] == "N":
        return None
    c1 = group.index(",")
    c2 = group.index(",", c1 + 1)
    return float(group[:c1]), float(group[c1 + 1:c2]), float(group[c2 + 1:])

def extract_features(data_dir=None, game=GAME):
    # extract per-frame features from the ATD.
    data_dir = data_dir or DATA_DIR
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    track_team = team_of_each_track(meta_path)
    n_slots = len(track_team)
    length, width = PITCH
    features = np.full((N_FRAMES, N_FEATURES), np.nan, dtype=np.float32)
    ball_xy = np.full((N_FRAMES, 2), np.nan, dtype=np.float64)
    ball_visible = np.zeros(N_FRAMES, dtype=bool)
    fi = {name: i for i, name in enumerate(FEATURE_NAMES)}
    frames_since_ball = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            vf = frame_id - 1
            if vf < 0 or vf >= N_FRAMES:
                continue

            # parse players
            home_xs, home_ys, home_speeds = [], [], []
            away_xs, away_ys, away_speeds = [], [], []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                parsed = _parse_group(group)
                if parsed is None:
                    continue
                px, py, spd = parsed
                if slot < n_slots:
                    if track_team[slot] == HOME:
                        home_xs.append(px)
                        home_ys.append(py)
                        home_speeds.append(spd)
                    else:
                        away_xs.append(px)
                        away_ys.append(py)
                        away_speeds.append(spd)
            n_home = len(home_xs)
            n_away = len(away_xs)
            n_total = n_home + n_away
            features[vf, fi["n_home_tracked"]] = n_home
            features[vf, fi["n_away_tracked"]] = n_away
            features[vf, fi["n_total_tracked"]] = n_total

            # team centroids and spreads
            if n_home > 0:
                hx_arr = np.array(home_xs)
                hy_arr = np.array(home_ys)
                features[vf, fi["home_centroid_x"]] = hx_arr.mean() * length
                features[vf, fi["home_centroid_y"]] = hy_arr.mean() * width
                features[vf, fi["home_spread_x"]] = hx_arr.std() * length if n_home > 1 else 0.0
                features[vf, fi["home_spread_y"]] = hy_arr.std() * width if n_home > 1 else 0.0
                features[vf, fi["home_frac_attacking"]] = float(np.sum(hx_arr > 0.5)) / n_home
                features[vf, fi["home_mean_speed"]] = np.mean(home_speeds)
            if n_away > 0:
                ax_arr = np.array(away_xs)
                ay_arr = np.array(away_ys)
                features[vf, fi["away_centroid_x"]] = ax_arr.mean() * length
                features[vf, fi["away_centroid_y"]] = ay_arr.mean() * width
                features[vf, fi["away_spread_x"]] = ax_arr.std() * length if n_away > 1 else 0.0
                features[vf, fi["away_spread_y"]] = ay_arr.std() * width if n_away > 1 else 0.0
                features[vf, fi["away_frac_attacking"]] = float(np.sum(ax_arr < 0.5)) / n_away
                features[vf, fi["away_mean_speed"]] = np.mean(away_speeds)

            # parse ball
            ball_parsed = _parse_group(parts[2].strip())
            if ball_parsed is not None:
                bx, by, bspd = ball_parsed
                ball_visible[vf] = True
                ball_xy[vf, 0] = bx
                ball_xy[vf, 1] = by
                features[vf, fi["ball_visible"]] = 1.0
                features[vf, fi["ball_x"]] = bx
                features[vf, fi["ball_y"]] = by
                features[vf, fi["ball_speed"]] = bspd
                frames_since_ball = 0

                # ball-proximity features
                bx_m = bx * length
                by_m = by * width
                best_home, best_away = 1e9, 1e9
                home_5, away_5 = 0, 0
                home_10, away_10 = 0, 0
                for px, py in zip(home_xs, home_ys):
                    d = np.sqrt((px * length - bx_m) ** 2 + (py * width - by_m) ** 2)
                    if d < best_home:
                        best_home = d
                    if d <= 5.0:
                        home_5 += 1
                    if d <= 10.0:
                        home_10 += 1
                for px, py in zip(away_xs, away_ys):
                    d = np.sqrt((px * length - bx_m) ** 2 + (py * width - by_m) ** 2)
                    if d < best_away:
                        best_away = d
                    if d <= 5.0:
                        away_5 += 1
                    if d <= 10.0:
                        away_10 += 1

                features[vf, fi["nearest_home_dist"]] = best_home
                features[vf, fi["nearest_away_dist"]] = best_away
                denom = best_home + best_away
                features[vf, fi["dist_ratio"]] = best_home / denom if denom > 0 else 0.5
                features[vf, fi["home_within_5m"]] = home_5
                features[vf, fi["away_within_5m"]] = away_5
                features[vf, fi["home_within_10m"]] = home_10
                features[vf, fi["away_within_10m"]] = away_10

                # which team centroid is closer to ball
                if n_home > 0 and n_away > 0:
                    hc_d = np.sqrt((features[vf, fi["home_centroid_x"]] - bx_m) ** 2 +(features[vf, fi["home_centroid_y"]] - by_m) ** 2)
                    ac_d = np.sqrt((features[vf, fi["away_centroid_x"]] - bx_m) ** 2 + (features[vf, fi["away_centroid_y"]] - by_m) ** 2)
                    features[vf, fi["centroid_closer_to_ball"]] = 1.0 if hc_d < ac_d else -1.0
            else:
                frames_since_ball += 1
                features[vf, fi["ball_visible"]] = 0.0
                features[vf, fi["centroid_closer_to_ball"]] = 0.0
            features[vf, fi["frames_since_ball"]] = frames_since_ball
    return features, list(FEATURE_NAMES), ball_visible, ball_xy

if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    print(f"extracting features from ATD in {data_dir} ...")
    feats, names, bvis, bxy = extract_features(data_dir=data_dir)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")
    print(f"feature matrix: {feats.shape} ({len(names)} features)")
    print(f"ball visible: {bvis.sum()} / {len(bvis)} frames ({100*bvis.mean():.1f}%)")
    print(f"\nper-feature non-NaN coverage:")
    for i, name in enumerate(names):
        col = feats[:, i]
        pct = 100.0 * np.sum(~np.isnan(col)) / len(col)
        print(f"  {name:<28s}  {pct:6.1f}%")