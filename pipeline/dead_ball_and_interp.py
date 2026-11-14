import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]

# ball interpolation
def interpolate_ball(ball_xy, ball_visible):
    # filling short and medium gaps in ball position with linear interpolation.
    n = len(ball_xy)
    interp_xy = ball_xy.copy()
    confidence = np.where(ball_visible, 1.0, 0.0).astype(np.float32)
    MAX_SHORT = 50
    MAX_MEDIUM = 250
    MAX_INTERP = MAX_MEDIUM

    # find gaps, runs of invisible ball
    visible_indices = np.flatnonzero(ball_visible)
    if len(visible_indices) < 2:
        return interp_xy, confidence
    for i in range(len(visible_indices) - 1):
        idx_start = visible_indices[i]
        idx_end = visible_indices[i + 1]
        gap = idx_end - idx_start - 1
        if gap <= 0 or gap > MAX_INTERP:
            continue

        # linear interpolation
        x0, y0 = ball_xy[idx_start]
        x1, y1 = ball_xy[idx_end]
        for j in range(1, gap + 1):
            t = j / (gap + 1)
            f = idx_start + j
            interp_xy[f, 0] = x0 + t * (x1 - x0)
            interp_xy[f, 1] = y0 + t * (y1 - y0)

            # clamp to pitch boundaries with small margin
            interp_xy[f, 0] = np.clip(interp_xy[f, 0], -0.05, 1.05)
            interp_xy[f, 1] = np.clip(interp_xy[f, 1], -0.05, 1.05)

            # confidence decays with gap length
            if gap <= MAX_SHORT:
                confidence[f] = 0.8 - 0.3 * (gap / MAX_SHORT)
            else:
                confidence[f] = 0.4 - 0.3 * ((gap - MAX_SHORT) / (MAX_MEDIUM - MAX_SHORT))
            confidence[f] = max(confidence[f], 0.1)

    return interp_xy, confidence

# dead-ball detection
def detect_dead_ball(features, ball_visible, ball_xy, interp_xy, interp_conf):
    # score each frame for dead-ball likelihood using tracking signals.
    from pipeline.features import FEATURE_NAMES
    fi = {name: i for i, name in enumerate(FEATURE_NAMES)}
    n = len(features)
    signals = np.zeros((n, 6), dtype=np.float32)

    # signal_1: low player count (halftime, before/after match, water break)
    n_tracked = features[:, fi["n_total_tracked"]]
    n_tracked_clean = np.nan_to_num(n_tracked, nan=0.0)
    signals[:, 0] = np.clip(1.0 - n_tracked_clean / 10.0, 0, 1)

    # signal_2: ball out of bounds
    for f in range(n):
        if ball_visible[f]:
            bx, by = ball_xy[f]
            if bx < -0.02 or bx > 1.02 or by < -0.02 or by > 1.02:
                signals[f, 1] = 0.8
        elif interp_conf[f] > 0.3:
            bx, by = interp_xy[f]
            if bx < -0.02 or bx > 1.02 or by < -0.02 or by > 1.02:
                signals[f, 1] = 0.5

    # signal_3: ball stationary (low speed)
    ball_speed = features[:, fi["ball_speed"]]
    # rolling window: mean ball speed over 1 second
    window = FPS
    speed_clean = np.nan_to_num(ball_speed, nan=-1.0)
    # only compute where ball is visible
    for f in range(n):
        if speed_clean[f] >= 0:
            lo = max(0, f - window // 2)
            hi = min(n, f + window // 2 + 1)
            chunk = speed_clean[lo:hi]
            valid = chunk[chunk >= 0]
            if len(valid) > 5:
                mean_spd = valid.mean()
                if mean_spd < 0.5:  # very slow ball
                    signals[f, 2] = 0.6

    # signal_4: low team speeds (players walking, not playing)
    home_speed = np.nan_to_num(features[:, fi["home_mean_speed"]], nan=0.0)
    away_speed = np.nan_to_num(features[:, fi["away_mean_speed"]], nan=0.0)
    # smooth with a 2-second rolling mean
    kernel = np.ones(FPS * 2) / (FPS * 2)
    home_speed_smooth = np.convolve(home_speed, kernel, mode="same")
    away_speed_smooth = np.convolve(away_speed, kernel, mode="same")
    combined_speed = (home_speed_smooth + away_speed_smooth) / 2
    # very low combined speed likely dead ball
    signals[:, 3] = np.clip(1.0 - combined_speed / 1.5, 0, 1) * 0.5

    # signal_5: ball trajectory jump (teleportation = was out of play)
    length, width = PITCH
    for f in range(1, n):
        if ball_visible[f] and ball_visible[f - 1]:
            dx = (ball_xy[f, 0] - ball_xy[f - 1, 0]) * length
            dy = (ball_xy[f, 1] - ball_xy[f - 1, 1]) * width
            jump = np.sqrt(dx**2 + dy**2)
            if jump > 15.0:
                for j in range(max(0, f - FPS), min(n, f + 5)):
                    signals[j, 4] = max(signals[j, 4], 0.7)

    # signal_6: long gap in ball visibility
    frames_since = features[:, fi["frames_since_ball"]]
    frames_since_clean = np.nan_to_num(frames_since, nan=0.0)
    # if ball has been invisible for > 3 seconds, more likely dead
    signals[:, 5] = np.clip((frames_since_clean - FPS * 2) / (FPS * 5), 0, 1) * 0.3

    # Combine signals using maximum of complementary indicators + low speed bonus
    s_halftime = np.clip(1.0 - n_tracked_clean / 10.0, 0, 1)

    # out of bounds (near touchline/goal line or beyond)
    s_oob = np.zeros(n, dtype=np.float32)
    for f in range(n):
        if ball_visible[f]:
            bx, by = ball_xy[f]
            if bx < 0.01 or bx > 0.99 or by < 0.015 or by > 0.985:
                s_oob[f] = 0.85
            elif bx < 0.03 or bx > 0.97 or by < 0.03 or by > 0.97:
                s_oob[f] = 0.50
        elif interp_conf[f] > 0.3:
            bx, by = interp_xy[f]
            if bx < 0.01 or bx > 0.99 or by < 0.015 or by > 0.985:
                s_oob[f] = 0.70

    # ball teleportation jump
    s_jump = np.zeros(n, dtype=np.float32)
    length, width = PITCH
    for f in range(1, n):
        if ball_visible[f] and ball_visible[f - 1]:
            dx = (ball_xy[f, 0] - ball_xy[f - 1, 0]) * length
            dy = (ball_xy[f, 1] - ball_xy[f - 1, 1]) * width
            jump = np.sqrt(dx * dx + dy * dy)
            if jump > 10.0:
                lo = max(0, f - int(FPS * 1.5))
                hi = min(n, f + int(FPS * 0.5))
                s_jump[lo:hi] = np.maximum(s_jump[lo:hi], 0.8)

    # long ball invisibility (> 2.0 s = 50 frames)
    frames_since = np.nan_to_num(features[:, fi["frames_since_ball"]], nan=0.0)
    s_invisible = np.clip((frames_since - FPS * 2.0) / (FPS * 4.0), 0, 1.0) * 0.75

    # low player movement (walking/jogging)
    home_speed = np.nan_to_num(features[:, fi["home_mean_speed"]], nan=0.0)
    away_speed = np.nan_to_num(features[:, fi["away_mean_speed"]], nan=0.0)
    kernel = np.ones(FPS * 2) / (FPS * 2)
    combined_speed = (np.convolve(home_speed, kernel, mode="same") +
                      np.convolve(away_speed, kernel, mode="same")) / 2
    s_low_speed = np.clip(1.0 - combined_speed / 1.5, 0, 1) * 0.4

    # combine: take maximum of primary indicators + modulate with speed
    primary_max = np.maximum.reduce([s_halftime, s_oob, s_jump, s_invisible])
    dead_score = np.clip(primary_max + s_low_speed * (primary_max > 0.2), 0.0, 1.0)
    return dead_score.astype(np.float32)

if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    from pipeline.features import extract_features
    feats, names, bvis, bxy = extract_features(data_dir=data_dir)

    print("interpolating ball positions:")
    ixy, iconf = interpolate_ball(bxy, bvis)
    n_interp = int(np.sum((iconf > 0) & ~bvis))
    print(f"original ball: {bvis.sum()} frames ({100*bvis.mean():.1f}%)")
    print(f"after interp:  {int(np.sum(iconf > 0))} frames "
          f"({100*np.mean(iconf > 0):.1f}%)  (+{n_interp} recovered)")

    print("\ndetecting dead ball:")
    dead = detect_dead_ball(feats, bvis, bxy, ixy, iconf)
    thresholds = [0.2, 0.3, 0.4, 0.5]
    for t in thresholds:
        n = int(np.sum(dead > t))
        print(f"  Dead score > {t}: {n} frames ({100*n/N_FRAMES:.1f}%)")

    print(f"\ndone in {time.time() - t0:.1f} s")
