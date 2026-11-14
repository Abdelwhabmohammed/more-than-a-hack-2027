
import os
import numpy as np
import pandas as pd

from starter import load_atd, load_homography
from starter.baseline import team_of_each_track


print("Loading datasets...")

atd = load_atd.load()
df = atd.to_df()

meta_path, _ = load_atd.atd_paths()
team_codes = team_of_each_track(meta_path)

_, homography_error, homography_valid = load_homography.load()

n = len(df)

if n == 0:
    raise ValueError("ATD dataset is empty.")

if len(team_codes) != 46:
    raise ValueError(f"Expected 46 track slots, found {len(team_codes)}")

if len(homography_error) != n or len(homography_valid) != n:
    raise ValueError("Homography and ATD lengths do not match.")

home_indices = np.flatnonzero(team_codes == 0)
away_indices = np.flatnonzero(team_codes == 1)

if len(home_indices) == 0 or len(away_indices) == 0:
    raise ValueError("Could not identify both teams from metadata.")

print("Barcelona track slots:", len(home_indices))
print("Athletic track slots:", len(away_indices))


# =====================================================
# 1. Ball position
# =====================================================

ball_x = df["ball_x"].to_numpy(dtype=float)
ball_y = df["ball_y"].to_numpy(dtype=float)

ball_visible = np.isfinite(ball_x) & np.isfinite(ball_y)


# =====================================================
# 2. Player positions and speeds
# =====================================================

player_x = []
player_y = []
player_speeds = []
player_visibility = []

for track_id in range(1, 47):
    x = df[f"{track_id}_x"].to_numpy(dtype=float)
    y = df[f"{track_id}_y"].to_numpy(dtype=float)
    speed = df[f"{track_id}_s"].to_numpy(dtype=float)

    visible = np.isfinite(x) & np.isfinite(y)

    player_x.append(x)
    player_y.append(y)
    player_speeds.append(speed)
    player_visibility.append(visible)

x_matrix = np.column_stack(player_x)
y_matrix = np.column_stack(player_y)
speed_matrix = np.column_stack(player_speeds)
player_matrix = np.column_stack(player_visibility)

visible_players = player_matrix.sum(axis=1)
visible_home_players = player_matrix[:, home_indices].sum(axis=1)
visible_away_players = player_matrix[:, away_indices].sum(axis=1)


# =====================================================
# 3. Row statistics
# =====================================================

def row_mean(values, valid_mask):
    safe_values = np.where(valid_mask, values, 0.0)
    counts = valid_mask.sum(axis=1)

    result = np.full(values.shape[0], np.nan)
    valid_rows = counts > 0

    result[valid_rows] = (
        safe_values.sum(axis=1)[valid_rows] / counts[valid_rows]
    )

    return result


def row_std(values, valid_mask, means):
    differences = np.where(
        valid_mask,
        values - means[:, None],
        0.0
    )

    counts = valid_mask.sum(axis=1)
    result = np.full(values.shape[0], np.nan)
    valid_rows = counts > 0

    variance = (
        (differences ** 2).sum(axis=1)[valid_rows]
        / counts[valid_rows]
    )

    result[valid_rows] = np.sqrt(variance)
    return result


def row_min(values, valid_mask):
    safe_values = np.where(valid_mask, values, np.inf)
    result = safe_values.min(axis=1)
    result[~valid_mask.any(axis=1)] = np.nan
    return result


def row_max(values, valid_mask):
    safe_values = np.where(valid_mask, values, -np.inf)
    result = safe_values.max(axis=1)
    result[~valid_mask.any(axis=1)] = np.nan
    return result


# =====================================================
# 4. Team spatial and movement features
# =====================================================

print("Calculating team spatial features...")


def team_spatial_features(team_name, indices):

    x = x_matrix[:, indices]
    y = y_matrix[:, indices]
    speed = speed_matrix[:, indices]

    visible = player_matrix[:, indices]
    speed_valid = np.isfinite(speed)

    x_meters = x * 105.3
    y_meters = y * 68.0

    center_x = row_mean(x_meters, visible)
    center_y = row_mean(y_meters, visible)

    spread_x = row_std(x_meters, visible, center_x)
    spread_y = row_std(y_meters, visible, center_y)

    min_x = row_min(x_meters, visible)
    max_x = row_max(x_meters, visible)
    min_y = row_min(y_meters, visible)
    max_y = row_max(y_meters, visible)

    width = max_x - min_x
    depth = max_y - min_y

    average_speed = row_mean(speed, speed_valid)
    maximum_speed = row_max(speed, speed_valid)

    result = {
        f"{team_name}_center_x": center_x,
        f"{team_name}_center_y": center_y,
        f"{team_name}_spread_x": spread_x,
        f"{team_name}_spread_y": spread_y,
        f"{team_name}_width": width,
        f"{team_name}_depth": depth,
        f"{team_name}_average_speed": average_speed,
        f"{team_name}_maximum_speed": maximum_speed,
    }

    for zone_name, lower, upper in [
        ("left_zone", 0.0, 1.0 / 3.0),
        ("middle_zone", 1.0 / 3.0, 2.0 / 3.0),
        ("right_zone", 2.0 / 3.0, 1.000001),
    ]:
        zone_mask = visible & (x >= lower) & (x < upper)
        result[f"{team_name}_{zone_name}_players"] = zone_mask.sum(axis=1)

    for lane_name, lower, upper in [
        ("upper_lane", 0.0, 1.0 / 3.0),
        ("center_lane", 1.0 / 3.0, 2.0 / 3.0),
        ("lower_lane", 2.0 / 3.0, 1.000001),
    ]:
        lane_mask = visible & (y >= lower) & (y < upper)
        result[f"{team_name}_{lane_name}_players"] = lane_mask.sum(axis=1)

    return result


home_features = team_spatial_features("barcelona", home_indices)
away_features = team_spatial_features("athletic", away_indices)


# =====================================================
# 5. Ball-to-player distances
# =====================================================

print("Calculating ball-distance features...")

all_distances = []

for track_id in range(46):

    x = x_matrix[:, track_id]
    y = y_matrix[:, track_id]

    player_visible = player_matrix[:, track_id]

    distance = np.sqrt(
        ((x - ball_x) * 105.3) ** 2
        + ((y - ball_y) * 68.0) ** 2
    )

    valid_distance = player_visible & ball_visible
    distance[~valid_distance] = np.inf

    all_distances.append(distance)

distances = np.column_stack(all_distances)

home_distances = distances[:, home_indices]
away_distances = distances[:, away_indices]

nearest_home_dist = np.min(home_distances, axis=1)
nearest_away_dist = np.min(away_distances, axis=1)

nearest_home_dist[~np.isfinite(nearest_home_dist)] = np.nan
nearest_away_dist[~np.isfinite(nearest_away_dist)] = np.nan

distance_margin = nearest_away_dist - nearest_home_dist


# =====================================================
# 6. Nearest player information
# =====================================================

nearest_indices = np.argmin(distances, axis=1)
nearest_dist = np.min(distances, axis=1)

sorted_distances = np.sort(distances, axis=1)
second_nearest_dist = sorted_distances[:, 1]

nearest_dist[~np.isfinite(nearest_dist)] = np.nan
second_nearest_dist[~np.isfinite(second_nearest_dist)] = np.nan

nearest_gap = second_nearest_dist - nearest_dist

nearest_player_id = np.full(n, np.nan)
nearest_player_speed = np.full(n, np.nan)
nearest_team = np.full(n, "Unknown", dtype=object)

has_nearest = np.isfinite(nearest_dist)

nearest_player_id[has_nearest] = nearest_indices[has_nearest] + 1

nearest_player_speed[has_nearest] = speed_matrix[
    np.arange(n)[has_nearest],
    nearest_indices[has_nearest]
]

nearest_team[has_nearest] = np.where(
    team_codes[nearest_indices[has_nearest]] == 0,
    "Barcelona",
    "Athletic"
)


# =====================================================
# 7. Same-team nearest players
# =====================================================

nearest_two_same_team = np.full(n, np.nan)

has_two_valid_distances = (
    ball_visible
    & (visible_players >= 2)
)

second_indices = np.argsort(distances, axis=1)[:, 1]

nearest_two_same_team[has_two_valid_distances] = (
    team_codes[nearest_indices[has_two_valid_distances]]
    ==
    team_codes[second_indices[has_two_valid_distances]]
).astype(float)


# =====================================================
# 8. Build feature dataset
# =====================================================

print("Building feature table...")

features = pd.DataFrame({
    "frame_id": df["frame_id"].to_numpy(),
    "ball_x": ball_x,
    "ball_y": ball_y,
    "ball_visible": ball_visible,

    "visible_players": visible_players,
    "visible_home_players": visible_home_players,
    "visible_away_players": visible_away_players,

    "nearest_home_dist": nearest_home_dist,
    "nearest_away_dist": nearest_away_dist,
    "distance_margin": distance_margin,

    "nearest_player_id": nearest_player_id,
    "nearest_team": nearest_team,
    "nearest_player_speed": nearest_player_speed,

    "nearest_dist": nearest_dist,
    "second_nearest_dist": second_nearest_dist,
    "nearest_gap": nearest_gap,
    "nearest_two_same_team": nearest_two_same_team,

    "homography_error": homography_error,
    "homography_valid": homography_valid,
})

if "timestamp" in df.columns:
    features.insert(1, "timestamp", df["timestamp"].to_numpy())

for feature_group in [home_features, away_features]:
    for column_name, values in feature_group.items():
        features[column_name] = values

features["team_center_distance"] = np.sqrt(
    (
        features["barcelona_center_x"]
        - features["athletic_center_x"]
    ) ** 2
    +
    (
        features["barcelona_center_y"]
        - features["athletic_center_y"]
    ) ** 2
)

features["team_speed_difference"] = (
    features["barcelona_average_speed"]
    - features["athletic_average_speed"]
)


# =====================================================
# 9. Validation
# =====================================================

print("Validating features...")

if len(features) != 199319:
    raise ValueError(f"Unexpected number of rows: {len(features)}")

if not features["frame_id"].is_monotonic_increasing:
    raise ValueError("Frame IDs are not ordered.")

if features["frame_id"].duplicated().any():
    raise ValueError("Duplicate frame IDs detected.")

if not (
    features["visible_home_players"]
    + features["visible_away_players"]
    == features["visible_players"]
).all():
    raise ValueError("Player visibility counts do not match.")

missing_ball = ~features["ball_visible"]

spatial_columns = [
    "barcelona_center_x",
    "barcelona_center_y",
    "athletic_center_x",
    "athletic_center_y",
]

available_spatial = features.loc[
    missing_ball,
    spatial_columns
].notna().all(axis=1)

print(
    "Frames with missing ball but valid team centers:",
    int(available_spatial.sum()),
    "/",
    int(missing_ball.sum())
)


# =====================================================
# 10. Save
# =====================================================

output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "possession_features.csv"
)

features.to_csv(output_path, index=False)


# =====================================================
# 11. Report
# =====================================================

print("\n" + "=" * 60)
print("FEATURE DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print("Rows:", len(features))
print("Columns:", len(features.columns))
print("Output:", output_path)

print("\nPlayer visibility:")
print(features["visible_players"].describe())

print("\nTeam spatial features:")

summary_columns = [
    "barcelona_center_x",
    "barcelona_center_y",
    "barcelona_average_speed",
    "athletic_center_x",
    "athletic_center_y",
    "athletic_average_speed",
    "team_center_distance",
]

print(features[summary_columns].describe().round(3))

print("\nBall visibility:")
print(features["ball_visible"].value_counts())

print("\nFirst 5 rows:")
print(features.head().to_string(index=False))

print("\nDone!")
