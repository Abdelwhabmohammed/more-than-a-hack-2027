
import os
import numpy as np
import pandas as pd

print("Loading feature dataset...")

# =====================================================
# 1. Load dataset
# =====================================================

input_path = os.path.join(
    "data",
    "processed",
    "possession_features.csv"
)

df = pd.read_csv(input_path)

if len(df) != 199319:
    raise ValueError(
        f"Unexpected number of rows: {len(df)}"
    )

if not df["frame_id"].is_monotonic_increasing:
    raise ValueError("Frame IDs are not ordered.")

print("Loaded rows:", len(df))
print("Original columns:", len(df.columns))

# =====================================================
# 2. Define independent time periods
# =====================================================

# frame_id is 1-based.
# Each group is processed independently to prevent
# temporal information from crossing half-time.

df["temporal_group"] = 0

# First half: video frames 11185 to 90651
df.loc[
    df["frame_id"].between(11186, 90652),
    "temporal_group"
] = 1

# Halftime
df.loc[
    df["frame_id"].between(90653, 110808),
    "temporal_group"
] = 2

# Second half: video frames 110808 to 190197
df.loc[
    df["frame_id"].between(110809, 190198),
    "temporal_group"
] = 3

# Frames after the second-half graded range
df.loc[
    df["frame_id"] > 190198,
    "temporal_group"
] = 4

print("\nTemporal groups:")
print(df["temporal_group"].value_counts().sort_index())

# =====================================================
# 3. Prepare basic signals
# =====================================================

df["nearest_team"] = (
    df["nearest_team"].fillna("Unknown")
)

df["is_barcelona_nearest"] = (
    df["nearest_team"] == "Barcelona"
).astype(int)

df["is_athletic_nearest"] = (
    df["nearest_team"] == "Athletic"
).astype(int)

df["is_unknown_nearest"] = (
    df["nearest_team"] == "Unknown"
).astype(int)

df["both_teams_visible"] = (
    (df["visible_home_players"] > 0)
    & (df["visible_away_players"] > 0)
).astype(int)

df["player_count_ratio"] = (
    df["visible_home_players"]
    / df["visible_players"].replace(0, np.nan)
)

# =====================================================
# 4. Define spatial features
# =====================================================

spatial_features = [

    # Barcelona
    "barcelona_center_x",
    "barcelona_center_y",
    "barcelona_spread_x",
    "barcelona_spread_y",
    "barcelona_width",
    "barcelona_depth",
    "barcelona_average_speed",
    "barcelona_maximum_speed",

    "barcelona_left_zone_players",
    "barcelona_middle_zone_players",
    "barcelona_right_zone_players",

    "barcelona_upper_lane_players",
    "barcelona_center_lane_players",
    "barcelona_lower_lane_players",

    # Athletic
    "athletic_center_x",
    "athletic_center_y",
    "athletic_spread_x",
    "athletic_spread_y",
    "athletic_width",
    "athletic_depth",
    "athletic_average_speed",
    "athletic_maximum_speed",

    "athletic_left_zone_players",
    "athletic_middle_zone_players",
    "athletic_right_zone_players",

    "athletic_upper_lane_players",
    "athletic_center_lane_players",
    "athletic_lower_lane_players",

    # Relative team features
    "team_center_distance",
    "team_speed_difference",
]

missing_spatial = [
    column
    for column in spatial_features
    if column not in df.columns
]

if missing_spatial:
    raise ValueError(
        "Missing spatial columns: "
        + ", ".join(missing_spatial)
    )

# =====================================================
# 5. Calculate team center movement
# =====================================================

print("\nCalculating team center movement...")

fps = 25.0

df["barcelona_center_movement_speed"] = np.nan
df["athletic_center_movement_speed"] = np.nan

for group_id, indices in df.groupby(
    "temporal_group",
    sort=False
).groups.items():

    segment = df.loc[indices]

    for team in ["barcelona", "athletic"]:

        x = segment[f"{team}_center_x"]
        y = segment[f"{team}_center_y"]

        dx = x.diff()
        dy = y.diff()

        movement = np.sqrt(
            dx ** 2 + dy ** 2
        ) * fps

        df.loc[
            indices,
            f"{team}_center_movement_speed"
        ] = movement.to_numpy()

# Add movement speeds to spatial features.
spatial_features.extend([
    "barcelona_center_movement_speed",
    "athletic_center_movement_speed",
])

# =====================================================
# 6. Define temporal windows
# =====================================================

windows = {
    "1s": 25,
    "3s": 75
}

# Existing temporal signals.
base_temporal_features = [
    "is_barcelona_nearest",
    "is_athletic_nearest",
    "is_unknown_nearest",
    "ball_visible",
    "visible_players",
    "visible_home_players",
    "visible_away_players",
    "both_teams_visible",
    "player_count_ratio",
    "nearest_dist",
    "distance_margin",
]

# =====================================================
# 7. Create output columns
# =====================================================

print("\nPreparing temporal columns...")

for name in windows:

    # Existing temporal features
    output_columns = [
        f"barcelona_ratio_{name}",
        f"athletic_ratio_{name}",
        f"unknown_ratio_{name}",
        f"ball_visibility_ratio_{name}",
        f"avg_visible_players_{name}",
        f"avg_home_players_{name}",
        f"avg_away_players_{name}",
        f"both_teams_visible_ratio_{name}",
        f"player_count_ratio_{name}",
        f"avg_nearest_dist_{name}",
        f"avg_distance_margin_{name}",
        f"team_changes_{name}",
    ]

    for column in output_columns:
        df[column] = np.nan

    # New spatial rolling averages
    for column in spatial_features:
        df[f"avg_{column}_{name}"] = np.nan

# =====================================================
# 8. Calculate temporal features by group
# =====================================================

print("\nCalculating temporal features...")

for group_id, indices in df.groupby(
    "temporal_group",
    sort=False
).groups.items():

    segment = df.loc[indices].copy()

    print(
        "Processing group:",
        group_id,
        "| Frames:",
        len(segment)
    )

    # Team changes inside this group only.
    changes = (
        segment["nearest_team"]
        .ne(segment["nearest_team"].shift())
        .astype(int)
    )

    if len(changes) > 0:
        changes.iloc[0] = 0

    for name, window in windows.items():

        # ---------------------------------------------
        # Existing team ratios
        # ---------------------------------------------

        ratio_features = {
            "barcelona_ratio": "is_barcelona_nearest",
            "athletic_ratio": "is_athletic_nearest",
            "unknown_ratio": "is_unknown_nearest",
            "ball_visibility_ratio": "ball_visible",
            "avg_visible_players": "visible_players",
            "avg_home_players": "visible_home_players",
            "avg_away_players": "visible_away_players",
            "both_teams_visible_ratio": "both_teams_visible",
            "player_count_ratio": "player_count_ratio",
            "avg_nearest_dist": "nearest_dist",
            "avg_distance_margin": "distance_margin",
        }

        for output_name, source_name in ratio_features.items():

            result = (
                segment[source_name]
                .rolling(
                    window=window,
                    min_periods=1
                )
                .mean()
            )

            df.loc[
                indices,
                f"{output_name}_{name}"
            ] = result.to_numpy()

        # ---------------------------------------------
        # Team changes
        # ---------------------------------------------

        change_result = (
            changes
            .rolling(
                window=window,
                min_periods=1
            )
            .sum()
        )

        df.loc[
            indices,
            f"team_changes_{name}"
        ] = change_result.to_numpy()

        # ---------------------------------------------
        # New spatial rolling averages
        # ---------------------------------------------

        for column in spatial_features:

            rolling_mean = (
                segment[column]
                .rolling(
                    window=window,
                    min_periods=1
                )
                .mean()
            )

            df.loc[
                indices,
                f"avg_{column}_{name}"
            ] = rolling_mean.to_numpy()

# =====================================================
# 9. Remove helper columns
# =====================================================

df = df.drop(
    columns=[
        "temporal_group",
        "is_barcelona_nearest",
        "is_athletic_nearest",
        "is_unknown_nearest",
        "both_teams_visible",
        "player_count_ratio",
    ]
)

# =====================================================
# 10. Validation
# =====================================================

print("\nValidating temporal features...")

for name in windows:

    ratio_sum = (
        df[f"barcelona_ratio_{name}"]
        + df[f"athletic_ratio_{name}"]
        + df[f"unknown_ratio_{name}"]
    )

    if not np.allclose(
        ratio_sum,
        1.0,
        atol=1e-8
    ):
        raise ValueError(
            f"Team ratios do not sum to 1 for {name}"
        )

    visibility = df[
        f"ball_visibility_ratio_{name}"
    ]

    if not visibility.between(0, 1).all():
        raise ValueError(
            f"Invalid ball visibility ratio for {name}"
        )

# Check new spatial features.
new_spatial_columns = [
    f"avg_{column}_{name}"
    for column in spatial_features
    for name in windows
]

missing_spatial_columns = [
    column
    for column in new_spatial_columns
    if column not in df.columns
]

if missing_spatial_columns:
    raise ValueError(
        "Some spatial temporal columns were not created."
    )

if len(df) != 199319:
    raise ValueError("Unexpected row count after processing.")

# =====================================================
# 11. Save dataset
# =====================================================

output_path = os.path.join(
    "data",
    "processed",
    "temporal_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

# =====================================================
# 12. Report
# =====================================================

print("\n" + "=" * 60)
print("TEMPORAL FEATURES CREATED SUCCESSFULLY")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Original spatial features:", len(spatial_features))
print("Temporal windows:", list(windows.keys()))
print("New spatial temporal columns:", len(new_spatial_columns))
print("Output:", output_path)

print("\nTemporal feature validation: PASSED")

print("\nFirst-half start:")
print(
    df.loc[
        df["frame_id"] == 11186,
        [
            "frame_id",
            "barcelona_ratio_1s",
            "athletic_ratio_1s",
            "unknown_ratio_1s",
            "avg_barcelona_center_x_1s",
            "avg_athletic_center_x_1s",
        ]
    ].to_string(index=False)
)

print("\nSecond-half start:")
print(
    df.loc[
        df["frame_id"] == 110809,
        [
            "frame_id",
            "barcelona_ratio_1s",
            "athletic_ratio_1s",
            "unknown_ratio_1s",
            "avg_barcelona_center_x_1s",
            "avg_athletic_center_x_1s",
        ]
    ].to_string(index=False)
)

print("\nDone!")
