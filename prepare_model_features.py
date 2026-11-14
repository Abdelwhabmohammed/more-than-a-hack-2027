
import os
import numpy as np
import pandas as pd

print("Loading temporal features...")

# =====================================================
# 1. Load dataset
# =====================================================

input_path = os.path.join(
    "data",
    "processed",
    "temporal_features.csv"
)

df = pd.read_csv(input_path)

if len(df) != 199319:
    raise ValueError(
        f"Unexpected number of rows: {len(df)}"
    )

# =====================================================
# 2. Original ball and tracking features
# =====================================================

base_features = [
    "ball_visible",
    "visible_players",
    "visible_home_players",
    "visible_away_players",

    "nearest_home_dist",
    "nearest_away_dist",
    "distance_margin",

    "nearest_player_speed",
    "nearest_dist",
    "second_nearest_dist",
    "nearest_gap",
    "nearest_two_same_team",

    "homography_error"
]

# =====================================================
# 3. Existing temporal features
# =====================================================

temporal_features = [
    # 1-second features
    "barcelona_ratio_1s",
    "athletic_ratio_1s",
    "unknown_ratio_1s",
    "ball_visibility_ratio_1s",
    "avg_visible_players_1s",
    "avg_home_players_1s",
    "avg_away_players_1s",
    "both_teams_visible_ratio_1s",
    "player_count_ratio_1s",
    "avg_nearest_dist_1s",
    "avg_distance_margin_1s",
    "team_changes_1s",

    # 3-second features
    "barcelona_ratio_3s",
    "athletic_ratio_3s",
    "unknown_ratio_3s",
    "ball_visibility_ratio_3s",
    "avg_visible_players_3s",
    "avg_home_players_3s",
    "avg_away_players_3s",
    "both_teams_visible_ratio_3s",
    "player_count_ratio_3s",
    "avg_nearest_dist_3s",
    "avg_distance_margin_3s",
    "team_changes_3s"
]

# =====================================================
# 4. New spatial features
# =====================================================

spatial_features = [

    # Barcelona spatial features
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

    # Athletic spatial features
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

    # Relative features
    "team_center_distance",
    "team_speed_difference",

    # Team center movement
    "barcelona_center_movement_speed",
    "athletic_center_movement_speed"
]

# =====================================================
# 5. New spatial temporal features
# =====================================================

spatial_temporal_features = []

for window in ["1s", "3s"]:

    for feature in spatial_features:

        spatial_temporal_features.append(
            f"avg_{feature}_{window}"
        )

# =====================================================
# 6. Combine all selected features
# =====================================================

selected_features = (
    base_features
    + temporal_features
    + spatial_features
    + spatial_temporal_features
)

# Prevent duplicate feature names.
if len(selected_features) != len(set(selected_features)):
    raise ValueError(
        "Duplicate feature names detected."
    )

print("\nFeature selection summary:")
print("Base features:", len(base_features))
print("Existing temporal features:", len(temporal_features))
print("Spatial features:", len(spatial_features))
print("Spatial temporal features:", len(spatial_temporal_features))
print("Total selected features:", len(selected_features))

# =====================================================
# 7. Check required columns
# =====================================================

missing_columns = [
    col
    for col in selected_features
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

# =====================================================
# 8. Create model input
# =====================================================

print("\nPreparing model input...")

model_df = df[
    ["frame_id"] + selected_features
].copy()

# Convert boolean columns to numeric.
for col in selected_features:

    if model_df[col].dtype == bool:

        model_df[col] = (
            model_df[col].astype(int)
        )

# Ensure all selected features are numeric.
for col in selected_features:

    model_df[col] = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )

# Replace infinite values with NaN.
model_df[selected_features] = (
    model_df[selected_features]
    .replace([np.inf, -np.inf], np.nan)
)

# =====================================================
# 9. Create missing-value indicators
# =====================================================

print("Creating missing-value indicators...")

missing_indicator_columns = []

for col in selected_features:

    if model_df[col].isna().any():

        indicator_name = col + "_missing"

        model_df[indicator_name] = (
            model_df[col].isna().astype(int)
        )

        missing_indicator_columns.append(
            indicator_name
        )

# Keep original missing values.
# Imputation must happen later inside the model pipeline.

# =====================================================
# 10. Validate dataset
# =====================================================

print("\nValidating model input...")

if len(model_df) != len(df):
    raise ValueError(
        "Row count changed unexpectedly!"
    )

if model_df["frame_id"].isna().any():
    raise ValueError(
        "Missing frame IDs detected!"
    )

if model_df["frame_id"].duplicated().any():
    raise ValueError(
        "Duplicate frame IDs detected!"
    )

if not model_df["frame_id"].is_monotonic_increasing:
    raise ValueError(
        "Frame IDs are not in order!"
    )

for col in missing_indicator_columns:

    if not model_df[col].isin([0, 1]).all():
        raise ValueError(
            f"Invalid missing indicator: {col}"
        )

# Verify that all model features are numeric.
non_numeric = [
    col
    for col in selected_features
    if not pd.api.types.is_numeric_dtype(model_df[col])
]

if non_numeric:
    raise ValueError(
        f"Non-numeric features found: {non_numeric}"
    )

# =====================================================
# 11. Save model input
# =====================================================

output_path = os.path.join(
    "data",
    "processed",
    "model_input.csv"
)

model_df.to_csv(
    output_path,
    index=False
)

# =====================================================
# 12. Print report
# =====================================================

print("\n" + "=" * 60)
print("MODEL INPUT CREATED SUCCESSFULLY")
print("=" * 60)

print("Rows:", len(model_df))
print("Columns:", len(model_df.columns))

print("\nSelected feature counts:")
print("Base:", len(base_features))
print("Existing temporal:", len(temporal_features))
print("Spatial:", len(spatial_features))
print("Spatial temporal:", len(spatial_temporal_features))
print("Total selected:", len(selected_features))

print("\nMissing indicators:", len(missing_indicator_columns))

print("\nMissing indicator columns:")

for col in missing_indicator_columns:
    print("-", col)

print("\nMissing values in indicator columns:")

if missing_indicator_columns:

    print(
        model_df[missing_indicator_columns]
        .isna()
        .sum()
        .to_string()
    )

else:
    print("No missing indicators were needed.")

print("\nFrame ID validation:")
print("First frame:", model_df["frame_id"].iloc[0])
print("Last frame:", model_df["frame_id"].iloc[-1])
print(
    "Frames ordered:",
    model_df["frame_id"].is_monotonic_increasing
)

print("\nOutput:", output_path)

print("\nFirst 5 rows:")
print(model_df.head().to_string(index=False))

print("\nDone!")
