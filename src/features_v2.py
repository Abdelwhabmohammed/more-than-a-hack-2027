import numpy as np
import pandas as pd

def compute_advanced_features(df, ball_speed_threshold=0.3):
    """
    Computes spatial and physical features for possession estimation:
    1. Distance-based team assignment (Home vs Away).
    2. Ball Speed filtering to catch Dead Ball ('D') state accurately.
    """
    features_df = df.copy()
    
    # 1. Compute Ball Speed (if ball coordinates exist)
    if 'ball_x' in features_df.columns and 'ball_y' in features_df.columns:
        dx = features_df['ball_x'].diff().fillna(0)
        dy = features_df['ball_y'].diff().fillna(0)
        # Assuming 25 FPS tracking data -> calculate speed in meters/second
        features_df['ball_speed'] = np.sqrt(dx**2 + dy**2) * 25.0
    else:
        features_df['ball_speed'] = 1.0  # Default fallback
        
    # 2. Raw Proximity Assignment
    if 'd_home' in features_df.columns and 'd_away' in features_df.columns:
        cond_home = features_df['d_home'] < features_df['d_away']
        raw_labels = np.where(cond_home, 'H', 'A')
        
        # Override with Dead Ball 'D' when ball is nearly stationary
        cond_dead = features_df['ball_speed'] < ball_speed_threshold
        raw_labels = np.where(cond_dead, 'D', raw_labels)
        
        features_df['raw_possession'] = raw_labels
    
    return features_df


def extract_possession_statistics(labels, fps=25):
    """
    Enhanced version of your original V1 statistics extractor.
    Extracts continuous possession spells (Home 'H' & Away 'A').
    """
    labels = np.asarray(labels)
    records = []

    current = None
    start = None

    for frame, label in enumerate(labels):
        if label in ("H", "A"):
            if label != current:
                if current is not None:
                    records.append({
                        "team": current,
                        "start_frame": start,
                        "end_frame": frame - 1,
                        "duration_frames": frame - start,
                        "duration_seconds": (frame - start) / fps
                    })
                current = label
                start = frame
        else:
            if current is not None:
                records.append({
                    "team": current,
                    "start_frame": start,
                    "end_frame": frame - 1,
                    "duration_frames": frame - start,
                    "duration_seconds": (frame - start) / fps
                })
            current = None
            start = None

    if current is not None:
        records.append({
            "team": current,
            "start_frame": start,
            "end_frame": len(labels) - 1,
            "duration_frames": len(labels) - start,
            "duration_seconds": (len(labels) - start) / fps
        })

    return pd.DataFrame(records)
