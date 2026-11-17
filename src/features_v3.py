import numpy as np
import pandas as pd

def compute_advanced_features_v3(df, ball_speed_threshold=0.40):
    """
    Features V3 (Enhanced):
    - Adjusts Dead Ball speed threshold (0.40 m/s) to cut out false dynamic plays.
    - Resolves tight player-ball proximity to reduce frame ambiguity.
    """
    features_df = df.copy()
    
    if 'ball_x' in features_df.columns and 'ball_y' in features_df.columns:
        dx = features_df['ball_x'].diff().fillna(0)
        dy = features_df['ball_y'].diff().fillna(0)
        features_df['ball_speed'] = np.sqrt(dx**2 + dy**2) * 25.0
    else:
        features_df['ball_speed'] = 1.0
        
    if 'd_home' in features_df.columns and 'd_away' in features_df.columns:
        cond_home = features_df['d_home'] < features_df['d_away']
        raw_labels = np.where(cond_home, 'H', 'A')
        
        # Dead Ball check with 0.40 m/s cutoff
        cond_dead = features_df['ball_speed'] < ball_speed_threshold
        raw_labels = np.where(cond_dead, 'D', raw_labels)
        
        features_df['raw_possession'] = raw_labels
    
    return features_df


def extract_possession_statistics_v3(labels, fps=25):
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
