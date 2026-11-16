
import numpy as np
import pandas as pd


def extract_possession_statistics(labels, fps=25):

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
