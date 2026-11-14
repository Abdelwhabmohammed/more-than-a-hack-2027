import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require  # noqa: E402
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]

# label codes - same convention as the rest of the pipeline
H, A, D, UNKNOWN = 0, 1, 2, -1
# team name = side code.
TEAM_MAP = {
    "barcelona": H,
    "atletic club": A,
}

# phase codes = (who_has_ball_relative_to_tagged_team, confidence)
PHASE_RULES = {
    "build up": ("self", 0.70),
    "progression": ("self", 0.75),
    "creating chances": ("self", 0.80),
    "ball in final third": ("self", 0.70),
    "ball in the box": ("self", 0.75),
    "long ball": ("self", 0.70),
    "goals": ("self", 0.60),
    "players in the box": ("none", 0.00),  # Not a possession event
    "defending in attacking third": ("other", 0.60),
    "defending in middle third": ("other", 0.60),
    "defending in defensive third": ("other", 0.60),
    "attacking transition": ("self", 0.70),
    "defensive transition": ("other", 0.70),
    "set pieces": ("set_piece", 0.85),
}

# dead-ball duration before each set piece restart
SET_PIECE_DEAD_DURATIONS = {
    "kick off (start)": 15.0,
    "kick off (after goal)": 35.0,
    "goal kick": 18.0,
    "corner kick": 20.0,
    "free kick": 18.0,
    "throw-in": 10.0,
    "default": 15.0,
}

def _load_smart_tagging(data_dir=None, game=GAME):
    """parse the Smart Tagging XML into a list of dicts."""
    st_path = require(["smart_tagging"], game, data_dir or DATA_DIR)[0]
    events = []
    for instance in ET.parse(st_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {}
        for label in instance.findall("label"):
            group = label.findtext("group")
            text = label.findtext("text")
            if group and text:
                tags[group] = text
        events.append({
            "code": (instance.findtext("code") or "").strip(),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * FPS)),
            "end_frame": int(round(end_s * FPS)),
            "tags": tags,
        })
    return events

def _resolve_team(team_tag):
    """map a team tag value to H/A/UNKNOWN."""
    if team_tag is None:
        return UNKNOWN
    return TEAM_MAP.get(team_tag.strip().lower(), UNKNOWN)

def build_smart_tagging_prior(data_dir=None, game=GAME, n_frames=None):
    """per-frame priors from Smart Tagging"""
    n_frames = n_frames or N_FRAMES
    events = _load_smart_tagging(data_dir, game)
    prior_labels = np.full(n_frames, UNKNOWN, dtype=np.int8)
    prior_confidence = np.zeros(n_frames, dtype=np.float32)
    dead_ball_mask = np.zeros(n_frames, dtype=bool)
    set_piece_restarts = []
    for ev in events:
        code = ev["code"].strip().lower()
        rule = PHASE_RULES.get(code)
        if rule is None:
            continue
        rel, conf = rule
        team_code = _resolve_team(ev["tags"].get("Team"))
        sf = max(0, ev["start_frame"])
        ef = min(n_frames, ev["end_frame"])
        if rel == "set_piece":
            sp_type = ev["tags"].get("Type", "default").strip().lower()
            dead_dur_s = SET_PIECE_DEAD_DURATIONS.get(sp_type, SET_PIECE_DEAD_DURATIONS["default"])
            dead_frames = int(dead_dur_s * FPS)
            # the preparation leading up to and including sf is dead ball
            dead_start = max(0, sf - dead_frames // 2)
            dead_end = min(n_frames, sf + dead_frames // 2)
            dead_ball_mask[dead_start:dead_end] = True
            for f in range(dead_start, dead_end):
                if prior_confidence[f] < conf:
                    prior_labels[f] = D
                    prior_confidence[f] = conf
            if team_code != UNKNOWN:
                restart_point = dead_end
                set_piece_restarts.append((restart_point, team_code))
                for f in range(restart_point, ef):
                    if prior_confidence[f] < conf:
                        prior_labels[f] = team_code
                        prior_confidence[f] = conf
        elif rel == "self":
            if team_code == UNKNOWN:
                continue
            for f in range(sf, ef):
                if prior_confidence[f] < conf:
                    prior_labels[f] = team_code
                    prior_confidence[f] = conf
        elif rel == "other":
            if team_code == UNKNOWN:
                continue
            other = A if team_code == H else H
            for f in range(sf, ef):
                if prior_confidence[f] < conf:
                    prior_labels[f] = other
                    prior_confidence[f] = conf
    return prior_labels, prior_confidence, dead_ball_mask, set_piece_restarts

if __name__ == "__main__":
    data_dir = os.environ.get("MTAH_DATA_DIR") or DATA_DIR
    events = _load_smart_tagging(data_dir)
    print(f"loaded {len(events)} Smart Tagging instances")
    labels, conf, db_mask, restarts = build_smart_tagging_prior(data_dir=data_dir)
    label_names = {H: "H (home)", A: "A (away)", D: "D (dead)", UNKNOWN: "unknown"}
    print(f"\nprior label coverage:")
    for code in [H, A, D, UNKNOWN]:
        n = int(np.sum(labels == code))
        print(f"  {label_names[code]:<16s}  {n:>8d} frames  ({100*n/N_FRAMES:.1f}%)")

    covered = int(np.sum(labels != UNKNOWN))
    print(f"\ntotal covered: {covered} / {N_FRAMES} ({100*covered/N_FRAMES:.1f}%)")
    print(f"dead-ball mask: {int(db_mask.sum())} frames")
    print(f"set piece restarts found: {len(restarts)}")
    if covered > 0:
        print(f"mean confidence (where covered): {conf[labels != UNKNOWN].mean():.3f}")