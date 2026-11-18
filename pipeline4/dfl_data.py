"""Reader for the external Bundesliga training matches (the DFL/IDSSE release).

The model is trained on: seven full
2022/23 Bundesliga / 2. Bundesliga matches published by the Deutsche Fussball Liga
under CC BY 4.0 (Bassek, Weber, Rein & Memmert, "An integrated dataset of
spatiotemporal and event data in elite soccer", *Scientific Data*, 2025), fetched
from the `pysport/idsse-data` mirror on Hugging Face into
`pipeline4/_cache/external_data/dfl/`

Why this dataset: every ball frame carries the provider's `BallStatus`
(in play or dead) and `BallPossession` (home or away), set on corrected tracking —
the same kind of label the challenge ground truth is built from (dead where the
corrected tracking marks the ball dead, otherwise the team it names as owner, per
CHALLENGE.md).  So per-frame H / A / D training labels come for free.

`load()` returns the challenge ATD layout so the same feature code runs on both:

  px, py   (frames, players)  normalised positions, NaN off the pitch
  ps (frames, players)  speed in m/s (the files carry km/h)
  team (players,) 0 = home, 1 = away
  bx, by (frames,) normalised ball position
  label (frames,) 0 = H, 1 = A, 2 = D
  half (frames,) 1 / 2, with a single empty frame between the halves
"""

import os
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
CACHE = os.environ.get("LP4_DFL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "external_data", "dfl"))
PARSED = os.environ.get("LP4_DFL_CACHE", os.path.join(CACHE, "parsed"))
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"

MATCH_FILES = {  # match id -> competition id used in the published file names
    "J03WMX": "000001", "J03WN1": "000001",
    "J03WOH": "000002", "J03WOY": "000002", "J03WPY": "000002",
    "J03WQQ": "000002", "J03WR9": "000002",
}
HALF_TAG = {"firstHalf": 1, "secondHalf": 2}
KMH_TO_MS = 1.0 / 3.6
HOME, AWAY, DEAD = 0, 1, 2

def xml_path(match_id, kind):
    prefix = {"info": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
              "positions": "DFL_04_03_positions_raw_observed"}[kind]
    name = "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(prefix, MATCH_FILES[match_id], match_id)
    return os.path.join(CACHE, name)

def fetch(match_id):
    """Download the three files of a match, skipping complete ones."""
    os.makedirs(CACHE, exist_ok=True)
    for kind in ("info", "events", "positions"):
        path = xml_path(match_id, kind)
        if os.path.isfile(path) and os.path.getsize(path) > 0 and _downloaded(path):
            continue
        url = MIRROR + os.path.basename(path)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(path, "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)

def _downloaded(path):
    """A finished XML ends with its closing root tag."""
    with open(path, "rb") as handle:
        handle.seek(max(os.path.getsize(path) - 200, 0))
        return b"</PutDataRequest>" in handle.read()

def read_match_info(match_id):
    root = ET.parse(xml_path(match_id, "info")).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    person_team = {}
    for team in root.iter("Team"):
        for player in team.iter("Player"):
            person_team[player.get("PersonId")] = team.get("TeamId")
    return {"home": general.get("HomeTeamId"), "away": general.get("GuestTeamId"),
            "title": general.get("MatchTitle"), "competition": general.get("CompetitionName"),
            "pitch_length": float(environment.get("PitchX")),
            "pitch_width": float(environment.get("PitchY")),
            "person_team": person_team}

def stream_frames(match_id):
    """{(half, person-or-BALL): (N, cols) array} streamed from the positions XML.

    Ball rows carry (N, X, Y, S, BallPossession, BallStatus); player rows carry
    (N, X, Y, S)."""
    streams = {}
    open_set, buffer = None, None
    for event, element in ET.iterparse(xml_path(match_id, "positions"), events=("start", "end")):
        tag = element.tag
        if event == "start" and tag == "FrameSet":
            half = HALF_TAG.get(element.get("GameSection"))
            team_id = element.get("TeamId")
            if half is None or team_id == "referee":
                open_set = None
            else:
                open_set = (half, "BALL" if team_id == "BALL" else element.get("PersonId"))
            buffer = []
        elif event == "end" and tag == "Frame":
            if open_set is not None:
                a = element.attrib
                frame_no = int(a["N"])
                x, y = float(a["X"]), float(a["Y"])
                speed = float(a.get("S", "nan"))
                if open_set[1] == "BALL":
                    buffer.append((frame_no, x, y, speed, int(a.get("BallPossession", "0")), int(a.get("BallStatus", "0"))))
                else:
                    buffer.append((frame_no, x, y, speed))
            element.clear()
        elif event == "end" and tag == "FrameSet":
            if open_set is not None and buffer:
                streams[open_set] = np.array(buffer, dtype=np.float64)
            open_set, buffer = None, None
            element.clear()
    return streams

def assemble(match_id):
    """Combine the streams into the ATD layout, normalised and labelled."""
    info = read_match_info(match_id)
    streams = stream_frames(match_id)
    length, width = info["pitch_length"], info["pitch_width"]
    players = sorted({person for (_, person) in streams if person != "BALL"})
    team = np.array([0 if info["person_team"].get(p) == info["home"] else 1 for p in players], dtype=np.int8)
    slot = {p: i for i, p in enumerate(players)}

    halves = []
    for half in (1, 2):
        ball = streams[(half, "BALL")]
        first, last = int(ball[:, 0].min()), int(ball[:, 0].max())
        n = last - first + 1
        px = np.full((n, len(players)), np.nan, dtype=np.float32)
        py = np.full_like(px, np.nan)
        ps = np.full_like(px, np.nan)
        for (set_half, person), rows in streams.items():
            if set_half != half or person == "BALL":
                continue
            idx = rows[:, 0].astype(np.int64) - first
            inside = (idx >= 0) & (idx < n)
            slot_no = slot[person]
            px[idx[inside], slot_no] = (rows[inside, 1] + length / 2) / length
            py[idx[inside], slot_no] = (rows[inside, 2] + width / 2) / width
            ps[idx[inside], slot_no] = rows[inside, 3] * KMH_TO_MS
        idx = ball[:, 0].astype(np.int64) - first
        bx = np.full(n, np.nan, dtype=np.float32)
        by = np.full(n, np.nan, dtype=np.float32)
        bx[idx] = (ball[:, 1] + length / 2) / length
        by[idx] = (ball[:, 2] + width / 2) / width
        possession, status = ball[:, 4].astype(int), ball[:, 5].astype(int)
        label = np.full(n, DEAD, dtype=np.int8)
        alive = status == 1
        label[idx] = np.where(alive, np.where(possession == 1, HOME, np.where(possession == 2, AWAY, DEAD)), DEAD)
        halves.append((px, py, ps, bx, by, label, np.full(n, half, dtype=np.int8)))
        if half == 1:
            # one empty frame between the halves: nothing may run across halftime
            gap = np.full((1, len(players)), np.nan, dtype=np.float32)
            halves.append((gap, gap, gap, np.full(1, np.nan, np.float32),
                           np.full(1, np.nan, np.float32), np.full(1, DEAD, np.int8),
                           np.zeros(1, np.int8)))
    out = {name: np.concatenate([part[i] for part in halves]) for i, name in
           enumerate(("px", "py", "ps", "bx", "by", "label", "half"))}
    out["team"] = team
    out["length"] = np.array(length)
    out["width"] = np.array(width)
    return out, info

def load(match_id):
    """The match in ATD layout; parsed once, cached as npz."""
    os.makedirs(PARSED, exist_ok=True)
    path = os.path.join(PARSED, match_id + ".npz")
    if not os.path.isfile(path):
        fetch(match_id)
        data, _ = assemble(match_id)
        np.savez_compressed(path, **data)
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}

def sanity_checks(match_id, data):
    """Label and unit sanity: does BallPossession=1 mean home, and is S really km/h?"""
    label, half = data["label"], data["half"]
    graded = half > 0
    alive = graded & (label != DEAD)
    seen = np.isfinite(data["bx"])
    length, width = float(data["length"]), float(data["width"])
    dist = np.sqrt(((data["px"] - data["bx"][:, None]) * length) ** 2 + ((data["py"] - data["by"][:, None]) * width) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest_team = data["team"][dist.argmin(axis=1)]
    near_ball = alive & seen & (dist.min(axis=1) < 1.0)
    agreement = float((nearest_team[near_ball] == label[near_ball]).mean())
    dx = np.diff(data["px"], axis=0) * length
    dy = np.diff(data["py"], axis=0) * width
    from_positions = np.hypot(dx, dy) * 25
    from_file = data["ps"][1:]
    usable = np.isfinite(from_positions) & np.isfinite(from_file) & (from_file > 1)
    speed_ratio = float(np.median(from_positions[usable] / from_file[usable]))
    codes = label[graded]
    flips = int((np.diff(codes) != 0).sum())
    return {"frames": int(graded.sum()), "minutes": graded.sum() / 1500.0,
            "H": float((codes == HOME).mean()), "A": float((codes == AWAY).mean()),
            "D": float((codes == DEAD).mean()), "flips": flips,
            "nearest_agrees_with_possession": agreement, "speed_ratio": speed_ratio,
            "players_on_pitch": float(np.isfinite(data["px"][graded]).sum(axis=1).mean())}

if __name__ == "__main__":
    import sys
    for match_id in (sys.argv[1:] or sorted(MATCH_FILES)):
        data = load(match_id)
        report = sanity_checks(match_id, data)
        print("{}  {:5.1f} min  H {:.0%} A {:.0%} D {:.0%}  flips {:4d}  players {:.1f}  "
              "nearest==possession {:.0%}  speed ratio {:.2f}".format(
                  match_id, report["minutes"], report["H"], report["A"], report["D"],
                  report["flips"], report["players_on_pitch"],
                  report["nearest_agrees_with_possession"], report["speed_ratio"]))
