# V2 — Core Pipeline

**Status: SHIPPED** — this is the variant behind `Lets_Play.csv`.
**Code:** `pipeline2/` · **Run:** `python -m pipeline2.run --variant v2 -o Lets_Play.csv`
**Public-window sanity score:** spellF1 0.4615 / macroF1 0.7173 (attempt 1: 0.4343 / 0.6873)

---

## Idea

Possession is a segmentation problem — `spellF1` scores whole possessions by temporal
IoU — so V2 replaces attempt-1's hand-weighted boundary heuristics (~15 tuned knobs)
with a **globally optimal DP partition** fed by a much better tracking data layer, and
composes the alive/dead call with the team call instead of voting between them.

## Architecture

```
ATD ──> data_layer.py ──> evidence.py ──> segmentation.py ──> postprocess2.py ──> CSV
        dense parse       s(f), p_D(f)    DP partition (λ=30,      merge + absorb
        + signals         (affinity,      dwell 15/15/75)          micro-segments
                          dead prob)
```

### Data layer (`pipeline2/data_layer.py`)

One dense parse of the tactical ATD (cached to `pipeline2/_cache/`, ~30 s once), then:

- **Nearest-player geometry** per team (metres), and the margin
  `(d_away − d_home)/(d_away + d_home)`, persisted through ball occlusions with a
  ~4 s exponential decay so contested/loose phases stay with the last team.
- **Ball velocity** from centred position differences, teleports (> 8 m/frame)
  invalidated; the ATD's own ball-speed column is ~93% empty, so consumers get the
  best available estimate (ATD value where present, else position-derived).
- **Carry score** per team: ball within 1.8 m of a team's player, relative speed
  < 2.5 m/s, ~1 s window — the kinematic signature of control.
- **Physics-aware gap fill**: constant-velocity blend for gaps ≤ 0.5 s (usable as
  evidence), straight-line for ≤ 3 s (geometry only), nothing longer — a long gap
  is information about stoppages, not a fake position.
- **Out-of-play geometry** (±0.5 m beyond the surface) and **freeze** (all players
  < 1 m/s over 2 s).
- **Data-driven set-piece windows**: for each of the 63 tagged set pieces/goals,
  walk back from the tag to the last frame with a visible, in-bounds, moving ball
  (play stopped) and forward to the first sustained live frame (restart). 47/63
  measured from tracking; fixed football priors only as fallback. Goal
  celebrations are covered by pairing each GOALS tag with its Kick Off (After Goal).

### Evidence (`pipeline2/evidence.py`)

- **Team affinity** `s(f) ∈ [−1, +1]`: margin (w 1.2) + carry difference (w 1.2) +
  Smart Tagging team tags only where geometry is silent (w 0.8), tanh-squashed,
  median+uniform smoothed (~0.75 s).
- **Dead probability** `p_D(f) = σ(evidence − 2.0)`: out-of-bounds 3.0; measured
  set-piece window 2.5 / prior 1.8; long ball gap 2.0→4.0 (2 s→5 s, **gated by
  mean player speed** — a 3 s occlusion during a sprint is an aerial ball, not a
  whistle; GT-stoppage frames run 1.43 m/s vs 2.37 alive); freeze 1.6; halted
  ball 1.0; empty pitch 4.0/1.0.

### Segmentation (`pipeline2/segmentation.py`)

**Compositional emissions** (the one modeling fix that mattered most):
`P(D) = p_D`, `P(H) = σ(k·s)·(1−p_D)`, `P(A) = σ(−k·s)·(1−p_D)` — a softmax would
let a strong affinity outvote a decisive dead probability; composition makes the
alive/dead call trump the team call exactly when it should.

Then a per-half DP that minimises `Σ segment emission cost + λ·(number of
segments)` subject to minimum dwell (H/A 0.6 s, D 3.0 s), with **λ = 30
calibrated structurally**: it lands the full match on 380 possessions of mean
10.3 s vs the published truth-window stats extrapolated (~360 of ~14 s). The DP
is O(N) per state via monotonic deques over prefix sums.

### Post-processing (`pipeline2/postprocess2.py`)

Merge adjacent same-label segments, absorb possessions < 0.8 s into the longer
neighbour, merge dead stretches < 3 s between same-team possessions. Deliberately
merge-averse elsewhere: a split possession keeps ~half its IoU credit, a wrong
merge loses both.

## Why it wins

| property | value |
|---|---|
| Full-match possessions | 380 (prior ~300–400) |
| Boundary stability (ST off, λ±20%) | 99.6% within 1 s |
| Self-supervised ball-mask recovery | 99.7% of masked frames |
| Public spellF1 / macroF1 | 0.4615 / 0.7173 |
| Team accuracy on alive frames | 0.89 (attempt 1: 0.73) |
| Determinism | byte-identical across runs |

## Known weaknesses (see main doc, §Limitations)

Transitions fire ~0.4 s early (margin flips mid-pass); brief deflection
possessions are absorbed; D recall on the public window is 0.38.
