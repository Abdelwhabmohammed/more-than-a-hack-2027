# V3 — Broadcast-ATD Fusion (investigated, dropped by its own gate)

**Status: NOT SHIPPED** — dropped by a pre-registered alignment-quality gate.
**Code:** `pipeline2/broadcast_align.py` · **Run:** `python -m pipeline2.run --variant v3` (prints the gate report, then falls back to V2 behaviour)

---

## Idea

The broadcast camera carries a second, independent ATD of the same ball. The
tactical tracking loses the ball on 41% of graded frames; two independent cameras
rarely occlude on the same frames, so where the tactical track has a gap the
broadcast track might hold a real measurement instead of an interpolation.

## Method

1. **Stretch extraction** — split the broadcast timeline into continuous
   ball-present stretches (gaps > 2 records split).
2. **Per-stretch offset alignment** — the broadcast feed is *produced*: cuts,
   replays and time-shifts mean **no global offset exists** (measured: half 2
   had 0/18 chunks consistent with any single offset). Each stretch therefore
   gets its own constant offset, found by cross-correlating ball-x (both sides
   in normalised pitch coordinates) over ±30 s in 0.2 s steps.
3. **Quality gate** — a stretch is accepted only if
   - its best Pearson r ≥ 0.75 over ≥ 20 s of overlap, **and**
   - its offset agrees (±1 s) with an accepted neighbouring stretch in
     broadcast time — isolated "agreements" are spurious correlations between
     two slow pans.
4. **Fusion** — accepted stretches fill tactical ball gaps only (never overwrite
   a tactical observation), tagged with their correlation, and every
   ball-dependent signal is recomputed on the fused arrays.

## Result

| metric | value |
|---|---|
| usable stretches (≥ 20 s) | 58 |
| passed correlation alone | 19 |
| accepted (corr + neighbourhood) | 19 (mean r = 0.885) |
| tactical frames fused | 2,028 of 199,319 (**1.0%**) |
| usefulness gate (≥ 3%) | **FAILED** |

The alignment works — correlations of 0.885 are genuine matches — but the fused
coverage is an order of magnitude below the usefulness threshold: **the broadcast
camera loses the ball on substantially the same frames as the tactical camera**.
The remaining 1% does not justify the pipeline complexity and the risk of a
miscalibrated stretch injecting phantom ball motion.

## What survives

- The module and its gate report remain runnable and documented
  (`python -m pipeline2.broadcast_align`), satisfying the reproducibility and
  inspection requirements.
- The per-stretch alignment machinery is reusable if a future dataset has a
  properly synchronised second feed.
- A useful negative fact for the technical report: *independent-camera fusion
  does not meaningfully extend ball coverage on this fixture.*

## Rule-compliance note

The gate threshold (0.75 r, 20 s overlap, 3% coverage, ±30 s offset range) was
fixed before running, from engineering priors — not chosen after seeing results.
