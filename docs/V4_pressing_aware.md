# V4 — Pressing-Aware Possession Model (investigated, not shipped)

**Status: NOT SHIPPED** — ablated and rejected on evidence; kept as a reproducible
negative result.
**Code:** `pipeline3/` · **Run:** `python -m pipeline3.run --score`
**Reproduction of the shipped alternative stays:** `python -m pipeline2.run --variant v2`

---

## Hypothesis

Barcelona (home) press high with a high defensive line and man-oriented marking —
a documented tactical fact about this fixture (PPDA-style metrics, Barca
Innovation Hub pressing analytics). Consequence for possession inference:

> The **nearest player to the ball** is often a *pressing* Barcelona player while
> Athletic still possesses. Raw proximity measures the presser, not the possessor.

This predicts exactly the failure modes observed in V2's public-window
diagnostics: spurious team flickers inside possessions (the presser converging),
and transitions firing ~0.4 s early (margin flips mid-pass, before control).

## What was built (`pipeline3/pressing.py`, `pipeline3/evidence.py`)

### A. Time-to-Interception (TTI) — mini pitch-control model

The Pressing Intensity idea: pressure is not distance, it is *who can actually
reach the ball's future path*.

- Ball simulated forward ~2 s under exponential friction, **k self-supervised** by
  regressing log-speed ratios of consecutive visible fast-ball pairs: **k = 0.73
  1/s** (speed halving ≈ 0.9 s — physically right for a passed ball on grass).
- Each player projected under its own tracked velocity with a 0.3 s reaction
  delay (literature parameterisation) and 8 m/s sprint reach + 0.7 m control
  radius; a player intercepts at the first horizon where the gap closes.
- Team TTI = its best interceptor; score `tanh((TTI_away − TTI_home)/0.7)`.
- Opinionated on ~46% of frames (home sooner 27%, away 19%).

### B. Pressure discount

Per-team pressure on the ball = opponents within 3 m (smoothed 0.6 s) + closing
speed of the nearest player. Where the margin favours a team **and that team is
the one swarming the ball**, the margin evidence is scaled by up to −45%: it is
pressing bias, not possession.

### C. Counter-press window (the six-second rule)

Turnover detector: sustained flip of the smoothed raw evidence (margin + carry,
2 s smoothing) with ball alive within ±2 s (tackles hide the ball exactly at
turnovers — the ±0.2 s first version caught almost nothing). For 5.5 s after
losing a live ball, evidence toward the team that lost it is halved **unless it
shows a carry** (ball glued to and moving with its player): convergence is not
control. 511 turnovers detected; windows cover ~17% of frames per team.

### D. Defensive-line height

Own-goal-relative x of the 2nd-deepest defender per frame (~2 s smoothing) —
recognises the high-line press state from data instead of assuming it. Reported
but not gated on: with 46 anonymous track slots the deepest lines are noisy.

### A bug found on the way

The ATD's ball-speed column is ~93% empty even on visible frames. Everything
keyed on it (live-play tests, friction fit, halted-ball dead evidence) had been
reading NaN→0. All consumers now get the best-available estimate (ATD value where
present, else position-derived velocity). **This fix benefits V2 too** and is part
of `pipeline2/data_layer.py`.

## Ablation (structural metric = full-match possessions/mean duration; public = one-shot correctness read)

| configuration | possessions | mean s | public spellF1 / macroF1 |
|---|---|---|---|
| V2 (shipped) | 380 | 10.3 | **0.4615 / 0.7173** |
| V4, TTI as **additive vote** | 417 | 9.4 | 0.3429 / 0.6944 |
| V4, discount + CCW + TTI **veto** | 369 | 10.6 | 0.4103 / 0.7107 |
| V4, discount only | 374 | 10.4 | 0.4513 / 0.7170 |

## Findings

1. **TTI as an additive vote fragments the match.** It is opinionless on half the
   frames and flickers between opinions where it speaks — a third, noisy voice
   the DP happily splits on. It was demoted to a multiplicative veto (damp margin
   only where the interception race *firmly disagrees*).
2. **Even the veto misfires against a deep block.** Athletic defending deep in
   their half sits between ball and goal: on backward/sideways passes the
   *defenders* win the interception race, the veto damps Barcelona's evidence, and
   one near-perfect 42 s Barcelona possession collapsed from IoU 0.95 to 0.55.
   Without player identity we cannot separate "defender in a passing lane" from
   "receiver of the next pass".
3. **The pressure discount and counter-press gate are neutral** — structurally
   (374 vs 380 possessions) and on the public window (0.4513 vs 0.4615, inside
   the ±0.04 noise band). The presser bias the hypothesis targeted is real but is
   already absorbed by the affinity's occlusion persistence (~4 s decay) and the
   carry term.
4. Per the pre-registered tie-break (simpler variant on a noise-band tie), **V2
   ships; V4 does not** — the code stays runnable for inspection and the finding
   is documented.

## Reproduce

```bash
python -m pipeline3.run --score                  # full V4
python -m pipeline3.run --tti-veto 0 --score     # ablation: discount + CCW only
```
