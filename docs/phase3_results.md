# Phase 3 Results — Rebuilt Good Detection (offline)

**Status:** offline portion complete on branch `phase3-proavalon` (2026-06-17);
Gate-0 passed, the threshold sweep is done, **Gate-1 (live smoke) PASSED on
2026-06-19**, and **Gate-2 (live confirmation) PASSED on 2026-06-20** — Good
self-play win rate against frozen GRAIL-evil rose from the 23.3 % phase-2
baseline to **51.1 % (47/92 pooled, p = 0.0001)**. This document reports the data
pipeline, training, the full offline evaluation (Gate 0), the recalibration that
unblocks the live phase, and the Gate-1/Gate-2 live results. Rationale and
design: `phase3_plan.md`; phase-2 diagnosis: `phase2_results.md`; pre-data
scaffolding: `phase3_days1_3_status.md`.

## TL;DR

- **Detection is decisively rebuilt (Gate-0 binding criterion: PASS).** Against
  the phase-2 chance baseline (factor_v2 self-play-90 AUC 0.598, mean P(evil) on
  true evil exactly 0.500), the card models reach **0.72** (Track A) / **0.69**
  (Track B) on the same self-play set and **0.65–0.69** on held-out human data —
  all well above chance, calibrated (ECE 0.01–0.04, T≈1.0). The aspirational
  targets (AUC ≥ 0.75, mean P(evil|evil) ≥ 0.65, ≥ 40 % flip) are **not** met by
  the deployed-style config and stay open.
- **Rejected-proposal history is a real, distribution-specific component of the
  signal.** Under matched train/eval ablation, removing rejected-proposal cards
  drops self-play AUC 0.722 → **0.539** and human AUC 0.646 → 0.623 — a large
  self-play effect, smaller on human. (An earlier "collapse to 0.454, below
  chance" was an artifact of scoring the ablated model on full card streams it
  never saw in training; an independent verification caught it and a new
  inference-time ablation hook fixes it. The dramatic below-chance figure is
  **not** cited.)
- **Proposer identity does not transfer:** dropping it *raises* self-play AUC to
  **0.796** (paired-bootstrap 95 % CI [+0.058, +0.089], consistent across quests
  2–5) but *slightly hurts* human AUC (−0.006, CI excludes 0). It is a
  distribution-shift feature (a human proposer→evil correlation that misleads on
  heuristic GRAIL-evil), not "null." Drop it for a self-play target; down-weight
  rather than delete if human play matters.
- **The win-rate translation is bottlenecked by calibration, not ranking.**
  Detection ranks evil strongly late-game (self-play AUC 0.70/0.81/0.82 in
  quests 3/4/5) but the calibrated P(evil) stays compressed (~0.5–0.62 on true
  evil), so it rarely crosses Good's 0.55–0.65 reject thresholds. Decisive
  counterexample: the *higher*-AUC no_proposer model (0.796) flips *fewer* evil
  teams (26.3 %) than full (27.8 %). **Live runs must recalibrate Good's reject
  thresholds to the new model** — the single most important pre-live finding.

## Data pipeline (Workstream 1 on real data)

`parse_proavalon.py`, written pre-arrival against the `voteHistory` format,
was validated on `dataset/six_player_games_latest_compressed.json`:

- 101,280 raw 6p records → **100,645 parsed** (635 skipped, all leader-rotation
  edge cases, 0.6%). Every game has exactly 2 evil.
- **Adversarial cross-check:** the decoded events, projected back onto the
  legacy 21-int state vector, match the battle-tested `generate_dataset_2`
  vectorizer on **2976/2976** sampled games (observable fields).
- **Bug found and fixed** by the cross-check: all 3,288 hammer games
  (`howTheGameWasWon == "Hammer rejected."`) log their unplayed final quest as
  `"failed"` in `missionHistory`. The parser now keeps those rejected proposals
  (real votes — exactly the new signal) but emits **no** QuestEvent for an
  unplayed quest; `GameRecord.validate` now rejects that class of bug outright.

`build_corpus.py` → **97,594 games** after the role-variant filter
(Merlin/Percival/Assassin/Morgana only; 3,051 dropped for Mordred/Oberon/
Tristan/Isolde). 1.55M voted proposals. **Chat-presence check: `{}`** — the
ProAvalon dump carries no chat, so the LLM-vibes signal cannot be trained from
it; vibes stays a runtime-only prior, as the plan anticipated.

Coverage (the evidence the card features have support): rejected proposals are
abundant — most games carry 8–17 rejected proposals, and proposal-depth is
deep at every level (q1 averages 4.6 proposals; depth-5 occurs 228k times).

## Training (Workstreams 2 & 3 on real data)

Both tracks trained on a seeded **12k/2.5k-game** subset of the ProAvalon
splits (`proavalon_capped.json`, **1.52M / 318k snapshots**) — the plan's
~1.8M design point, past saturation for these small models, within 16 GB RAM.
Model selection / early stopping / temperature on the human val split. Honest
held-out evaluation uses the **full** ProAvalon test split (disjoint — leakage
audit passed: 0 of the 4000 scored test games appear in training).

| | Track A (`v4_trackA_proavalon`) | Track B (`seq_v1_proavalon`) |
|---|---|---|
| model | `ProposalSetDistribution` + exact 15-pair enumeration | `CardSeqGRU` 2×128 + 15-way pair head |
| params | 8.5k | 205k |
| val loss | 0.5685 | 1.7609 |
| val pair top-1 | — | 0.3125 (chance 0.067) |
| temperature | **1.000** | 0.992 |

T≈1.0 confirms native calibration without `pos_weight` (the documented phase-3
lesson; pos_weight stays off).

## Offline evaluation (Gate 0)

### Detection quality (vibes off, pure model)

| eval set (n games) | factor_v2 | Track A | Track B |
|---|---|---|---|
| **self-play-90** (79) | 0.598 | **0.722** | 0.694 |
| avalonlogs-test (268) | —¹ | 0.671 | 0.687 |
| proavalon-test (4000) | —¹ | 0.646 | 0.652 |
| pair top-1, self-play (chance 0.067) | 0.112 | 0.281 | **0.345** |
| ECE (self-play) | 0.088 | 0.035 | 0.023 |
| mean P(evil) on true evil, self-play | 0.500 | 0.496 | 0.467 |

¹ factor_v2 uses loopy BP (~64 s/game → ~84 min for 79 games); it is
intractable on the human test splits and its chance-level behavior is already
established on self-play-90 and by the phase-2 diagnosis.

**The detection ramps with evidence** (Track A self-play AUC by quest):
q1 0.51 → q2 0.50 → q3 0.70 → q4 0.81 → q5 0.82. Early game (no rejected-team
history yet) is at chance; by mid/late game, once rejection dynamics
accumulate, ranking is strong. The pooled 0.722 averages the weak-early and
strong-late states.

In-distribution human (proavalon-test) AUC (0.65) is *lower* than self-play
(0.72): GRAIL-evil's heuristic blend-in is a more learnable, more consistent
tell than varied human deception, so a human-trained detector transfers
**up** to the patterned GRAIL adversary.

### Rejected-proposal attribution (Track A)

Each ablated model is scored **in-distribution** — the same feature mask used at
training is applied at inference (new `ablation` hook in `FactorGraphModelV3`/
`FactorV3Backend`). This matters: scoring the `no_rejected` model on the full
card stream it never saw drove a spurious below-chance self-play AUC of 0.454,
which an independent verification traced to the train/eval mismatch.

| variant (matched eval) | self-play AUC | proavalon-test AUC |
|---|---|---|
| full | 0.722 | 0.646 |
| no_rejected | 0.539 | 0.623 |
| no_proposer | 0.796 | 0.641 |

- **Rejected proposals carry most of the self-play signal, less on human.**
  Removing them drops self-play AUC 0.722 → 0.539 (a large effect) but human AUC
  only 0.646 → 0.623. So the phase-2 hypothesis — that the tell against
  GRAIL-evil lives substantially in rejection dynamics — is **supported** (and
  is distribution-specific: it helps the patterned GRAIL adversary more than
  varied humans). It is **not** "almost entirely" rejected dynamics; the
  below-chance figure was an evaluation artifact and is not cited.
- **Proposer does not transfer.** no_proposer raises self-play AUC 0.722 → 0.796
  (consistent across quests 2–5; paired-bootstrap CI [+0.058, +0.089]) but
  *slightly lowers* human AUC (0.646 → 0.641, CI excludes 0). A human
  proposer→evil correlation that misleads on heuristic GRAIL-evil — distribution
  shift, not a null feature. **Recommendation: drop proposer for a self-play
  target; down-weight/regularize rather than hard-delete if human play matters.
  Reconfirm with the self-play fine-tune / live runs.**

### Policy-penalty ablation (counterfactual, Track A on self-play-90)

The counterfactual replays the real `GoodPolicy` with the model's beliefs at
every historical good vote and measures evil-team reject-flips.

| `GRAIL_POLICY_OVERRIDES_GOOD` | evil-team flip | clean-team false-reject |
|---|---|---|
| default | **0.278** | 0.154 |
| off | 0.114 | 0.063 |
| no_rejected (zero rejected/failed penalties) | 0.266 | 0.151 |

By the plan's selection rule (maximize evil-team flips at ≤ +10 pts clean-team
cost): default beats off (+16.4 pts flip for +9.1 pts clean cost) → **keep the
default behavior-risk penalties**.

### LLM-vibes ablation (factor_v2, self-play-90)

| | belief AUC | mean P(evil) on true evil |
|---|---|---|
| vibes off (pure v2) | 0.598 | 0.500 |
| vibes recorded (live v2 + LLM priors) | **0.493** | 0.440 |

The LLM-vibes priors **degraded** the deployed detector below chance and pushed
mean belief on true evil the wrong way. (Recorded beliefs are the actual live
beliefs that drove the historical votes, so their counterfactual flip rate is
0 by construction — only the belief-quality columns are meaningful here.)
This supports `GRAIL_DISABLE_VIBES` for the calibrated card models; the live
question (whether vibes help a *good* detector) is deferred.

### Integrity

Split-leakage audit **PASSED**: the 12k training games are disjoint from the
14.6k full test split, and 0 of the 4000 scored test games appear in training;
the cache manifest confirms training used `proavalon_capped.json`.

## Gate-0 verdict

Rendered from an independent verification workflow (3 agents each tried to
*refute* a headline finding by reproducing it from scratch; 1 agent
synthesized). It reproduced every descriptive number, **refuted** the
"below-chance rejected-proposal collapse" as an eval artifact, **confirmed** the
no_proposer and calibration-bottleneck findings (the latter with a paired
bootstrap), and the leakage audit passed.

**GATE-0: PASS on the binding criterion.** Detection ranking is decisively
rebuilt above the phase-2 chance baseline on both self-play (0.598 → 0.722) and
held-out human data (0.65–0.69), with clean integrity and strong calibration
(ECE ≤ 0.04, T ≈ 1.0). The aspirational targets — AUC ≥ 0.75, mean P(evil|evil)
≥ 0.65 by quest 3, ≥ 40 % counterfactual flip — are **not** met by the
deployed-style configuration and remain open.

- **Promote Track A** as primary (best self-play AUC, and the track on which the
  ablation/counterfactual/calibration evidence was established). **Keep Track B
  as a tracked challenger** — it is actually *better on both human splits*
  (avalonlogs 0.687 vs 0.671, proavalon 0.652 vs 0.646) and has higher pair
  top-1 (0.345 vs 0.281). If human play becomes the target, Track B leads.
- **Drop the proposer feature for the self-play objective** (confirmed +0.073),
  flagging the small human-data cost (−0.006).
- **The deferred live runs are gated on a threshold-recalibration sweep.**
  Higher AUC did not yield more flips (no_proposer 0.796 AUC → 26.3 % flip <
  full 0.722 AUC → 27.8 %): ranking quality and flip rate are decoupled because
  the calibrated probabilities sit below Good's fixed 0.55–0.65 reject band.
  Sweep `reject_thresholds` (or temperature-sharpen outputs) on self-play-90 +
  human splits, trading evil-team flips against clean-team false rejects, before
  any live game. Until then the 40 % flip target is undemonstrated either way.
- **Rejected-proposal attribution: supported, not "almost entirely."** Under
  matched inference, rejected dynamics carry most of the self-play signal
  (0.722 → 0.539) and a small share on human (0.646 → 0.623).

## Threshold recalibration (pre-live, the blocker-clearing step)

The Gate-0 verdict's blocker — calibrated beliefs sitting below Good's v2-era
reject band — is an offline-fixable decision-boundary problem, not a detection
problem. `offline/threshold_sweep.py` replays the real GoodPolicy with Track A's
beliefs over a threshold × penalty grid on self-play-90 (the live distribution),
reusing one belief pass:

| config | evil-team flip | clean-team false-reject | flip/clean ratio |
|---|---|---|---|
| default (0.65–0.55), penalties default | 27.8% | 15.4% | 1.80 |
| flat 0.50, penalties default | 46.2% | 27.7% | 1.67 |
| **flat 0.45, penalties off** | **42.4%** | 22.9% | **1.85** |
| flat 0.50, penalties off (conservative) | 31.4% | 14.3% | 2.19 |
| flat 0.45, penalties default | 55.8% | 34.0% | 1.64 |

The evil mass sits at ~0.48/0.53/0.62 in quests 3/4/5 vs the good mass at
~0.35/0.31/0.26 — a clear window. Lowering the bar into it converts the AUC edge
into flips: default 27.8% → **flat 0.45 cleans the plan's 40% target (42.4%)**.
**Penalties-off gives a better flip/false-reject ratio at every threshold** (the
card model already ingests the rejected/failed-quest signal the penalties
re-add — the WS4 double-count). **Locked operating point: `reject_thresholds`
flat 0.45 + behavior-risk penalties off** (`flat 0.50/off` is the conservative
fallback if the smoke shows hammer blowup).

Two honest caveats the offline metric can't resolve, deferred to the live smoke:
the flip/false-reject numbers ignore the **hammer dynamic** (each clean-team
reject advances `failed_party_votes` toward the 5-reject evil auto-win), and
they replay *historical* (v2-proposed) teams — live, the better model also
*proposes* cleaner teams, an upside the counterfactual can't see.

The operating point is wired into `evaluation/phase3_gate1.json` and the C2/C4
cells of `phase3_grid.json` (`GRAIL_BELIEF_MODEL_GOOD=factor_v3:v4_trackA_proavalon`,
the recalibrated `GRAIL_POLICY_OVERRIDES_GOOD`, evil frozen).

## Gate-1 smoke result (live, 2026-06-19) — PASS

Run `evaluation/phase3_live_runs/20260619T033023Z`: 30 games of c2 (promoted
Good = factor_v3 Track A + flat-0.45 reject thresholds + behavior-penalties off)
vs frozen GRAIL-evil (factor_v2), concurrency 3, 90-min per-game timeout.

| Gate-1 criterion | result | verdict |
|---|---|---|
| no crashes / abandonments | 0 / 30 failed; durations 19–85 min (avg 45) | PASS |
| Good win rate ≥ +10 pts over 23 % baseline | **40.0 % (12/30)** = **+17 pts** | PASS |
| ≥ ~10/30 Good wins | 12 / 30 | PASS |

The recalibrated operating point converts the offline AUC edge into a decisive
live win-rate lift (23 % → 40 %). The early 71 % (n=7) was small-sample noise;
40 % is the settled rate.

**Loss-mode attribution (all 30 games):** every one of the 18 Evil wins was a
legitimate **3-failed-quest** loss; **zero hammer auto-wins** (terminal
`failed_party_votes` ranged 0–4, never the 5 that triggers the reject auto-win).
Four games reached fpv=4 — real reject pressure from flat-0.45 — but none tipped
into a hammer. This **resolves the threshold sweep's deferred hammer caveat**:
flat-0.45/penalties-off is hammer-safe in practice, so the conservative fallback
(flat 0.50/off) is **not** needed. When Good loses it is because the detector
still doesn't keep Evil off quest teams late-game (consistent with the offline
calibration-compression finding), not because of reject spirals.

**Operational note:** the first Gate-1 attempt (`20260618T061557Z`) deadlocked —
not a game/model fault (all its games completed and recorded winners) but an
orchestration one: the launcher was run as a session-bound background task and
got reaped mid-run, skipping `parallel_games.run_game`'s in-process
`finally: docker compose down`; two completed games' containers leaked and held
both slots + ~7 GB RAM for ~20 h. Fix: `evaluation/run_gate1_smoke.ps1` wrapper
with a pre-launch reaper (force-removes stale `avalon-*` containers + prunes
`.parallel_workspaces`), run detached/babysat rather than session-bound. The
per-game timeout + teardown were verified sound (a 120 s-timeout test abandoned
its game and reaped to zero) — they only protect while the launcher is alive.

## Gate-2 confirmation result (live, 2026-06-20) — PASS

Run on a 64 GB / 18-core Apple-Silicon Mac (Gate-1 was the Windows host; game
outcomes depend only on config + prompts + the DeepSeek `deepseek-v4-flash`
backend, not the host, so the two runs pool validly). **C2 = the 30 Gate-1 games
+ 62 new completed games here** (identical frozen `evaluation/phase3_gate1.json`:
promoted Good = factor_v3 Track A + flat-0.45 reject thresholds +
behavior-penalties off; frozen GRAIL-evil = factor_v2), tested against **C1 = the
phase-2 90-game baseline (21/90, 23.3 %)** with a two-proportion z-test at
α = 0.05.

| Gate-2 metric | result |
|---|---|
| new Good wins (this machine) | **35 / 62 = 56.5 %** |
| **C2 pooled Good win rate** | **47 / 92 = 51.1 %**  (95 % CI 40.9–61.3 %) |
| C1 baseline | 21 / 90 = 23.3 % |
| two-proportion z-test | **z = 3.87, p = 0.0001 — significant** |
| aspirational target (≥ 45 %) | **cleared** |

The recalibrated factor_v3 detector roughly **doubles Good's self-play win rate**
(23.3 % → 51.1 %) against frozen Evil, decisively confirming the Gate-1 trend
(40 %, p ≈ 0.07 at n=30) at n=92. The sub-runs were 3/3, 26/52, and 6/7 Good —
the low/mid-50s rate is stable across batches.

**Loss-mode attribution (62 new games):** all 27 Evil wins were legitimate
**3-failed-quest** losses; **zero hammer auto-wins** (`failed_party_votes` never
reached 5), reproducing the Gate-1 hammer-safety finding at larger n. Flat-0.45 /
penalties-off remains the operating point.

**Operational notes (this run):** launched at concurrency 15 — measured ~3.1–3.3
GiB/game (46.9 GiB of the 54.8 GiB Docker VM) after raising Docker Desktop's
memory on the 64 GB host; CPU is not the bottleneck (games are DeepSeek-bound).
5 of the first-batch games hung and were abandoned at the 90-min per-game timeout
— all in batch 1, a transient DeepSeek load spike when 15 games fire their first
calls at once (a gentler concurrency-7 top-up went 7/7 with no hangs) — so they
are excluded as technical failures, not game outcomes. macOS has no `setsid`, so
the launcher (`evaluation/run_gate2.sh`, the bash port of the Windows `.ps1` with
the same pre/post `avalon-*` reaper) was detached via a double-fork daemon
(`evaluation/daemonize.py`) to survive independent of the agent session; the
per-game timeout + `finally: docker compose down` again held (0 leaked
containers). Total DeepSeek cost: **$5.47** (~$0.08/game). Pooling, z-test, and
loss-mode classification: `evaluation/gate2_analysis.py`.

## What remains

- **Gate 2 — DONE** (see the Gate-2 section above): C2 = 92 pooled games,
  **51.1 % vs the 23.3 % baseline, z = 3.87, p = 0.0001**, hammer-safe (0/27 Evil
  wins were hammer). Clears the 45 % aspirational target. The C3/C4 forced-fifth
  cells remain optional exploratory follow-ups.
- **Self-play fine-tune** and **proposer-free retrain** remain deferred (need the
  nightly self-play campaign / a schema change), as detailed below.

Still genuinely deferred (need more data / a schema change, not just time):

- **Self-play fine-tune** (`*_ft`): the corrective for the human↔self-play shift
  (plan risk #1). Needs the nightly self-play campaign to accumulate games; must
  NOT train on the 79-game self-play-90 eval gate.
- **Proposer-free retrain**: confirm "drop proposer" on the self-play
  distribution before baking it into the deployed schema.

## Reproduce

```
# corpus (reads dataset/six_player_games_latest_compressed.json)
python code/agent/our/training/dataset/build_corpus.py --proavalon <dump> \
    --avalonlogs E:\Local\avalonlogs_6p --selfplay <phase2-runs-root>
# training (Track A then B, shared cache)
bash code/agent/our/training/run_proavalon_training.sh
bash code/agent/our/training/run_proavalon_ablations.sh
# offline eval matrix
cd code/evaluation && python -m offline.run_eval_matrix --proavalon-test-cap 4000
```
