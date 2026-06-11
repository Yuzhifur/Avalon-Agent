# Phase 2 Results: Evil-GRAIL Policy Tuning

**Status: complete (2026-06-10).** Phase 2's goal ("improve win rate without retraining the graph", see `grail_improvement_ideas.md`) is achieved structurally: Evil GRAIL wins ~77% of self-play games on the default heuristic, and further policy tuning has no measurable effect. The interesting output of phase 2 is the *diagnosis* of why — which redirects phase 3.

## Experiment setup

- Overnight grid run `code/evaluation/policy_runs_overnight/20260608T180435Z/` via `code/evaluation/tune_policy.py` (grid: `code/evaluation/evil_overnight_grid.json`).
- 3 Evil policy candidates × 30 games each, all-GRAIL self-play (4 good Servants, 2 evil Minions, 6 players, no special roles). Good side at default policy throughout.
- Some runs lost to OOM/timeout (Docker 7.8 GB; concurrency 2 is the safe max — see memory note), leaving n = 25–28 scored runs per arm.

## Win rates

| Candidate | Overrides | Scored runs | Evil wins | Evil win rate |
|---|---|---|---|---|
| `evil_baseline` | none | 26 | 20 | **76.9%** |
| `evil_blend_aggressive` | `clean_team_cover_approve_rate: 0.8`, `clean_team_reject_through_quest: 0` | 28 | 22 | **78.6%** |
| `evil_fail_half` | `midgame_fail_rate: 0.5` | 25 | 19 | **76.0%** |

**Read: the knobs are inert.** A 1.7 pp spread at n≈26/arm is far inside the binomial noise band (95% CI half-width ≈ ±16 pp). The earlier n=1 "blend-in hypothesis" (aggressive clean-team approval helps Evil) is **not confirmed** at scale — blend_aggressive ≈ baseline. Consequence: `DEFAULT_POLICY_CONFIG` in `code/agent/our/policy_models/heuristic.py` stays unchanged, and Evil is frozen as the fixed adversary for phase 3 (model `our/models/v2` + default `EvilPolicy`).

## Loss-attribution diagnostic

To decide where phase-3 effort should go, every Good loss across the 90 games was decomposed using the agents' `POLICY_DECISION` logs (which record per-player `party_risk` beliefs, `failed_party_votes` pressure, and each approve/reject vote). Across **61 Good-loss games**, there were **893 instances of a good player approving an evil-laden team**, classified as:

| Cause | Share | Meaning | Lever |
|---|---|---|---|
| **Misidentified** | **67%** | Good's belief never flagged the evil member (risk below threshold) | Detection model |
| **Forced** | **30%** | Approved at hammer pressure (`failed_party_votes ≥ 3`) under the 5-reject evil auto-win | Game rule (hammer) |
| **Threshold** | **3%** | Suspected the evil member but approved anyway | Policy thresholds |

Supporting statistic: Good's belief that a **true evil** teammate is evil averaged **0.555** (median 0.507; chance = 0.50), and 41% of the time a true minion was rated *more likely good than evil*. The deployed detector is near chance against GRAIL-evil's blend-in behavior, even though Good's policy logic is sound (Good rejects suspected-evil teams 3.4× more often than clean ones — when it suspects correctly, it acts correctly).

## Conclusions → phase 3

1. **Stop tuning Evil.** Knobs are inert; Evil at 77% is a strong fixed adversary with headroom for Good to improve against.
2. **Stop tuning Good thresholds.** They address 3% of the loss volume; tuning thresholds optimizes a belief that is wrong 67% of the time.
3. **Phase 3 = rebuild Good's detection model.** 67% of losses are pure belief failure, and most of the 30% "forced" bucket is downstream of the same failure (Good gets pushed to the hammer because it cannot assemble a team it believes is clean). Realistically ~80%+ of loss volume traces to detection.
4. **Hammer rule as evaluation axis.** Even a perfect detector is structurally forced to seat evil in some fraction of games under the 5-reject auto-win; a milder variant (5th proposal force-approved) bounds that cap. The current rule stays the default.

The full phase-3 plan is in `phase3_plan.md`.
