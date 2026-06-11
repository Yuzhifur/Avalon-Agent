# Phase 3 — Days 1–3 Status (2026-06-11)

Implements the plan's "Days 1–3 (before data)" block (`phase3_plan.md`):
Workstreams 0 and 1 complete with both hard gates passed, Tracks A and B
running end-to-end on dev data, hammer toggle and A/B plumbing in place.
ProAvalon has not arrived; nothing here is a final model number, and no
sweeps or live evals were run.

## Workstream 0 — offline replay harness: GATE PASSED

`code/evaluation/offline/`: `replay_eval.py` (CLI), `metrics.py`,
`model_adapters.py` (BeliefBackend protocol; factor_v2/factor_v3/seq_v1),
`counterfactual_policy.py` (replays the real `HeuristicOracle(GOOD)`),
`phase2_artifacts.py`, `reproduce_phase2.py`, `splits/`.

`python -m offline.reproduce_phase2 --cross-check` reproduces every phase-2
figure from committed artifacts (server JSONs + belief CSVs + the real
policy code):

| target (docs/phase2_results.md) | reproduced |
|---|---|
| 79 completed / 61 Good-loss games | 79 / 61 exact |
| 893 good approvals of evil-laden teams | 893 exact |
| 67% / 30% / 3% buckets | 598 / 269 / 26 = 67.0 / 30.1 / 2.9% |
| mean 0.555, median 0.507 belief on true evil | 0.5552 / 0.5075 |
| 41% of true evils rated more likely good | 41.04% |
| 3.4× rejection of suspected vs clean teams | 3.36× |

Three structural validations all came back clean: the replayed GoodPolicy
vote matches **every** historical good vote (the policy is deterministic
given beliefs + context); the recomputed behavior risks match the original
`POLICY_DECISION_DETAIL` logs bit-exact; every CSV belief trace aligns 1:1
with the server-log vote sequence.

**Corrected record:** the "lost DEBUG logs" caveat applied only to the Mac
checkout — this Windows machine ran the grid and has all 540 logs. Their
decision streams are now committed as `policy_decisions_*.jsonl` per run
(2.7 MB total), and `tune_policy.py` archives them automatically for every
future run (log-retention rule).

**Recovered methodology** (now encoded in `reproduce_phase2.py`): the doc's
"belief" = the policy's behavior-adjusted `party_risk` over true-evil party
members at every good-ego vote of the 61 evil-win games (n = 2500);
"forced" = approvals at `failed_party_votes ≥ 3`; "threshold" = non-hammer
approvals with a true-evil member at risk ≥ 0.55 (the quest-5 base
threshold — suspected at the policy's most aggressive bar, approved because
the current quest's bar was higher).

## Workstream 1 — schema, parsers, shared cards: PARITY PASSED

`code/agent/our/training/dataset/`: `event_schema.py` (GameRecord JSONL),
`parse_avalonlogs.py`, `parse_selfplay.py`, `parse_proavalon.py` (written
against the known `voteHistory` format, iterates **all** proposal indices;
validated on arrival), `augment.py`, `build_corpus.py`.

`code/agent/our/proposal_cards.py` — the single shared featurizer (25 slots
× 34 dims: proposer/team/votes one-hots rolled target-centric, accepted,
forced, present, quest + proposal-index one-hots, outcome). Consumed by the
dataset pipeline, the offline harness, and `GameInfo` at runtime.

Tests (`code/agent/test_proposal_cards.py`, 13 passing): pipeline-vs-runtime
card equality on synthetic and real logs, roll-by-1 ×6 identity and
rotation equivariance, forced-fifth encoding through all three paths,
legacy 21-vector compatibility against both `generate_dataset_1` and the
real `GameInfo.get_state_vector`. The GameInfo-replay parity check also
passed on **all 79/79** completed phase-2 games.

Dev corpora (`build_corpus.py`): avalonlogs **1,777 games** after the plan's
role filter (the scrape has Mordred in ~43% of games — dropped per risk #2;
still ~12× the dev-data size the plan assumed) with 70/15/15 splits at seed
20260610; selfplay90 = 79 games, eval-only. Stats + splits committed;
corpora gitignored (rebuildable).

Supporting refactor: `agent_enums.py` now holds ATEAM/AROLE/LLM
(re-exported from `agent_base`), so the belief/policy stack imports without
LLM dependencies.

## Tracks A & B — end-to-end on dev data (plumbing validation ONLY)

Both trained on the avalonlogs train split (1,243 games), early-stopped on
the human val split, temperature-scaled, and replayed through the harness.
**These numbers validate plumbing, not models — nothing here transfers to
the ProAvalon-trained models and no design decisions were made from them.**

| selfplay-90 replay (vibes off) | factor_v2 (deployed arch) | Track A `factor_v3` | Track B `seq_v1` |
|---|---|---|---|
| pooled AUC | ~chance (live ≈ 0.55 incl. vibes/penalties) | **0.720** | 0.642 |
| ECE | — | 0.026 | 0.028 |
| evil-pair top-1 (10 hyps) | — | 0.264 | 0.239 |
| counterfactual evil-team reject flip | — | 33.2% | 29.0% |
| clean-team new false rejects | — | 14.6% | 15.7% |
| eval wall-clock | ~90 s/game (loopy BP) | 5.9 s total | 7.7 s total |

- Track A: `ProposalSetDistribution` (8.5k params) + `FactorGraphModelV3`
  (closed-form enumeration over the 15 evil pairs — no pomegranate). Val
  acc 0.709, T = 1.026 (native calibration without pos_weight, as
  intended). Human-trained, yet 0.720 AUC against GRAIL-evil self-play it
  never saw — early signal that rejected-proposal features carry across the
  human↔self-play shift (plan risk #1).
- Track B: `CardSeqGRU` (2×128, 271k params) + 15-way pair head. Val pair
  top-1 0.336 vs 6.7% chance (plan's sanity bar passed). Overfits after
  ~10 epochs at 1.7k games — consistent with the plan's low-data note;
  the real comparison waits for ProAvalon.
- factor_v2 full-replay baseline on selfplay-90 is running (slow BP); the
  3-game smoke gave mean P(evil) on true evil ≈ 0.486 vibes-off (near
  chance, as diagnosed).
- In-domain (avalonlogs test split, 268 games): Track A AUC 0.666 /
  ECE 0.046, Track B AUC 0.662 / ECE 0.023 — nearly tied on human data
  (and in the same ballpark as the prior card-model experiment's 0.64
  proposal-level AUC), while Track A transfers visibly better to the
  self-play distribution.

## Workstream 5 — hammer toggle (default unchanged)

`AvalonGame.ts`: `hammer_rule` = `"evil_win"` (default) | `"forced_fifth"`,
from `game.hammer_rule` config or `AVALON_HAMMER_RULE` env. Under
`forced_fifth`, the fifth proposal skips the party vote: marker message
("The fifth proposal is forced through!") → synthetic unanimous vote
summary → standard approval message, then quest votes. Agents record
`forced=true` via the marker (`GameInfo.add_party_proposal`), parsers set
the event flag, cards carry the bit, and `build_corpus.py` keeps forced
games out of training corpora. Server file type-checks clean; a live
scripted game under each rule value is deferred to the Gate-1 smoke (needs
the Docker stack + LLM keys).

## Workstream 4 — A/B plumbing

`our/model_factory.py`: `GRAIL_BELIEF_MODEL_GOOD`/`_EVIL`/fallback
(default `factor_v2`; `name:dir` syntax for alternate checkpoints).
`ACLAgent` now builds its model through the factory and feeds card-based
models the full proposal history through the shared featurizer;
`GRAIL_DISABLE_VIBES` skips the LLM prior. All six agent services +
the server get the new env passthroughs in `docker-compose.yml`;
`tune_policy.py` candidates accept an `"environment": {...}` dict and the
eval cells are declared in `phase3_grid.json` (C1–C4, **not run**).

## Deferred / next

- ProAvalon arrival: validate `parse_proavalon.py`, `build_corpus.py
  --proavalon` (chat check included), full training both tracks + self-play
  fine-tunes, penalty/vibes ablations, rejected-proposals attribution
  checkpoint, promotion, Gates 1–2.
- Nightly self-play campaign: ready to start (runner now archives decision
  logs); not started this session.
- No sweeps on dev data; no Track A/B promotion decision; no live runs.
