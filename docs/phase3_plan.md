# Phase 3 Plan: Rebuild Good's Detection Model

**Primary goal:** rebuild Good's evil-detection model so it is decisively better than chance, and convert that into Good self-play win rate against frozen GRAIL-Evil under current rules.

Rationale and evidence: see `phase2_results.md`. The phase-3 items in `grail_improvement_ideas.md` (rejected-proposal history, proposer identity, richer sequence features, retrain neural factors, compare vs current GRAIL) target exactly the diagnosed failure. Key data fact: both raw datasets already contain every rejected proposal and proposer — ProAvalon's `voteHistory` carries `VHleader/VHpicked/VHapprove` per proposal, and avalonlogs' `missions[].proposals[]` carries proposer/team/votes — the current vectorizers (`generate_dataset_2.py` reads only proposal `[-1]`; `GameInfo.get_state_vector` likewise) simply discard them. The work is pipeline + model inputs + retraining, not new signal.

Both model tracks share one input representation: the **proposal-card schema** (Workstream 1), which replaces opaque combination-index categoricals with structured binary cards. This realizes ideas #16 (richer game-state inputs) *and* #17 (replace categorical encodings with general encodings) from the ideas doc in one move, and goals 1–3 of the doc's phase 3 become three evaluable increments of a single representation change rather than independent edits.

**Fixed decisions**

1. Two model tracks, compared **offline** before any live games: Track A = extended factor graph (doc-faithful), Track B = sequence-model challenger. Winner promoted.
2. Hammer-rule variant ("forced fifth": 5th proposal force-approved instead of evil auto-win) added as a server toggle and used as an **evaluation axis**; current rule stays default.
3. Training: human data primary; a self-play fine-tuned variant trained and reported separately. Model selection / early stopping / calibration on **human-only val**; self-play enters base training only as targeted augmentation if the corpus coverage analysis (rejected-count / proposal-depth histograms from `build_corpus.py`) shows gaps in the states the new features create.
4. Evil stays **frozen** (factor_v2 model + default `EvilPolicy`) as the fixed adversary; only Good gets the new model in eval.

**Success criteria**

- Offline (replay of the 90 phase-2 self-play games): pooled ROC-AUC ≥ **0.75** (baseline ≈ 0.55); mean P(evil) on true evil ≥ **0.65** entering quest 3; ECE ≤ 0.10; counterfactually flips ≥ **40%** of historically-approved evil teams to reject, at ≤ +10 pts new false rejections of clean teams.
- Live (current hammer rule): Good win rate ≥ **45%** (vs 23% baseline), n = 70/arm, significant at α = 0.05.
- Diagnostic: misidentified share of Good losses drops 67% → ≤ 40%.
- Secondary: quantify how much win rate the forced-fifth rule adds on top of the better model (the structural cap).

**Constraints:** local avalonlogs = 142 six-player games → ProAvalon (~101k games, arriving shortly) is the real training set. Self-play generation ≈ 30 games/night (Docker concurrency 2). RTX 4060 suffices (models are tiny); cluster only for sweeps > ~20 runs.

---

## Workstream 0 — Offline replay-eval harness (build FIRST)

Everything is judged through this; no new-model number is trusted until the harness reproduces the known phase-2 diagnostics.

New package `code/evaluation/offline/`:

- `replay_eval.py` — CLI: `--games <dir|jsonl> --model {factor_v2,factor_v3,seq_v1} --penalties {default,off,no_rejected} --vibes {off,recorded} --out report.json`
- `metrics.py` — AUC / ECE / Brier / pair top-k / mean-belief-by-quest
- `model_adapters.py` — `BeliefBackend` protocol: `predict(events, ego_index, ego_role) -> {seat: p_evil}`
- `counterfactual_policy.py` — instantiates `HeuristicOracle(GOOD)` against replayed beliefs, replaying `proposal_history` in the same dict shape as `GameInfo.policy_proposal_history()`

Details:

- Parse server logs with the message patterns already used by `code/evaluation/log_analyzer.py`; ground-truth roles from player entries.
- Metrics per model and eval set: per-prefix ROC-AUC (pooled + by quest), evil-pair top-1/top-3 over the 15 pair hypotheses, mean P(evil) on true evil by quest, ECE (10 bins) + Brier, counterfactual GoodPolicy reject-flip rate on historically-approved evil teams + clean-team false-reject cost.
- Eval sets: (a) self-play-90 (primary gate), (b) avalonlogs-142 test split, (c) ProAvalon 6p test split when available. Game-level splits, fixed seed, stored in `code/evaluation/offline/splits/`.
- **Acceptance test:** the `factor_v2` adapter replayed over the 90 phase-2 games must reproduce ≈0.555 mean belief on true evil, 893 evil-team approvals, ≈67/30/3 loss attribution.

## Workstream 1 — Unified event schema + data pipeline

In `code/agent/our/training/dataset/`:

- `event_schema.py` — `GameRecord {game_id, source ∈ {avalonlogs, proavalon, selfplay}, seats×6, evil×6, winner, win_reason, variant_roles, events}` with `ProposalEvent {quest, round_idx, leader_seat, team_seats}`, `VoteEvent {approves×6, accepted, forced}`, `QuestEvent {quest, success}`. JSONL, one game per line (required for the ~101k-game ProAvalon file).
- `parse_avalonlogs.py` — `missions[].proposals[]` → events; filter 6-player; roles from `outcome.roles`.
- `parse_proavalon.py` — reuse the `VHleader/VHpicked/VHapprove` decoding from `generate_dataset_2.py` but iterate **all** proposal indices per mission (not `[-1]`), emitting rejected proposals; alliance from `playerRoles`; record `variant_roles`; keep the `howTheGameWasWon` hammer handling. Written now against the known format, validated on data arrival.
- `parse_selfplay.py` — server log JSON → GameRecord (same patterns as the harness).
- `augment.py` — cyclic permutations (×6, reuse `cyclic_perm`) + **proposal-boundary snapshots** (one training example after every `VoteEvent`, future cards masked out; supersedes `partialize_vector` — mid-quest rejected proposals are exactly the new signal).
- `build_corpus.py` — emits `data/corpora/*.jsonl` + stats report (6p counts, role-variant breakdown, proposals-per-quest histogram, rejected-count/proposal-depth coverage histograms, **chat-presence check for ProAvalon**).

**`proposal_cards.py` — the shared card featurizer (single source of truth; train/serve skew killer).** Today the 21-int vector is built twice (dataset scripts vs `GameInfo.get_state_vector`); this module is imported by the dataset pipeline, the offline harness, AND `GameInfo` at runtime. GameRecord → fixed tensor of **25 ordered slots** (5 quests × ≤5 voted proposals; the reject counter resets on acceptance, so 5/quest is the hard max). Each card, encoded **target-centric** (all player vectors rolled so the factor's target player sits at index 0):

| Field | Encoding | Width |
|---|---|---|
| proposer | one-hot, rolled | 6 |
| team membership | binary, rolled | 6 |
| votes (approve = 1) | binary, rolled | 6 |
| accepted | binary | 1 |
| forced (forced-fifth variant) | binary | 1 |
| present (slot mask) | binary | 1 |
| quest number (1–5) | normalized scalar / small embedding | 1–2 |
| proposal index within quest (1–5) | normalized scalar / small embedding | 1–2 |
| quest outcome {unknown, fail, success} | one-hot | 3 |

≈ 26–28 dims per card; input tensor `[batch, 25, card_dim]` + mask. Binary membership vectors expose team/vote **overlap** to the net (teams {1,3} and {1,3,4} are similar, not unrelated symbols — unlike the current 16/21-way combination indices), and the target-centric frame is a trivial vector roll instead of the combination-index remapping in `permutate_vector_for_player_order`. Keep the ×6 seat-rotation augmentation — the binary encoding removes the index-lookup pain, not the need for the invariance prior. Side benefit: ProAvalon hammer games, whose final mission the current vectorizer discards, become usable signal (rejected proposals are real votes).

Both tracks and the harness consume only GameRecord/cards. Live agents produce identical events from `GameInfo.quest_proposals` (already records every proposal with leader/votes/accepted, `code/agent/agent_acl.py:35-50`) → train/serve parity, verified by a unit test (cards built from a server log via the pipeline vs via `GameInfo` must be identical).

## Workstream 2 — Track A: card-set neural factors in a simplified factor graph (doc-faithful)

The factor graph survives; only its neural factors' *inputs* change, from 15 opaque categoricals to the 25 proposal cards. This is lossless where the old plan's aggregate-count encoding was lossy: cards preserve the rejected-team composition × voter × proposer interaction — precisely the tell needed against the 67% misidentification bucket (GRAIL-evil *rejects* clean teams; human evil *approves* doomed evil-laden teams).

- **Encoder `ProposalSetDistribution`** (new class alongside `EgoNeuralDistribution`, which stays intact for the frozen v2 baseline — its `model` is a class-level singleton, so old and new must be separate classes): shared per-card MLP (≈28→64→64) → **masked mean-pool over the 25 cards** → head MLP (64→32→1) → one logit = evidence(target = evil). ≈ 10–15k params; trains in minutes on the 4060. Position fields inside each card preserve order under pooling. (Attention/recurrence over cards is deliberately NOT in Track A — Track B *is* that upgrade, so the A/B comparison isolates whether order matters beyond position features.)
- **Graph simplification — collapse the history nodes.** In `FactorGraphModelV2` the 15 history variables are always observed, so each neural factor already reduces to a unary potential on its role variable. `FactorGraphModelV3` (`code/agent/our/model_v3.py`) therefore becomes: **6 role marginals + 6 card-encoder unary evidence factors + the exactly-2-evil constraint** — mathematically identical inference, cheaper, and it deletes the riskiest plumbing (no 22 new observed categorical nodes, no factor-edge-insertion-order hazard, no generalization of `permutate_vector_for_player_order` — the target-centric frame is a vector roll in `proposal_cards.py`). Optionally skip BP entirely: posterior is closed-form, pair(i,j) ∝ exp(eᵢ+eⱼ) over the 15 valid evil pairs, marginals by summation; LLM-vibes `update_priors` becomes prior reweighting of the pairs (same mechanism as Track B's wrapper — the constraint/calibration code can be shared).
- Trainer `train_track_a.py` (scaffold cloned from `train_main.py`: BCE + pos_weight, early stopping, weight decay, temperature scaling). Models → `code/agent/our/models/v4_trackA/` and `v4_trackA_ft/` (self-play fine-tune). v2 weights and classes remain untouched for A/B.
- Live side: `GameInfo` feeds `quest_proposals` through the shared `proposal_cards.py` — no second vectorizer.
- Optional derived per-target fields (add ONLY if Stage-A eval shows the pooled encoder underfitting): "approved-a-later-rejected-team" count, "vote flipped across proposals" flag.

## Workstream 3 — Track B: sequence-model challenger

New package `code/agent/our/seq_model/` (`tokenizer.py`, `model.py`, `train.py`, `calibrate.py`) + wrapper `code/agent/our/seq_belief_model.py`:

- **Input = the same proposal cards** (one card per step, in order, from the shared `proposal_cards.py` builder, ego-centric roll) plus a running-score field; each card linearly projected → 128-d step embedding; sequences ≤ ~30 steps. No separate tokenizer to maintain.
- **Encoder:** 2-layer GRU, hidden 128 (~250k params) — right size for a tens-of-thousands-of-games regime. Vs Track A, this isolates two questions: does order matter beyond per-card position features (pooling vs recurrence), and does a joint pair head beat independent per-player evidence + constraint.
- **Head:** 15-way softmax over evil-pair hypotheses (structural exactly-2-evil; per-player marginal = Σ pairs containing the player). Ego role enters as input feature and as an inference-time pair mask.
- **Training:** cross-entropy on the true pair, all vote-level prefixes, ego seat sampled per example (rotation augmentation for free). ~1.8M samples → well under 2 h on the 4060. Temperature-scale the 15-way logits on val (reuse `temperature_scaling.py` pattern).
- **Interface:** `SeqBeliefModel` implements the same surface as `FactorGraphModelV2` (`construct` / `load_from_file` / `predict_probs` / `update_priors`); `update_priors` (LLM vibes) reweights the pair distribution by member prior odds.

## Workstream 4 — Integration & A/B plumbing

- `code/agent/our/model_factory.py`: `build_belief_model(team)` reading `GRAIL_BELIEF_MODEL_GOOD` / `_EVIL` / fallback `GRAIL_BELIEF_MODEL` (default `factor_v2`), mirroring the `load_policy_config` env pattern. Replace direct `FactorGraphModelV2()` construction in `ACLAgent.__init__` (`agent_acl.py:131-133`); `update_predictions` passes event history; the factor_v2 backend builds the legacy 21-vector internally.
- Compose: pass `GRAIL_BELIEF_MODEL*` + `GRAIL_DISABLE_VIBES` through all 6 agent services in `code/docker-compose.yml`; `AVALON_HAMMER_RULE` to the server service. Extend `code/evaluation/tune_policy.py` candidates with an optional `"environment": {...}` dict; eval cells declared in `code/evaluation/phase3_grid.json`.
- **Behavior-risk penalty double-counting:** `GoodPolicy._behavior_risk` adds mechanical penalties for exactly the events the new models now ingest. Offline ablation (defaults / all good penalties zeroed / only rejected-proposal+failed-quest penalties zeroed) via `GRAIL_POLICY_OVERRIDES_GOOD`; selection rule: maximize counterfactual evil-team rejection at ≤ +10 pts clean-team cost; winner carried into live runs.
- **LLM-vibes ablation:** `update_predictions_based_on_chat` applies crude 0.6/0.75 priors that may hurt a calibrated model. Offline: vibes-off vs recorded-vibes. Live: `GRAIL_DISABLE_VIBES` flag if offline says off.

## Workstream 5 — Hammer-rule toggle (eval axis; default unchanged)

- `code/phaser/server/src/rooms/AvalonGame.ts` (current rule ~lines 556–561): add `hammer_rule: "evil_win" | "forced_fifth"` to server config, overridable by `AVALON_HAMMER_RULE`. Under `forced_fifth`: a proposal at `failed_party_votes === 4` skips the party vote — emit the standard `"The party has been approved!"` (parser compatibility) plus a distinct marker message, then proceed to quest votes. Parsers set `forced: true`; Track A maps forced votes to the unanimous composition; Track B gets the `forced` bit.
- Policy consistency: Good's auto-approve and Evil's auto-reject-5th branches simply never fire (the vote is skipped) — safe. Frozen Evil does NOT adapt to the variant; acceptable because the rule is an eval axis present in both model cells. `GameInfo.add_party_proposal` gains a `forced` flag.
- Keep variant games out of training data (no human data has this rule).

## Workstream 6 — Live win-rate evaluation (staged; evil always frozen)

| Cell | Good model | Rule | Games |
|---|---|---|---|
| C1 baseline | factor_v2 | evil_win | reuse phase-2 90 if config-identical (diff run env first), else 70 fresh |
| C2 primary | promoted model | evil_win | 70 |
| C3 | factor_v2 | forced_fifth | 30 (exploratory) |
| C4 | promoted model | forced_fifth | 30 (exploratory) |

Power: 23% → 45%, α = 0.05, 80% power → ~70/arm ≈ 2.5 nights at ~30 games/night (concurrency 2).

**Gates:** Gate 0 = offline winner beats factor_v2 on self-play-90 AND holds on the human split. Gate 1 = 30-game smoke (no crashes, latency OK, ≥ +10 pts point estimate). Gate 2 = complete C2; C3/C4 after. Final deliverable: `docs/phase3_results.md`.

**Incremental attribution checkpoint (offline, cheap):** before crediting the full card schema, run a rejected-proposals ablation — the trained model evaluated with rejected-proposal cards masked out vs. full input (and, if time permits, a model trained accepted-only). This isolates how much of the gain comes from goal 1 (rejected history) vs goals 2–3 (proposer + sequence), and is the doc's "compare against current GRAIL" done per increment rather than only end-to-end.

## Sequencing around ProAvalon arrival

**Days 1–3 (before data):** event schema + all three parsers → replay harness + factor_v2 adapter passing the 0.555 / 893 / 67-30-3 reproduction test → hammer toggle → Track A and Track B end-to-end on dev data (avalonlogs-142 + self-play-90; validates plumbing, not final numbers) → start nightly self-play generation campaign (~30/night) to grow the fine-tune/eval pool.

**On arrival:** corpus build + stats (6p count, role variants, chat presence) → full training both tracks + self-play fine-tuned variants → small sweeps on the 4060 (cluster only if sweep matrix > ~20 runs) → offline comparison + penalty/vibes ablations → promote winner → Gate 1 → Gate 2 → results doc.

**Fallback (ProAvalon delayed > 1 week):** train on avalonlogs-142 + accumulated self-play (target 200+); prefer Track A or a shrunk Track B (hidden 64, dropout) in the low-data regime; proceed through the gates — the live opponent is GRAIL anyway; treat ProAvalon as a later retrain.

## Risks

1. **Human↔self-play distribution shift** (human rejection dynamics ≠ GRAIL-evil's blend-in meta) — selection gate is self-play replay; human split reported alongside; the fine-tuned variant is the corrective; `app_rej` captures both directions of the tell.
2. **ProAvalon role pollution** (6p games virtually always have Merlin/Assassin; the current v2 model was already trained on them) — keep games with roles ⊆ {vanilla, Merlin, Percival, Assassin, Morgana}, label by alliance only, record `variant_roles` so a vanilla-only ablation is one filter away.
3. **Pooled card encoder may underfit order effects** — position fields mitigate; Track B (GRU over the same cards) is the built-in hedge, and optional derived per-target fields are the cheap fallback. (The former top risk here — pomegranate edge-order/permutation plumbing — is largely eliminated by the graph simplification; what remains is keeping the frozen v2 classes untouched alongside the new `ProposalSetDistribution`.)
4. **Forced-fifth log handling** — missing vote summaries must not break parsers/featurizers (handled in WS5).
5. **Reusing phase-2 games as C1** — only after diffing run env/config.

## Verification

- Harness reproduction test (0.555 / 893 / 67-30-3) before any new-model evaluation.
- `proposal_cards.py` parity test: cards built from a real server log via the dataset pipeline vs via `GameInfo` at runtime must be identical; target-centric roll test (roll-by-1 six times = identity; evidence for a given target invariant to non-target seat rotation up to augmentation).
- Trained-model sanity: Track A val accuracy/F1 vs old `train_main.py` numbers; Track B pair top-1 well above the 1/15 ≈ 6.7% baseline.
- Rejected-proposals ablation checkpoint (see Workstream 6) to attribute gains per increment.
- Hammer toggle: scripted game reaching 4 rejects under each rule value; confirm server messages and parser `forced` flag.
- Gate 1 smoke run inspects `POLICY_DECISION` logs to confirm the new beliefs actually drive votes.
- Final: C2 vs C1 two-proportion test; loss-attribution rerun on C2 losses to confirm the misidentified share dropped.
