# GRAIL / Bayesian Avalon — Project Overview

*Last updated 2026-06-20 · branch `phase3-gate2`. Sources: `README.md`, `code/README.md`, `docs/grail_beginner_guide.md`, `docs/phase2_results.md`, `docs/phase3_plan.md`, `docs/phase3_results.md`, and the paper arXiv 2506.17788.*

## TL;DR

GRAIL (the system behind the paper *"Bayesian Social Deduction with Graph-Informed Language Models,"* arXiv 2506.17788) is a **hybrid social-reasoning agent** for the hidden-role game Avalon. It pairs a **structured probabilistic factor graph** for belief inference with an **LLM** for language understanding/generation and a **hand-written heuristic** for game actions. In the original paper it matched much larger reasoning models in agent–agent play and became the **first language agent to beat humans** in a controlled study (**67% win rate**, rated higher than both reasoning baselines and human teammates).

Our follow-on work (Phases 2–3) studied GRAIL in **GRAIL-vs-GRAIL self-play** with **Evil held frozen**. Phase 2 diagnosed that Good was losing not to clever sabotage but because its **evil-detection was at chance** (true evil rated only ~0.50–0.555 P(evil); the old model's detection AUC was later measured at ~0.598), yielding a Good baseline win rate of ~23.3%. Phase 3 **rebuilt Good's detector** with a proposal-card model (`factor_v3`) that ingests rejected proposals, lifting detection AUC to **0.722** and — after **threshold recalibration** — roughly **doubling** Good's self-play win rate to **51.1%** (47/92, p = 0.0001) with **zero hammer auto-wins**.

> **Scope note.** Results are labeled **[PAPER]** (original arXiv contribution) or **[PHASE 2/3]** (our self-play work). The two are not directly comparable: the paper measures play against humans and larger models; Phases 2–3 measure frozen-evil self-play.

---

## 1. The problem: Avalon as a social-deduction benchmark

Avalon is a hidden-role social-deduction game. In the 6-player configuration used here, **4 Good Servants** and **2 Evil Minions** play **5 quests** with party sizes **2, 3, 4, 3, 4**. Each round a leader proposes a party; all six players vote to approve or reject (strict majority — more than 3 of 6); if approved, party members secretly play success/fail cards. **One Evil card fails a quest.** Three successful quests win for Good; three failed quests win for Evil. An assassination mechanic lets Evil flip a Good sweep by killing Merlin.

This makes Avalon a stress test for **social reasoning**: inferring **unobservable** roles, beliefs, and intentions from **partial observations** (who proposed whom, who approved, which quests failed) under **deception**, all mediated through **multi-agent natural-language interaction**. The paper's core finding **[PAPER]** is that large reasoning LLMs do well but need **extensive test-time inference** and **degrade sharply when distilled** to smaller, real-time variants — motivating a hybrid that externalizes belief inference.

---

## 2. The original GRAIL contribution and results [PAPER]

GRAIL's thesis is **division of labor**: keep belief inference in a **structured probabilistic model** (cheap, calibrated, trained only on structured game logs) and use the **LLM** only for what it is good at — **language understanding and interaction**, plus a soft "vibes" prior.

Headline results from the paper:

- **Competitive with much larger reasoning models** in Agent–Agent play, at lower test-time inference cost.
- **First language agent to defeat human players** in a controlled study.
- **67% win rate vs humans**, with **higher qualitative ratings than both reasoning baselines and human teammates**.
- GRAIL is compared against LLM **reasoning baselines**, including **ReCon** (Recursive Contemplation, Wang et al.; reproduced in `code/agent/recon/`). Separately, `code/README.md` notes the human-experiment reasoning agent was run on `o4-mini` (the repo's ReCon fork itself defaults to `gpt-4o-mini`; the docs do not state ReCon and the human-study reasoning agent are the same configuration).

The codebase ships code, models, and a human-game dataset to support further work.

---

## 3. System and architecture

GRAIL is a multi-service system; a single agent combines three loosely-coupled parts.

### 3.1 Agent components (`code/agent/agent_acl.py`)

1. **Belief graph** (`factor_v2` baseline, or Phase-3 `factor_v3` / `seq_v1`), selected via `GRAIL_BELIEF_MODEL_GOOD/EVIL`; default `factor_v2`.
2. **LLM backend** (DeepSeek `deepseek-v4-flash`, or any OpenAI-compatible endpoint) for message generation and **vibes priors**.
3. **Hand-written heuristic policy** (`code/agent/our/policy_models/`) for `propose_party`, `vote_party`, `vote_quest`.

### 3.2 The factor-graph belief model (`factor_v2`)

- **Variables:** 6 binary role marginals (r1–r6, Good = 0 / Evil = 1) plus 15 observed quest-history fields (P1–P5 party compositions, V1–V5 approval votes, O1–O5 quest outcomes) — a 21-field state vector.
- **Factors:** six **neural factor nodes** (f1–f6) estimate p(role_j | history); one `JointCategorical` **exactly-2-evil constraint** (the 15 valid evil pairs).
- **Neural factor (`EgoNeuralDistribution`):** a shared **egocentric** network — the input is circular-shifted so the target player sits at index 0, so one network serves all six factors — with categorical embeddings → Linear(45,16)+ReLU → Linear(16,16)+ReLU → Linear(16,1) logit → sigmoid. **~1,861 parameters** (incl. a temperature scalar). The "small model" point: the belief core is tiny and trained only on structured logs.
- **Inference:** loopy belief propagation (`factor_graph.py`), `max_iter = 20`, tol `1e-6`; v2 uses **max-product**.
- **Training:** only the neural factor is trained — the graph structure, belief propagation, the 2-evil constraint, the heuristic, and the LLM weights are **not**. It learns from structured game states only (no language), from a large corpus of human Avalon logs (the GRAIL beginner guide reports a ~104k-game corpus), with **temporal masking** to prevent future-quest leakage.

### 3.3 LLM integration

The LLM produces chat messages (five prompt templates in `prompts.py`) and a **vibes-prior** signal: it reads the game log, beliefs, quest history, and the current proposed party, then maps players to "increase"/"decrease" evil belief. Blending maps "increase" → P(evil) = 0.75 (quest ≥ 3) or 0.6 (quest < 3); "decrease" → 0.25 or 0.4. Vibes can be disabled via `GRAIL_DISABLE_VIBES`.

### 3.4 Heuristic policy

Good proposes lowest-risk players (risk = belief-evil + mechanical penalties), approves a party only if all members are below a quest-dependent threshold (originally **0.55–0.65**), and plays success. Evil includes a known-evil teammate with good cover and fails strategically. Behavior-risk adjusts base beliefs via failed-quest penalties (~0.06–0.12) and other history.

### 3.5 Deployment harness

Multi-service Docker Compose (`code/docker-compose.yml`): a **Phaser/Node.js browser client**, a **Colyseus game server** (`AvalonGame.ts`) that authoritatively tracks state/proposals/votes/outcomes, a **Python FastAPI agent manager**, and **per-seat agent containers**. A **parallel game harness** (`evaluation/tune_policy.py` → `parallel_games.py`) runs concurrent headless self-play games in isolated workspaces, logging per-game `status.json` + `runs.csv` / `summary.json`.

---

## 4. Phase 2 — the self-play diagnosis [PHASE 2]

Phase 2 ran **GRAIL-vs-GRAIL self-play** to ask why Good loses. An overnight grid tested 3 **Evil** policy candidates × 30 games (some lost to OOM/timeout; n ≈ 25–28 scored/arm).

**Finding 1 — Evil knobs are inert.** Evil win rates: baseline **76.9%** (20/26), blend_aggressive **78.6%** (22/28), fail_half **76.0%** (19/25). The **1.7-pt** spread sits inside binomial noise — Evil's ~77% dominance is **not** from clever sabotage.

**Finding 2 — Good's detection is at chance.** Across **61 Good-loss games**, **893** instances of Good approving evil-laden teams decompose as:

| Loss cause | Share |
|---|---|
| **Misidentified** (belief never flagged evil) | **67%** |
| **Forced** (hammer / 5-reject pressure) | **30%** |
| **Threshold** (suspected evil, approved anyway) | **3%** |

True evil teammates were rated evil at mean **0.555** (median 0.507; chance 0.50), and **41%** of true minions were rated more likely good than evil — i.e. detection at chance. (Phase 3's Gate-0 later put a single number on the old model: self-play detection **AUC ≈ 0.598**.) Good's *policy* is sound — it rejects suspected-evil teams **3.4×** more often than clean teams — but its *beliefs* are wrong ~67% of the time.

**Decision.** Stop tuning Evil; stop tuning Good thresholds (only 3% of loss); **rebuild Good's belief model**, since the large majority of loss volume traces to detection (much of the "forced" bucket is downstream of being unable to assemble clean teams). The Phase-2 Good baseline used as the Phase-3 control is **21/90 = 23.3%**.

---

## 5. Phase 3 — rebuilding Good's detection [PHASE 3]

### 5.1 The new representation: proposal cards (`factor_v3`)

The key idea is to feed the detector the **full rejected-proposal history**, not just final quest outcomes. The schema (`proposal_cards.py`) has **25 ordered slots** (5 quests × ≤ 5 voted proposals) of **34-dim** target-centric cards: proposer one-hot, team membership, approval votes, accepted/forced flags, presence mask, quest one-hot, proposal-index one-hot, and quest-outcome one-hot. Because v2's history nodes are always observed, the graph collapses to **6 role marginals + 6 neural unary factors + the 2-evil constraint**, enabling **exact posterior enumeration over the 15 evil pairs** (no loopy BP needed).

### 5.2 Two tracks

- **Track A — `factor_v3` `ProposalSetDistribution` (~8.5k params, PROMOTED):** per-card MLP, masked mean-pool over the 25 cards + a density feature, head logit; exact 15-pair enumeration; **sum-product** by default.
- **Track B — `seq_v1` `CardSeqGRU` (~205k params, reserve):** the same cards as a sequence → 2-layer GRU → 15-way softmax over evil-pair hypotheses; per-player marginal = sum over pairs containing the player.

### 5.3 Training (ProAvalon human dump)

Parsed **101,280** raw 6-player records → **100,645** valid (99.4%) → role-variant filter → **97,594** usable games, **1.55M** voted proposals. ProAvalon has **no chat**, so vibes cannot be trained (runtime prior only). Models trained on a seeded **12k/2.5k** game split past saturation within 16 GB RAM; selection/calibration on the human val split. **Honest test:** a disjoint ProAvalon split of **4000 games**, leakage audit 0/4000.

### 5.4 Gate-0 (offline, 2026-06-17)

| Metric (self-play-90) | `factor_v2` | Track A | Track B |
|---|---|---|---|
| Detection **AUC** | **0.598** | **0.722** | 0.694 |
| Pair top-1 (chance 0.067) | 0.112 | 0.281 | 0.345 |
| **ECE** (lower = better) | 0.088 | 0.035 | 0.023 |
| Mean P(evil) on true evil | 0.500 | 0.496 | 0.467 |

Held-out human AUC: Track A 0.671 / 0.646, Track B 0.687 / 0.652 (avalonlogs / ProAvalon) — both ≥ 0.65–0.69. **Calibration** improves markedly (factor_v2 ECE 0.088 → Track A/B ≤ 0.04, T ≈ 1.0). Detection **ramps with evidence** (Track A by-quest AUC: q1 0.51 → q2 0.50 → q3 0.70 → q4 0.81 → q5 0.82): early game is at chance, then rejection dynamics accumulate.

**Ablations (matched inference):**
- **Rejected proposals carry the signal:** removing them drops self-play AUC **0.722 → 0.539** (−0.183) but ProAvalon only **0.646 → 0.623** (−0.023) — a strong patterned tell against heuristic Evil, much smaller against varied human deception.
- **AUC and flip-rate are decoupled (a load-bearing finding):** dropping the **proposer** feature *raises* self-play AUC to **0.796**, yet the higher-AUC model actually **flips *fewer* evil teams (26.3% vs 27.8%)** and slightly lowers human AUC (0.646 → 0.641). Higher ranking quality ≠ a better agent — the lesson that drove the recalibration in §5.5. (Recommendation: drop proposer for a pure self-play target; down-weight, not delete, if human play matters.)
- **Vibes hurt the calibrated card model:** factor_v2 with vibes recorded sits at AUC 0.493 (below chance) — supporting `GRAIL_DISABLE_VIBES` for the card models.

Track A was **promoted** (best self-play AUC, full evidence base); Track B is the **challenger** (better on both human splits, higher pair top-1).

### 5.5 The unlock: threshold recalibration

Higher AUC alone produced **no** win-rate flips, because the calibrated evil probabilities sit **below** Good's fixed 0.55–0.65 reject band — ranking quality and flip rate were **decoupled**. An offline sweep on the self-play distribution found a **flat 0.45 reject threshold with behavior-risk penalties OFF** achieves a **42.4%** evil-team flip rate (clearing the 40% target) at 22.9% clean-team false-reject. Penalties are off because the card model **already ingests** the rejected/failed-quest signal, so keeping them double-counts — penalties-off gives a better flip/false-reject ratio at every threshold. (An earlier counterfactual favored keeping penalties, but only under the *old* selection rule at the *old* 0.55–0.65 band; the chosen operating point is flat-0.45 **+** penalties-off.)

### 5.6 Live gates (frozen-evil design)

Phase 3 holds **Evil fixed at `factor_v2` + the default Evil policy** and swaps **only** Good's detector, so any win-rate gain is causally attributable to detection.

- **Gate-1 smoke (2026-06-19, n = 30, Windows host):** Good **40.0% (12/30)** vs 23% baseline (+17 pts, p ≈ 0.07 — trending). All 18 Evil wins were legitimate 3-fail losses; **zero hammer auto-wins** (`failed_party_votes` 0–4) → flat-0.45/penalties-off is **hammer-safe** in practice.
- **Gate-2 confirmation (2026-06-20, 64 GB Mac, 62 new games):** new run 35/62 = 56.5%. **Pooled C2 = 30 Gate-1 + 62 new = 47/92 = 51.1%** (95% CI 40.9–61.3%) vs **C1 baseline 21/90 = 23.3%**; two-proportion **z = 3.87, p = 0.0001**. Cleared the ≥ 45% target. Loss-mode: all **27** Evil wins were legitimate 3-fail; **0 hammer**. (Gate-1 ran on Windows and Gate-2 on a Mac; because game behavior is driven by the LLM backend and the frozen config — not the host — the runs pool validly.)

**Bottom line [PHASE 3]:** the recalibrated `factor_v3` detector roughly **doubles** Good's self-play win rate against frozen Evil (**23.3% → 51.1%, ≈ 2.2×**) at strong significance, while staying hammer-safe. Operationally, the Gate-2 run cost **$5.47** of DeepSeek (~$0.08/game); a concurrency-15 batch saw transient first-batch DeepSeek hangs (excluded as technical failures), and a gentler concurrency-7 top-up ran clean.

---

## 6. Current status and deferred work

**Status.** Gate-2 passed: detection rebuilt (AUC 0.598 → 0.722 self-play; 0.65–0.69 held-out human), calibrated (ECE ≤ 0.04), live Good win rate ~doubled to 51.1% (p = 0.0001), hammer-safe. Track A is the promoted detector; Track B is the tracked human-leaning challenger.

**Open / deferred:**
- **Human play is not yet re-tested.** Gate-2 is pure frozen-evil self-play; transfer of the Phase-3 detector (and the vibes-off interaction) to varied human players is the open target — and the dimension most comparable to the original paper's human study.
- **Aspirational Gate-0 targets still unmet** by the deployed config: detection AUC ≥ 0.75 and mean P(evil | evil) ≥ 0.65 by quest 3. (The 40% counterfactual-flip target *was* cleared, at 42.4%.)
- **Self-play fine-tune (`*_ft`):** corrective for the human↔self-play distribution shift; needs a nightly self-play campaign to accumulate games (must not train on the eval gate).
- **Proposer-free retrain:** confirm "drop proposer" on the self-play distribution before baking it into the deployed schema.
- **C3/C4 forced-fifth cells:** exploratory evaluation of the forced-fifth hammer variant remains optional.

---

## 7. Glossary

- **GRAIL** — the hybrid agent of arXiv 2506.17788: a structured factor-graph belief model for inference + an LLM for language/interaction + a hand-written heuristic policy for actions.
- **Factor graph** — a probabilistic graphical model of variables (here 6 role marginals + observed quest history) linked by factor nodes (neural p(role | history) factors + an exactly-2-evil constraint); beliefs are computed via belief propagation, or by exact enumeration of the 15 evil pairs in `factor_v3`.
- **Belief / detection AUC** — area-under-ROC for ranking true Evil above Good by the model's P(evil); 0.5 is chance. Old `factor_v2` ≈ 0.598; Phase-3 Track A = 0.722 (self-play). Note that AUC (ranking) and reject/flip rate (action) are **decoupled** — see §5.4–5.5.
- **Hammer rule** — the 5-rejection rule: by default (`evil_win`), five consecutive party rejections on a quest auto-end the game as an Evil win. "Hammer-safe" means a configuration produces zero such auto-wins. A `forced_fifth` variant instead forces the fifth proposal through.
- **ReCon (Recursive Contemplation)** — an LLM chain-of-thought reasoning baseline (Wang et al.), reproduced in `code/agent/recon/`, of the kind GRAIL is compared against in the paper.
- **Frozen-evil design** — the Phase-3 evaluation protocol that holds Evil fixed (`factor_v2` + Evil policy) and swaps only Good's detector, so win-rate changes are causally attributable to detection rather than to Evil-side changes.
