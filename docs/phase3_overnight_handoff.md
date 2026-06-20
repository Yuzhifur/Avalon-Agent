# Phase 3 — Overnight Live-Gate Experiment: Agent Handoff

**Audience:** an agent picking up this work with no prior session memory.
**Written:** 2026-06-19, branch `phase3-proavalon`.
**One-line state:** Gate-1 (live smoke) **PASSED**; Gate-2 (70-game confirmation)
is **running overnight** as a detached Windows Scheduled Task. Your job is to
let it finish, then render the Gate-2 verdict.

For the full offline story read `docs/phase3_results.md` (committed). For the
original design read `docs/phase3_plan.md`; for why phase 3 exists read
`docs/phase2_results.md`. This doc is the operational + live-phase brief that
ties them together and tells you exactly what to do next.

---

## 1. What phase 3 is

Phase 2 diagnosed that GRAIL-Evil wins ~77% of self-play not through clever
sabotage but because **Good's evil-detection was at chance** (factor_v2 belief
AUC ≈ 0.598, mean P(evil) on true evil exactly 0.500). Phase 3 rebuilds Good's
detector and converts the gain into Good self-play win rate against a **frozen**
GRAIL-Evil under current rules.

The new detector is a **proposal-card** model (`factor_v3`): it ingests rejected
proposals + proposer identity + per-player votes, not just final quest outcomes.
Two tracks were trained on the ProAvalon human dump:
- **Track A** = `v4_trackA_proavalon` (`ProposalSetDistribution` + exact 15-pair
  enumeration, 8.5k params). **Promoted** as primary.
- **Track B** = `seq_v1_proavalon` (`CardSeqGRU`, 205k params). Tracked
  challenger; better on human splits, kept in reserve.

## 2. Offline result (Gate 0) — PASSED

Detection is decisively rebuilt: Track A self-play AUC **0.722** (vs 0.598
baseline), held-out human 0.65–0.69, calibrated (ECE ≤ 0.04, T ≈ 1.0). Key
nuance that drove the live config:
- Ranking is strong late-game (AUC 0.70/0.81/0.82 in quests 3/4/5) but the
  **calibrated P(evil) is compressed** (~0.5–0.62 on true evil), so it rarely
  crossed Good's old v2-era 0.55–0.65 reject thresholds. Higher AUC alone did
  **not** yield more team-flips.
- Fix = **threshold recalibration**, the single most important pre-live step.
  Sweep on self-play-90 picked the operating point: **flat 0.45 reject
  thresholds + behavior-risk penalties OFF** (the card model already ingests the
  rejected/failed-quest signal the penalties re-added — a double-count).
- Two findings to remember: dropping the **proposer** feature *raises* self-play
  AUC (0.722→0.796) but slightly hurts human — it's distribution-shift, drop for
  a self-play target. Rejected-proposal history carries most of the self-play
  signal (matched ablation 0.722→0.539).

Details + the integrity/leakage audit: `phase3_results.md`.

## 3. The live experiment (what's running)

The promoted config is wired into **`code/evaluation/phase3_gate1.json`** —
note its comment: it is *both* the Gate-1 smoke cell and the Gate-2 C2 cell
(same frozen config):
- Good = `GRAIL_BELIEF_MODEL_GOOD=factor_v3:v4_trackA_proavalon`, reject
  thresholds flat 0.45, all six behavior-risk penalties 0.0.
- Evil = `GRAIL_BELIEF_MODEL_EVIL=factor_v2` (frozen), default policy.
- 6 players, 4 Servant / 2 Minion, no special roles. Hammer rule = evil_win.

### Gate 1 — smoke (30 games) — PASS ✅
Run dir: `code/evaluation/phase3_live_runs/20260619T033023Z/`
- **12/30 Good (40%)** vs the 23% phase-2 baseline = **+17 pts**; **0 crashes**.
- Pass bar was: no crashes + ≥+10 pts + ≥~10/30 Good wins → all met.
- **Loss-mode attribution:** all 18 Evil wins were legitimate 3-failed-quest
  losses; **zero hammer auto-wins** (terminal `failed_party_votes` 0–4, never 5).
  → flat-0.45 is hammer-safe in practice; the conservative 0.50/off fallback is
  **not** needed.
- Caveat: at n=30, 40% vs 23% is only p≈0.07 (trending, not yet significant).
  That is *why* Gate-2 exists.

### Gate 2 — confirmation (70 games) — RUNNING
Run dir: `code/evaluation/phase3_live_runs/20260619T174427Z/` (**40 new games**).
- Design: **C2 = 70 = the 30 Gate-1 games + 40 new games** (identical frozen
  config → valid to pool). C1 = the phase-2 90-game baseline (Good ≈ 23%), reused
  — do NOT re-run it.
- Test: two-proportion z-test at α=0.05, C2 vs C1. Aspirational target ≥45%.
- As of this writing the pooled C2 rate is tracking ~43%; if it holds at n=70 the
  test projects z≈2.7, p≈0.007 (clearly significant, though below 45%).

## 4. CRITICAL operational gotcha — read before launching anything

The **first** Gate-1 attempt (`20260618T061557Z`) deadlocked and wasted ~20h.
It was **not** a game/model bug — every game completed and recorded a winner.
The cause: the 30-game launcher was started as a **Claude-session-bound
background task**. When that session ended, the launcher process was reaped
mid-run, which killed its `docker compose up` children and **skipped
`parallel_games.run_game`'s in-process `finally: docker compose down`**. Two
already-completed games' containers (server + 6 agents each) leaked and held
both concurrency slots + ~7 GB RAM for ~20h.

**What is sound (verified):** the per-game timeout
(`AVALON_GAME_TIMEOUT`, `subprocess.run(timeout=...)`) and the `finally`
teardown work correctly — a 120s-timeout test abandoned its game and reaped all
containers to zero. They only protect **while the launcher process is alive**.

**The fix / how to launch correctly:**
- Use the wrapper **`code/evaluation/run_gate1_smoke.ps1`** (params: `-Runs
  -Concurrency -GameTimeout`). It runs a **pre-launch reaper**
  (`docker rm -f $(docker ps -aq --filter name=avalon-)` + prunes
  `.parallel_workspaces`) so a prior leak can never poison a run, then runs
  `tune_policy`, then a post-run reaper.
- Run it **detached**, not as a session-bound task. The current Gate-2 run uses a
  one-shot **Windows Scheduled Task** named **`AvalonGate2`** (created with
  `/ru <user> /it` so it reaches Docker Desktop and survives the session ending).
- **Concurrency 3 is RAM-safe** on this 16 GB host (~9.6 / 12 GiB; `.wslconfig`
  caps WSL at 12 GB). Concurrency 4 would risk OOM (exit 137).

## 5. Exactly what to do next

1. **Wait for Gate-2 to finish.** It's detached, so it survives session/agent
   restarts. Check progress without disturbing it:
   ```bash
   RUN=code/evaluation/phase3_live_runs/20260619T174427Z
   find "$RUN" -name status.json | wc -l                 # of 40
   ls "$RUN/summary.json"                                  # exists => run done
   docker ps --format '{{.Names}}' | grep -c server-1      # live games (0 => done)
   powershell -NoProfile -Command "(Get-ScheduledTask -TaskName AvalonGate2).State"
   ```
   Per-game winners live in `<run>/c2_promoted_good/run_NNN/status.json`
   (`winner` field) and the server JSON under `.../server/*.json`
   (`logs[].full.winner`, `quest_results`, `failed_party_votes`).
2. **When done, pool C2 = 30 + 40 = 70** and compute the **two-proportion
   z-test** vs C1 (phase-2 90 games, ~23% / ~21 Good wins). Report z, p, and the
   pooled Good win rate with a CI.
3. **Loss-mode attribution on the 70 C2 games** (reuse the Gate-1 method): for
   each game classify Good-win (3 successes) vs Evil-3-failed-quests vs
   Evil-hammer (`failed_party_votes == 5`). Confirms the hammer-safety finding at
   larger n.
4. **Write the Gate-2 verdict** into `phase3_results.md` (the live-phase section).
   Consider an independent verification workflow to adversarially confirm the
   significance test (Gate-0's verdict was rendered that way).
5. **Then:** the C3/C4 forced-fifth exploratory cells are optional afterward.

### Still genuinely deferred (need data/schema change, not just time)
- **Self-play fine-tune** (`*_ft`): corrective for the human↔self-play
  distribution shift. Needs the nightly self-play campaign to accumulate games.
  Must NOT train on the 79-game self-play-90 eval gate.
- **Proposer-free retrain:** confirm "drop proposer" on the self-play
  distribution before baking it into the deployed schema.

## 6. Environment & gotchas (this Windows machine)

- **Python venv:** `.venv\Scripts\python.exe` (Windows). It has pandas, openai,
  requests, shortuuid, torch.
- **Run command pattern** (from `code/`):
  `AVALON_GAME_TIMEOUT=5400 PYTHONPATH=<repo>/code/evaluation
  PYTHONIOENCODING=utf-8 python -m evaluation.tune_policy --grid
  evaluation/phase3_gate1.json --runs N --concurrency 3 --output
  evaluation/phase3_live_runs` — but prefer the `run_gate1_smoke.ps1` wrapper.
- **Encoding:** this machine's default is **GBK** — ALWAYS pass
  `encoding="utf-8"` when reading/writing files in Python, and `-Encoding utf8`
  in PowerShell `Out-File`.
- **.env:** the stack reads **`code/.env`**, not repo-root `.env`. All six roles
  must be automated (`ours`/non-`human`) or `validate_automated_roles` fails.
- **LLM backend:** agents run on DeepSeek (`deepseek-v4-flash`). DeepSeek
  occasionally returns HTTP-200 with empty content; the belief-update
  `json.loads` was guarded (returns neutral 0.5 priors) — if you see games
  hanging, check `agent_acl.py` json.loads sites and the "falling back to neutral
  priors" log frequency.
- **Reap orphans manually** if a run ever leaks:
  `docker rm -f $(docker ps -aq --filter name=avalon-)`.
- **Docker:** Docker Desktop must be running; compose v2 (`docker compose`),
  headless stack = `docker-compose.yml` + `docker-compose.headless.yml`.

## 7. Key files

| path | what |
|---|---|
| `code/evaluation/phase3_gate1.json` | the promoted C2 config (Gate-1 & Gate-2 cell) |
| `code/evaluation/run_gate1_smoke.ps1` | hardened detached launcher (pre/post reaper) |
| `code/evaluation/tune_policy.py` | grid runner → `candidate/run_NNN` + summary.json |
| `code/evaluation/parallel_games.py` | per-game Docker orchestration (timeout + teardown) |
| `code/evaluation/phase3_live_runs/20260619T033023Z/` | Gate-1 (30 games, 40% PASS) |
| `code/evaluation/phase3_live_runs/20260619T174427Z/` | Gate-2 (40 new games, running) |
| `docs/phase3_results.md` | offline + live results (authoritative) |
| `docs/phase3_plan.md` | original phase-3 design |
