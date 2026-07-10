# Concluding Experiments — v3-vs-v3 Co-adaptation & the Language Layer

*This document closes out the project's experimental arc. It executes the two
highest-priority next steps from [limitations_and_next_steps.md](limitations_and_next_steps.md)
(§7 items 1 and 3): the **v3-vs-v3 self-play** probe of co-adaptation /
equilibrium strength, and the **language-diversity + fingerprint** experiment
(§5's "least risky first move"). Design frozen and runs launched 2026-07-09.*

## Why these two, and what they conclude

Gate-2's headline (Good 23.3% → **51.1%**, p = 0.0001) is conditional on two
artifacts of the evaluation environment:

1. **Evil was frozen** on `factor_v2` + the default `EvilPolicy`. The detector
   exploits proposal/rejection dynamics produced by *that* opponent.
2. **Chat is templated.** All six agents speak through one fixed prompt
   template; the observed fingerprints are easy to spot, and agents almost
   never bluff, hedge, joke, or contradict themselves.

These experiments attack exactly those two conditionals. They are designed as
*probes with frozen references*, not new gates: every cell keeps the promoted
Good **byte-identical to Gate-2's C2** (factor_v3 Track A, flat-0.45 reject
thresholds, behavior-risk penalties off, LLM-vibes ON — the deployed live
config), so each cell's Good win rate reads directly against
**C2 = 47/92 = 51.1%** and **C1 = 21/90 = 23.3%**.

## Experiment A — v3-vs-v3 self-play (co-adaptation)

**What Evil's belief model actually does.** GRAIL-evil's "beliefs" are its
mirror of *how the table perceives each player* (see the Evil prompt intro and
`EvilSocialState`). They drive three behaviors: proposing the least-exposed
evil player with credible cover, distancing from an exposed teammate, and the
public cover plan fed into the speech layer. Under Gate-2, Evil computed that
mirror with `factor_v2` — a model of a *v2* observer — while the actual Good
observer had upgraded to `factor_v3`. Evil was blending in against a detector
that no longer existed.

| cell | Good | Evil beliefs | Evil policy | question |
|---|---|---|---|---|
| C2 (reference) | promoted (v3) | factor_v2 | default | Gate-2 result: 51.1% |
| **a1_v3_evil** | promoted (v3) | **factor_v3** (same Track A checkpoint) | default | Does an accurate mirror claw back Evil's losses? |
| **a2_v3_evil_quiet** | promoted (v3) | **factor_v3** | **signal-starving** | Can Evil counter-adapt by *starving the detector's food*? |

**The a2 policy rationale.** Phase 3's ablation showed the detection signal
lives substantially in **rejected-proposal dynamics** (removing them: self-play
AUC 0.722 → 0.539). The default Evil *generates* that signal: it rejects clean
teams through quest 3, cover-approves only 25% of the time, and evil votes
often move as a bloc. `a2` retunes exactly those knobs
(`clean_team_cover_approve_rate` 0.25 → 0.75, `clean_team_reject_through_quest`
3 → 1, `evil_vote_split_rate` 0.25 → 0.5, `teammate_distance_rate` 0.35 → 0.5):
a "quiet" Evil that approves clean teams early, splits its bloc, and distances
more. The tension is real — quiet Evil feeds the detector less but also blocks
fewer Good quests — and that tension *is* the co-adaptation question. (Phase 2
found Evil blend-in knobs inert, but that was against a chance-level detector
where there was no signal worth starving; against factor_v3 the knobs finally
have a mechanism to matter.)

Note the a1 swap changes **only** `GRAIL_BELIEF_MODEL_EVIL`; Evil keeps the
same ego-conditioning convention (`self_role=EVIL` restricts the pair
enumeration through itself) that it had under v2, so the cell isolates the
model upgrade, not an interface change.

## Experiment B — language diversity + fingerprint

Frozen Evil (factor_v2 + default policy, exactly C2's opponent) in both cells,
so any win-rate movement is attributable to the language layer:

| cell | speech layer | vibes | question |
|---|---|---|---|
| C2 (reference) | default template | on | 51.1% |
| **b1_diversified_chat** | **persona bank** (`GRAIL_PROMPT_PERSONA_BANK=1`) | on | Does the gain survive a diversified voice? Does the fingerprint drop? |
| **b2_no_chat** | **off** (`GRAIL_NO_CHAT=1`) | off (nothing to read) | How much of the result depends on language at all? |

**Persona bank mechanics** (`code/agent/our/prompts.py`): six role-symmetric
voices — `casual` (≈ the old voice plus an explicit license to hedge/joke/push
weak reads), `terse`, `analytical`, `hedger`, `blunt`, `playful`. Each persona
replaces **only** the `output_style` paragraph; the strategic wording around it
(including Evil's cover/vote-alignment instructions) is byte-identical to the
default template, so the intervention changes the *voice*, not the strategy
content. Assignment is deterministic per game (seeded by game id): every agent
computes the same mapping locally and the six players get six **distinct**
personas — persistent for the whole game ("six distinct speakers instead of
six copies"). Assignments are logged as `PROMPT_PERSONA <key>` lines in each
agent's LOG file. Off by default; the default `PromptHint` path is untouched
(`code/agent/test_prompt_personas.py` locks this in).

**No-chat mechanics** (`code/agent/agent_acl.py`): with `GRAIL_NO_CHAT=1`
agents skip proposal-justification messages and pass the turn instead of
chatting (the server rejects empty messages, so none are sent). Mechanical
play — beliefs, proposals, votes — is unchanged. Vibes are disabled in this
cell since there is no chat to read; b2 therefore measures the *joint*
contribution of chat + vibes on top of the structured detector.

**Fingerprint evaluation** (`code/evaluation/fingerprint_classifier.py`,
stdlib-only): multinomial naive Bayes over word 1-2 grams + char 3-4 grams,
**game-grouped** 5-fold CV, roster names masked so identity mentions can't
stand in for style. Tasks: role (good-vs-evil) per message and per
player-game, persona identification within b1 (here *high* accuracy is the
goal — it verifies the personas are actually distinct), and a
baseline-vs-diversified condition classifier. Style statistics (length,
hedging, questions, lexical diversity) complete §6's layer-1 evaluation.

### Baseline fingerprint result (already measurable — Gate-2 chat, 62 games)

The templated baseline is exactly as bad as the limitations doc feared:

| task (name-masked, game-grouped 5-fold CV) | accuracy | majority | AUC |
|---|---|---|---|
| role per message (n = 13,063) | 0.749 | 0.707 | **0.769** |
| **role per player-game** (n = 372) | **0.997** | 0.667 | **1.000** |

Pooling one player's messages over one game identifies the evil players
**essentially perfectly** — a bag-of-words model with no game-mechanical
input at all. The per-message signal is moderate (AUC 0.77), but ~35
messages/player/game aggregate it into a beacon. Masking player names barely
moves it (unmasked: 0.997/1.000), so this is style/content, not who-mentions-
whom. For calibration: the *structured* detector that doubled Good's win rate
reaches AUC 0.722 from mechanical history — the speech layer leaks **more
role information than the entire proposal/vote/quest record**, to anyone
listening. GRAIL's own agents do not exploit chat this way (the card model is
chat-blind, and vibes is a weak ±0.1–0.25 prior nudge), which is why the
fingerprint coexisted with a 77%-Evil phase 2; but any opponent that *reads*
— a human, or an LLM baseline like ReCon — gets the roles for free.
Style stats for the baseline: 210.7 messages/game, 10.7 words/message (p90 =
18), hedge rate 0.128, question rate 0.062.

## Cells, references, and analysis

Grid: [`code/evaluation/phase3_concluding.json`](../code/evaluation/phase3_concluding.json)
— 45 games/cell, concurrency 12, 90-min per-game timeout, launched detached
via `daemonize.py` + `run_gate2.sh` (the hardened Gate-2 launcher: pre/post
`avalon-*` reaper, foreground tune_policy so per-game teardown always runs,
`.done` sentinel). Decision logs are archived to JSONL automatically by
`tune_policy` (the phase-3 log-retention rule). Smoke = 1 game/cell first.

**Smoke outcome (2026-07-09).** The smoke caught a real deadlock: with
`GRAIL_NO_CHAT` nothing is ever said, so "message" never entered the per-turn
action history and `agent_base`'s suggestion chain re-suggested it forever —
`start_party_vote` was unreachable and the game cycled turns to the timeout.
Fixed by having the no-chat branch fall through to the ripe party vote (same
`this_leaders_turn` pacing as the chat flow). The re-smoked no-chat game
completed in **5.0 minutes** with 0 player messages and a full quest structure
(vs ~45 min for chat games). Persona assignment (all six voices, distinct,
logged), Evil-side `factor_v3` loading, and the a2 policy-override env were
all verified live in the other three smoke games.

```
# win rates + z-tests vs C1/C2 + loss modes (+ a2-vs-a1 contrast)
cd code/evaluation && python3 concluding_analysis.py phase3_concluding_runs/<ts> --json concluding_verdict.json

# fingerprint: baseline (Gate-2 roots) vs diversified (b1 cell)
python3 fingerprint_classifier.py \
    --baseline phase3_live_runs/20260620T032507Z phase3_live_runs/20260620T042241Z phase3_live_runs/20260620T074951Z \
    --diversified phase3_concluding_runs/<ts> --cell b1_diversified_chat --json fingerprint_report.json
```

Power note: n = 45/cell resolves *large* effects against C2 (a drop to ~33%
reaches z ≈ 2 at α = 0.05); it brackets rather than pins mid-size effects
(95% CI ≈ ±14 pts). Cells pool with any future identically-configured runs,
exactly as Gate-1/Gate-2 pooled.

## Results

**a1/b1/a2 IN FLIGHT — remaining rows are filled by `concluding_analysis.py`
/ `fingerprint_classifier.py` output as cells complete.**

| cell | Good wins | rate | 95% CI | vs C2 (51.1%) | vs C1 (23.3%) | hammer |
|---|---|---|---|---|---|---|
| **a1_v3_evil** (done 2026-07-10) | **23/46** | **50.0%** | [35.6%, 64.4%] | z = +0.12, p = 0.90 (ns) | z = −3.14, **p = 0.0017** | **0** |
| a2_v3_evil_quiet | — | — | — | — | — | — |
| b1_diversified_chat | — | — | — | — | — | — |
| **b2_no_chat** (done 2026-07-09) | **19/43** | **44.2%** | [29.3%, 59.0%] | z = +0.75, p = 0.45 (ns) | z = −2.45, **p = 0.014** | **0** |

**a1 verdict (the headline cell).** Upgrading Evil's belief model to the very
same `factor_v3` checkpoint Good uses moves Good's win rate from 51.1% to
**50.0%** — a statistically invisible change (p = 0.90) at n = 46, with the
edge over the phase-2 baseline fully intact (p = 0.0017) and zero hammers.
Per the pre-registered grid: **Good's rebuilt detector is robust to Evil's
model upgrade.** An accurate mirror of the new observer does not help Evil,
because the tells factor_v3 reads are generated by Evil's *policy* (its
clean-team rejects, bloc votes, and proposal habits), not by a
miscalibrated belief mirror — better self-knowledge cannot hide behavior the
policy keeps producing. The remaining co-adaptation question is therefore
squarely about the policy, which is what a2 tests.

**b2 verdict (first completed cell).** With the speech layer off entirely (no
messages, no vibes), the promoted Good still wins **44.2%** — statistically
indistinguishable from the with-chat C2 (51.1%) and still roughly **double**
the phase-2 baseline (significantly above C1). Zero hammer auto-wins even
with zero discussion. Per the pre-registered grid: **the Gate-2 win-rate
result is essentially structural** — factor_v3 + recalibrated thresholds do
the work; templated chat + vibes contribute at most a modest additive bump
(point estimate ~7 pts, CI too wide to pin) to self-play outcomes. This also
sharpens the fingerprint concern's practical reading: the speech layer leaks
role information (baseline AUC 1.000) while adding little self-play win rate
— high blind-side risk, low measured reward, in this ecology. (Ops: 42/45
full-run games + the re-smoke game scored; 3 first-batch games were excluded
as technical failures — a startup **registration race** when 12 compose-ups
fire at once, agents never initialized, no game ever started — plus the one
pre-fix deadlocked smoke game. No-chat games run ~4–6 min.)

| fingerprint task | baseline (templated) | diversified (b1) |
|---|---|---|
| role per player-game acc / AUC | 0.997 / 1.000 | — |
| role per message acc / AUC | 0.749 / 0.769 | — |
| persona id acc (want high) | n/a | — |

## Interpretation grid (pre-registered, so hindsight can't move the goalposts)

- **a1 ≈ C2** → Good's edge is robust to Evil's model upgrade: the detector
  reads *behavioral* tells the default policy can't stop producing.
- **a1 well below C2** (toward C1) → the Gate-2 number was substantially a
  stale-mirror artifact; v3-vs-v3 equilibrium is the real headline and further
  Good work must assume an adapted Evil.
- **a2 < a1** → policy counter-adaptation (signal starving) matters beyond
  model upgrading; the next Good iteration must be trained against varied Evil
  policies, and the phase-2 "evil knobs are inert" memory is scoped to
  chance-level detectors.
- **a2 ≥ a1** → quieting starves Evil's own win pressure as much as the
  detector; the default policy was already near its exploitability frontier.
- **b1 ≈ C2 with role-AUC dropping** → clean upgrade: fingerprint reduced at
  no strategic cost; adopt the bank as the default speech layer.
- **b1 ≈ C2 with role-AUC ≈ 1 persisting** → the fingerprint lives in *what*
  agents argue (template-shaped reasoning), not the surface voice; the next
  lever is the speech-act planner / role-symmetric content, not style.
- **b1 well below C2** → the old template was doing coordination work (its
  predictability helped Good coordinate); language and detection gains are
  entangled and must be co-designed.
- **b2 ≈ C2** → the win-rate result is essentially structural; chat+vibes are
  decorative for GRAIL-vs-GRAIL outcomes and the fingerprint concern is about
  *transfer*, not about the Gate-2 number.
- **b2 ≠ C2** → chat/vibes carry real win-rate weight in self-play; every
  language-layer change must be win-rate-gated, and Gate-2's number is partly
  a language-environment number.

## What this concludes (and what it leaves open)

With these four cells the project's experimental arc closes as: *diagnose*
(phase 2: Evil ~77%, Good detection at chance) → *rebuild* (phase 3: proposal-
card detector, AUC 0.598 → 0.722, thresholds recalibrated) → *confirm live*
(Gate-2: 23.3% → 51.1%, hammer-safe) → ***stress the two conditionals*** (this
document: opponent co-adaptation, language environment) → and quantify the
speech-layer fingerprint that self-play had been silently tolerating (baseline
role AUC 1.000 per player-game).

Still open beyond this (unchanged from the limitations doc, in priority
order): human / mixed-population transfer (the original paper's claim), the
self-play fine-tune (`*_ft`), the proposer-free retrain, C3/C4 forced-fifth
cells, richer speech-act planning if b1 shows the fingerprint lives in
content, and LLM/human blind panels to complement the classifier (the §6
layer-1 metric we could not run without human raters).
