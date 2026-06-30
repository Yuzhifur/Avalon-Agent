# Current Limitations & Next Steps

*Companion to [project_overview.md](project_overview.md). Phase 3 produced a strong, statistically significant result — but a specific one. This document is the honest scope statement: what the result does and does not establish, the open scientific and engineering concerns, and where the work goes next. Last updated 2026-06-20.*

## What Phase 3 actually proves (and what it doesn't)

Gate-2 proves one thing well: **Good's rebuilt detector (`factor_v3` Track A + recalibrated flat-0.45 thresholds) beats the *old, frozen* Evil (`factor_v2` + the default `EvilPolicy`) far more often than the previous Good** — its self-play win rate roughly doubles, 23.3% → 51.1% (p = 0.0001), with zero hammer auto-wins.

It does **not** prove that "GRAIL v3 is solved." It is a **frozen-Evil, Good-detector** result measured in **GRAIL-vs-GRAIL self-play with templated chat**. The sections below enumerate the gaps between that and a fully-validated, human-transferable, equilibrium-strength agent.

---

## 1. Opponent and co-adaptation concerns

- **Evil was frozen.** Gate-2 only swaps Good's detector; Evil keeps `factor_v2` beliefs and the default policy. We do **not** know what happens when Evil also uses `factor_v3` / `seq_v1`, new thresholds, or a retrained policy. The headline number is conditional on a fixed, older opponent.
- **No co-adaptation / equilibrium test.** Once Evil changes, Good's learned tells can change too. The Phase-3 detector exploits proposal/rejection dynamics produced by the *current* heuristic Evil; a stronger Evil could produce different rejection patterns and erase part of the signal. There is no v3-vs-v3 self-play equilibrium result yet.
- **Thresholds may be tuned to the current opponent.** Flat 0.45 + penalties-off was selected against frozen Evil. If Evil behavior changes — or if humans are noisier — that operating point could cause too many clean-team false rejects or miss different Evil patterns.
- **Behavior-risk penalties are not globally settled.** Penalties were turned off because the card model already consumes rejected/failed-quest information (avoiding a double-count). That is right for Track A in this setup, but the interaction may differ for Track B, human games, or new Evil policies.

## 2. Transfer and distribution-shift concerns

- **Human play has not been re-tested.** The original GRAIL claim (and its strongest result — first agent to beat humans, 67%) is about *humans*. Gate-2 is pure self-play. Whether the new detector transfers to human or mixed-population play is the central open question.
- **Human-vs-self-play distribution shift persists.** Track A is better on frozen self-play; Track B is slightly better on held-out human data. The best model already depends on the target environment, which is exactly the shift Phase 3 has not closed. Track B remains only a tracked **reserve/challenger** (Track A was promoted purely on self-play AUC); if the evaluation target shifts to humans, Track B may warrant **renewed live evaluation** rather than defaulting to Track A.
- **The proposer feature is unresolved.** Dropping the proposer feature *raises* self-play AUC (0.722 → 0.796) but slightly *hurts* human AUC and, crucially, *flips fewer* evil teams. "Remove proposer" is therefore a deployment choice, not a universal improvement; the proposer-free retrain still needs confirmation on the self-play distribution before being baked into the schema.
- **ProAvalon has no chat.** The detector is trained on structured proposal/vote/quest data only, not language. That fits the belief-model design, but it leaves a gap between structured detection and full social deduction with natural-language persuasion.
- **Training labels collapse role nuance into alliance.** The detector predicts Good/Evil alliance, not special-role structure. Fine for simplified 6-player self-play; human Avalon role variants may introduce behavior the alliance-only labels do not capture cleanly.
- **Offline replay uses historical proposal distributions.** Counterfactual flip metrics replay old decisions on old proposed teams; live play changes the proposals too. Useful as a screen (it correctly predicted Gate-2's direction), but offline metrics cannot fully predict the new live equilibrium.

## 3. Detection quality vs. actual play

- **Aspirational detection targets remain unmet.** AUC rose 0.598 → 0.722, but the planned aspirational target was **≥ 0.75**. Mean P(evil | true evil) is still compressed around **0.5–0.62**, below the "confident detection" level (target ≥ 0.65 by quest 3).
- **AUC still does not guarantee better play.** Phase 3's load-bearing finding is that ranking quality and vote flips are **decoupled** (the no-proposer model has higher AUC but flips fewer evil teams). Threshold recalibration fixed this for the *tested* setting; the same threshold may not be optimal against humans, v3 Evil, or future policies.
- **Loss attribution after the new live run is incomplete.** Gate-2 confirmed the remaining Evil wins were legitimate 3-fail losses (not hammers), but a deeper Phase-2-style bucketing (misidentification vs. forced approvals vs. bad proposals vs. threshold) of the new losses would sharpen where Good still fails.

## 4. The language template / fingerprinting concern

The LLM speech layer uses **fixed prompt templates** ([`code/agent/our/prompts.py`](../code/agent/our/prompts.py)). A single shared `output_style` instructs every agent to "respond… as if you are a college student, typing informally… do not use emojis… shorter messages are more effective." Evil has structurally-mirrored templates that add explicit instructions to "maintain cover," "subtly support" evil-containing teams "without being obvious," "redirect suspicion plausibly," and make the message "compatible with [the] planned vote" (`{planned_vote}` is passed into the prompt). Observed in practice: **template fingerprints are easy to spot, and agents almost never bluff, hedge, joke, contradict themselves, or push weak arguments.**

Why this matters as a scientific concern:

- **Template fingerprinting.** Uniform rhythm, length, and argument style make the agents stylistically predictable — itself a strategic weakness in Avalon, and a tell a human or classifier can learn instead of true game reasoning.
- **Good and Evil may have different linguistic fingerprints.** The Evil template's cover/redirect/vote-alignment wording can shape output style, so a detector or human might learn *Evil's speech pattern* rather than its play.
- **Self-play may overfit to a templated ecology.** All-GRAIL self-play tests against one particular templated social environment, not broad social behavior. Chat still affects votes through the LLM "vibes" prior, persuasion, and downstream behavior even though the card model ignores chat directly.
- **Over-consistency / over-honesty.** Good is told to explain beliefs and uncertainty; Evil to look plausible. Real players bluff, hedge, joke, and push weak arguments. Over-consistent agents create unnatural proposal/vote dynamics.
- **Reduced strategic diversity.** Scripted reactions narrow the behavior distribution; some Phase-3 gain may reflect beating that narrow distribution.
- **Policy-intent leakage.** Passing the planned vote into the prompt and asking the message to be "compatible" can yield overly synchronized speech/action pairs ("I object…" → reject) that humans and detectors can exploit.
- **Weakened Evil deception.** "Include evil if possible but don't be obvious" produces a limited deception strategy; strong Evil should sometimes bus a teammate, approve clean teams, or make messy human-like arguments.
- **Human-transfer uncertainty + attribution confound.** Humans don't follow this template, so self-play strength may not transfer. And it muddies attribution: is the Good gain better structured detection, or predictable templated Evil proposal/rejection behavior? Phase 3 attributes the gain to detection, but the social template is part of the environment.
- **Vibes interaction is unsettled.** If chat is templated, the vibes prior may react to template artifacts. Recorded vibes hurt the old detector; the interaction with the new detector is not fully characterized.

**Net:** Phase 3 has not cleanly separated *"better Avalon reasoning"* from *"better performance inside a templated, homogeneous bot-speaking environment."*

---

## 5. Solving the language layer — goal, options, difficulty

**The goal is not "make agents quirky for realism."** It is: *reduce exploitable language fingerprints while preserving strategic competence, private-information safety, and measurable transfer to humans / mixed opponents.* In Avalon, language is part of the policy; if it is template-shaped, self-play may reward beating a narrow scripted ecology rather than learning robust social deduction. Bluffing, hedging, joking, and contradiction are **tools, not goals** — they count as success only if they reduce exploitability and preserve or improve strategic outcomes.

| Solution | Difficulty | Benefit | Risk |
|---|---|---|---|
| **Prompt-variant bank** (terse / cautious / confident / casual / analytical / messy; randomize per agent/game) | Low | Quick reduction of obvious fingerprinting | Surface-level; underlying reasoning unchanged |
| **Persistent personas** (a stable style per agent for the whole game) | Low–Med | Six distinct speakers instead of six copies | A rigid persona is itself a fingerprint |
| **Role-symmetric public prompts** (public speech nearly identical across roles; private strategy handled separately) | Med | Removes Evil-specific language artifacts | Must keep Evil using private info strategically |
| **Speech-act planner** (choose an act — support/oppose/hedge/question/accuse/defend/deflect/joke/concede/bus — then generate) | Med | Controlled diversity matched to game state | Needs tuning so acts fit the state |
| **Explicit Evil deception policy** (sometimes bus, approve clean teams, defend Good, make weak human-like reads, stay quiet) | High | Directly fixes "Evil too pattern-like" | Hard to evaluate; bad deception can weaken Evil |
| **Human-log style imitation / style adapter** (sample or train on human chat for natural hedging, contradiction, jokes) | Med–High | Best path to human-like transfer | ProAvalon has no chat; chat datasets small/noisy |
| **Adversarial fingerprint detector** (classifier guesses agent/role/template; iterate to chance) | Med | Directly measures remaining fingerprints | Passing a detector ≠ playing Avalon well |
| **Mixed-population self-play** (mix prompt variants, old/new agents, no-chat, chaotic, replay agents) | Med–High | Tests robustness across social environments | Larger experiment matrix |
| **No-chat / limited-chat ablations** (chat off / templated / diversified / human-like) | Med | Isolates how much win rate depends on language artifacts | No-chat isn't full Avalon |

**Recommended first step (least risky):** prompt/persona variants **+** a fingerprint classifier or human blind check. If the fingerprint drops and win rate holds → a clean upgrade. If the fingerprint drops and win rate collapses → the old template was doing coordination work, which is itself an important finding.

## 6. How to evaluate a richer language layer

Because "can it bluff?" is not a goal in itself, evaluation should be layered:

1. **Fingerprint evaluation.** Can humans or classifiers identify agent / role / template from messages? Metrics: agent-vs-human classifier accuracy, role-classifier accuracy from public messages, template-classifier accuracy, human blind ratings ("bot-like?", "same speaker?", "scripted?"). **Target: role/template detection approaches chance.**
2. **Strategic-language evaluation.** Does the language help or hurt play? Metrics: vote-persuasion rate (do others change approve/reject after a message?), self-contradiction rate, private-info leakage rate, message/action mismatch rate; for Evil — successful cover, teammate distancing, clean-team sabotage; for Good — accurate suspicion communication, appropriate uncertainty.
3. **Game-outcome evaluation.** Run cells: old-vs-old template, diversified-vs-old, diversified-Good-only, diversified-Evil-only, diversified-both, no-chat baseline, human-like variants. Metrics: Good/Evil win rate, hammer losses, quest patterns, misidentification share in losses, **whether `factor_v3` gains survive when language changes.**
4. **Transfer evaluation.** Performance against varied prompt populations and against humans; the human-AUC vs self-play-AUC gap; whether the Track A vs Track B preference flips when the target changes.

---

## 7. Next steps (prioritized)

1. **v3-vs-v3 self-play** — replace frozen Evil with `factor_v3` (and tune Evil thresholds/policy) to probe co-adaptation and equilibrium strength.
2. **Human / mixed-population evaluation** — the key transfer test the original paper's claim rests on; pair with the language-diversity work so chat is not a confound.
3. **Language-diversity + fingerprint experiment** — the low-risk first move from §5–6: persona/variant bank + a fingerprint classifier / human blind check, with no-chat and diversified-chat ablation cells.
4. **Self-play fine-tune (`*_ft`)** — corrective for the human↔self-play shift; needs a nightly self-play campaign to accumulate games and must not train on the eval gate.
5. **Proposer-free retrain** — confirm "drop proposer" on the self-play distribution before changing the deployed schema.
6. **C3/C4 forced-fifth cells** — quantify the structural effect of the hammer-rule variant (currently exploratory/deferred).
7. **Deeper loss attribution** on the new live losses (Phase-2-style buckets), and **orchestration/log-retention hardening** so larger campaigns don't drop games or lose decision logs (a robustness matter, not a scientific result, but it guards against hidden bias).

---

### One-line summary

Phase 3 convincingly shows Good's detector upgrade works **against frozen old Evil in templated self-play**; it does not yet establish v3-vs-v3 equilibrium strength, human transfer, optimal model/threshold choices under changed opponents, or that the gains survive a non-templated language layer.
