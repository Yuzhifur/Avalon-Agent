# Bayesian Avalon
This repository contains the code and data for the "Bayesian Social Deduction with Graph-Informed Language Models".

[Project Page](https://camp-lab-purdue.github.io/bayesian-social-deduction/) | [Arxiv](https://arxiv.org/abs/2506.17788) | [Dataset](https://huggingface.co/datasets/shahabrahimirad/bayesian-social-deduction) 

# TLDR;
We developed GRAIL, an agent that uses ptobabilistic graph models to reason about beliefs in the social deduction game of Avalon. The GRAIL agent is able to match the performance of the biggest LRMs with smaller models. 

# Extending GRAIL (this fork): rebuilt Good-side detection

This repository **extends** the original GRAIL research described above. The full overview is in **[docs/project_overview.md](docs/project_overview.md)**.

### What this builds on

The game engine, the factor-graph belief model, the heuristic action policy, the LLM interaction layer, and the **ReCon** reasoning baseline (Recursive Contemplation, Wang et al.; `code/agent/recon/`) are all inherited from the original paper — *Bayesian Social Deduction with Graph-Informed Language Models* ([arXiv 2506.17788](https://arxiv.org/abs/2506.17788)), whose headline result is that GRAIL is the **first language agent to beat humans** in a controlled study (**67% win rate**, rated above reasoning baselines and human teammates). Training is grounded in human game-log corpora (avalonlogs, ProAvalon).

### What this extension adds (Phases 2–3)

The original evaluation pitched GRAIL against humans and larger models. This fork studies GRAIL in **GRAIL-vs-GRAIL self-play with Evil held frozen**, to isolate and improve Good's *evil-detection*:

- **Phase 2 — diagnosis.** All-GRAIL self-play showed Evil winning **~77%**, but not from clever sabotage: Good's evil-detection was **at chance** (true evil rated ~0.5 P(evil); detection AUC ≈ 0.598). Good's baseline self-play win rate was **~23%**.
- **Phase 3 — rebuild.** A new **proposal-card belief model (`factor_v3`)** ingests the *rejected*-proposal history + proposer identity + per-player votes (not just final quest outcomes), trained on the ProAvalon human dump. Offline detection AUC rose **0.598 → 0.722**; the key unlock was **threshold recalibration** (flat 0.45 reject thresholds, behavior-risk penalties off). A frozen-Evil live confirmation (**Gate-2**) raised Good's self-play win rate from **23.3% → 51.1%** (47/92 pooled, *p* = 0.0001), with zero hammer auto-wins.

Details: **[docs/project_overview.md](docs/project_overview.md)** (overview) and **[docs/phase3_results.md](docs/phase3_results.md)** (full results).

# Current limitations & next steps

The Phase-3 result is strong but **specific**: it is a *frozen-Evil, Good-detector* result measured in *templated self-play*. It does **not** yet establish full v3-vs-v3 equilibrium strength, human transfer, or robustness to a non-templated language layer. Key open concerns:

- **Evil was frozen** — no v3-vs-v3, co-adaptation, or equilibrium test; the recalibrated thresholds may be tuned to the current opponent.
- **Human play has not been re-tested** — the original claim is about humans; a human↔self-play distribution shift remains (Track A is best on self-play, Track B on held-out human data).
- **Detection targets are partly unmet** — AUC 0.722 is below the ≥ 0.75 aspirational target, and confidence on true evil is still compressed (~0.5–0.62). AUC and actual play are **decoupled** (higher-AUC ablations can flip *fewer* evil teams).
- **The LLM speech layer uses fixed prompt templates** (`code/agent/our/prompts.py`) with **easily-spotted fingerprints**; agents almost never bluff, hedge, joke, or contradict themselves — which may inflate self-play gains and threatens transfer to humans.
- **Deferred work:** self-play fine-tune (`*_ft`), proposer-free retrain, the C3/C4 forced-fifth hammer variant, and orchestration/log-retention hardening.

**Likely next steps:** v3-vs-v3 self-play → human / mixed-population evaluation → a language-diversity + fingerprint-classifier experiment → self-play fine-tune. Full discussion, including the language-fingerprinting analysis, a menu of solutions with difficulty estimates, and a four-layer evaluation framework: **[docs/limitations_and_next_steps.md](docs/limitations_and_next_steps.md)**.

**Concluding experiments (completed 2026-07-10).** The two highest-priority items above were executed as the project's closing experiments — full design, pre-registered interpretations, and results in **[docs/concluding_experiments.md](docs/concluding_experiments.md)** (170 scored games): (1) **v3-vs-v3 self-play** — upgrading frozen Evil to the same `factor_v3` detector Good uses changes nothing (**50.0%** Good, p = 0.90 vs C2's 51.1%), and neither does retuning Evil's policy to starve the rejected-proposal signal (**51.4%**, p = 0.98): the rejection dynamics the detector reads are load-bearing for Evil's own win condition, so the Gate-2 result is *not* a frozen-opponent artifact. (2) The **language-diversity + fingerprint experiment** — templated chat identifies the evil players at **99.7% accuracy (AUC 1.000)** per player-game (more role leakage than the entire mechanical record, structured-detector AUC 0.722); a working six-persona voice bank leaves the win rate intact (**54.5%**) but barely dents that fingerprint (AUC 0.997) — it is **content-borne**, not surface-style — and removing chat entirely still yields **44.2%** (ns vs C2), so the detector gain is structural. Every cell: zero hammer auto-wins. Net: the detector rebuild is robust in-ecology; the speech layer is a *transfer* liability (any listening opponent gets the roles for free) whose fix must target content, not phrasing.

## Code

The `code/` directory contains the game engine and agent implementations.

The subdirectory includes: core game logic for social deduction mechanics, pre-trained GRAIL agents and baseline reasoning agents, and scripts to simulate games or run human-agent interactions.

See `code/README.md` for full documentation on code structure, agent training, and game configuration [here](code/README.md).

For a beginner-friendly explanation of GRAIL's factor graph, neural factor functions, action heuristic, and training data, see [docs/grail_beginner_guide.md](docs/grail_beginner_guide.md).

For a high-level overview of the whole project — the original GRAIL contribution plus the Phase 2–3 work that rebuilt Good's evil-detection (lifting its self-play win rate from ~23% to 51%) — see [docs/project_overview.md](docs/project_overview.md).

## Human Experiment Data

The `data/` directory contains logs from human-vs-agent experiments. These are the same games from the [human experiment dataset](https://huggingface.co/datasets/shahabrahimirad/bayesian-social-deduction/tree/main/human_experiments), but they have been specifically formatted to be replayable within the Avalon Game Client.

There are 15 experiment folders `02` to `16`, each containing two JSON files &mdash; one GRAIL game and one Reasoning game &mdash; that can be used to re-run the human experiments.

The `data/results/` directory contains two CSV files, one for GRAIL (ours) and one for the Reasoning agent, each providing human evaluation results for the games. Analyses of these results are presented in the main paper and the appendix.

To re-run experiments or map logs to specific agent types, refer to `code/README.md` [here](code/README.md).

## Citation

```bibtex
@misc{rahimirad2025bayesiansocialdeductiongraphinformed,
          title={Bayesian Social Deduction with Graph-Informed Language Models}, 
          author={Shahab Rahimirad and Guven Gergerli and Lucia Romero and Angela Qian and Matthew Lyle Olson and Simon Stepputtis and Joseph Campbell},
          year={2025},
          eprint={2506.17788},
          archivePrefix={arXiv},
          primaryClass={cs.AI},
          url={https://arxiv.org/abs/2506.17788}, 
}
```

## License

MIT License - Open for research and commercial use.
