Below is a ranked improvement backlog, ordered roughly from **lowest combined cost** to **highest combined cost**, with emphasis on:

1. expanding game features;
2. improving agent win rate;
3. ideas that may replace or significantly rework the current GRAIL/action heuristic.

I’m treating “cost” as both **implementation cost** and **training/data cost**.

## Cost Tier 0: No Training, Small Code Changes

### 1. Build a real Evil/Minion action heuristic
**Goal:** Let GRAIL play Evil intelligently.

Current issue: the heuristic is Good-oriented. It proposes high-`P(Good)` teams and rejects teams containing likely Evil players. That is exactly wrong for Minions.

**Improvement:**

- If Evil, propose teams containing at least one Evil player.
- Approve teams that include Evil players.
- Reject teams that look too clean.
- Fail quests strategically, not always.
- Avoid obvious collusion with known Evil teammate.

**Training cost:** none.  
**Implementation cost:** low.  
**Win-rate impact:** high for Evil agents.  
**Feature impact:** enables GRAIL Minion games.

This is probably the best first improvement.

---

### 2. Add role-aware quest-fail strategy for Evil agents
Current `vote_for_quest` simply fails if Evil. That can be too obvious.

**Improvement:**

- On early quests, sometimes play success to build trust.
- If multiple Evil players are on a quest, coordinate so only one fail is cast.
- If Evil already leads 2 failed quests, fail aggressively.
- If Good is close to winning, fail aggressively.
- If failing would expose a single Evil too early, consider passing.

**Training cost:** none.  
**Implementation cost:** low.  
**Win-rate impact:** medium to high for Evil agents.

---

### 3. Make the action heuristic game-score aware
The current party vote heuristic mostly checks whether members are more likely Good than Evil. It does not deeply account for game pressure.

**Improvement:**

- If four party proposals have already failed, approve more often to avoid automatic Evil win/loss edge cases.
- If Good is at 2 successes, become stricter.
- If Evil is at 2 failures, become stricter as Good and more aggressive as Evil.
- Account for quest size and required fail count if expanded rules are added.

**Training cost:** none.  
**Implementation cost:** low.  
**Win-rate impact:** medium.

---

### 4. Add thresholds instead of hard `P(Evil) > P(Good)`
Currently the heuristic rejects a player when evil probability exceeds good probability, i.e. roughly `P(Evil) > 0.5`.

**Improvement:**

Use tunable thresholds:

```text
Good agent:
  reject if any party member has P(Evil) > 0.60

Evil agent:
  approve if party has known/likely Evil
  reject clean teams unless forced
```

Could vary by quest number.

**Training cost:** none.  
**Implementation cost:** low.  
**Win-rate impact:** medium.

---

### 5. Fix first-quest party behavior
Current heuristic always approves parties of size 2 and proposes one top-good player plus a random partner.

**Improvement:**

- As Good, prefer self plus one low-risk player.
- As Evil, prefer self or teammate plus a plausible Good player.
- Avoid random partner selection when beliefs exist.
- Use leader/proposer behavior as weak evidence.

**Training cost:** none.  
**Implementation cost:** low.  
**Win-rate impact:** small to medium.

---

### 6. Improve the LLM prompts for Evil agents
The current prompts are mostly framed around Good reasoning.

**Improvement:**

- Add Evil-specific prompt templates.
- Tell Evil agents to appear uncertain, helpful, and consistent.
- Avoid repeating suspicious canned phrases.
- Encourage plausible alternative suspicion narratives.
- Use private knowledge of Evil teammate carefully.

**Training cost:** none.  
**Implementation cost:** low to medium.  
**Win-rate impact:** high for Evil social plausibility.

---

## Cost Tier 1: No New Training, Moderate Code Changes

### 7. Add a policy layer separate from `HeuristicOracle`
Right now, action selection is too compressed into a simple heuristic class.

**Improvement:**

Create separate policies:

```text
GoodPolicy
EvilPolicy
MerlinPolicy
PercivalPolicy
AssassinPolicy
```

Even before adding special roles, `GoodPolicy` and `EvilPolicy` would make behavior cleaner.

**Training cost:** none.  
**Implementation cost:** medium.  
**Win-rate impact:** high.

---

### 8. Add self-consistency checks between message and action
Agents can say one thing and do another.

**Improvement:**

Before sending a message or action:

- verify the proposed party matches the message;
- verify accusations are compatible with current beliefs;
- avoid claiming impossible facts;
- flag contradictions like “I trust Paul” then voting against Paul’s team without explanation.

**Training cost:** none.  
**Implementation cost:** medium.  
**Win-rate impact:** medium.  
**Quality impact:** high.

---

### 9. Use LLM priors more carefully
Currently the LLM “vibes” can update priors, but this is risky if noisy.

**Improvement:**

- Blend graph posterior and LLM prior with a tunable weight.
- Make the blend depend on quest number.
- Reduce LLM influence early when evidence is weak.
- Increase LLM influence when chat contains concrete contradictions.
- Track uncertainty rather than only point probabilities.

**Training cost:** none.  
**Implementation cost:** medium.  
**Win-rate impact:** medium to high.

---

### 10. Add logging/evaluation dashboards for agent decisions
Before changing models, make failures easier to see.

**Improvement:**

Log per turn:

- graph beliefs;
- LLM prior beliefs;
- final blended beliefs;
- selected action;
- alternative actions considered;
- whether action helped or hurt after ground truth reveal.

**Training cost:** none.  
**Implementation cost:** medium.  
**Win-rate impact:** indirect but important.

---

### 11. Add stronger hallucination prevention at runtime
The paper discusses hallucination detection, but runtime behavior could be stricter.

**Improvement:**

- Add a fact-checking pass before messages are sent.
- Prevent claims about impossible/private information.
- Prevent references to future or hidden data.
- Validate quest history claims against actual logs.

**Training cost:** none, unless training a detector.  
**Implementation cost:** medium.  
**Win-rate impact:** medium, especially with humans.

---

## Cost Tier 2: Low Training, Moderate Implementation

### 12. Tune heuristic parameters through self-play
Instead of hand-picking thresholds, optimize them.

**Improvement:**

Use grid search or Bayesian optimization over parameters:

```text
Good rejection threshold
Evil rejection threshold
early fail probability
late fail probability
LLM prior weight
risk tolerance by quest number
```

Run many simulated games and select best configs.

**Training cost:** low.  
**Implementation cost:** medium.  
**Win-rate impact:** high.

This is not neural training, but it is still empirical policy optimization.

---

### 13. Train a lightweight action-value model
Keep the factor graph, but replace the heuristic with a small learned policy/value model.

**Inputs:**

- graph beliefs;
- quest number;
- score;
- proposed party;
- leader;
- failed proposal count;
- agent team;
- party size.

**Outputs:**

- approve/reject value;
- candidate team score;
- fail/pass probability for Evil.

**Training data:**

- generated self-play games;
- existing logs if action/outcome labels are usable.

**Training cost:** low to medium.  
**Implementation cost:** medium.  
**Win-rate impact:** high.

---

### 14. Add opponent modeling features without changing the factor graph
The current factor graph uses only party composition, party votes, and quest outcomes. Many useful behavioral signals are outside it.

**Improvement:**

Compute separate heuristic features:

- who proposed failed teams;
- who approved failed teams;
- who repeatedly avoids certain players;
- who switches votes oddly;
- who pushes teams including themselves;
- who contradicts previous claims.

Feed these into the action policy or LLM prompt.

**Training cost:** none to low.  
**Implementation cost:** medium.  
**Win-rate impact:** medium to high.

---

### 15. Add chat-derived structured features
Instead of letting the LLM directly alter priors vaguely, extract structured claims.

Examples:

```text
Alice accuses Bob
Alice defends Carol
Alice claims uncertainty
Alice contradicts previous vote
Alice pushes a team including Bob
```

Use these features in policy or belief updates.

**Training cost:** low if LLM-extracted; medium if supervised.  
**Implementation cost:** medium.  
**Win-rate impact:** high in human games.

---

## Cost Tier 3: Moderate Training, Moderate to High Implementation

### 16. Retrain the neural factor with richer game-state inputs
Current neural factor only sees:

```text
party composition, approval votes, quest outcome
```

It does not directly see proposer identity, proposal sequence, rejected parties, or discussion behavior.

**Improvement:**

Extend input to include:

- proposer/leader per proposal;
- rejected party compositions;
- number of failed proposals;
- individual vote history across proposals;
- quest team member fail/success behavior;
- maybe anonymized chat-derived features.

**Training cost:** medium.  
**Implementation cost:** high.  
**Win-rate impact:** high.

This is one of the most promising “real GRAIL upgrade” paths.

---

### 17. Replace reduced categorical encodings with more general encodings
Current encoding is tightly bound to 6 players and fixed categorical spaces.

**Improvement:**

Represent game state as sets or binary matrices:

```text
party_members[player, quest]
approval_votes[player, proposal]
quest_outcome[quest]
leader[player, proposal]
```

This would make extension to more player counts easier.

**Training cost:** medium.  
**Implementation cost:** high.  
**Win-rate impact:** medium.  
**Feature impact:** high.

---

### 18. Train separate Good and Evil policy models
Rather than one generic heuristic, train policies for each side.

**Good policy learns:**

- safest party construction;
- when to reject;
- how much risk to tolerate.

**Evil policy learns:**

- when to infiltrate;
- when to sabotage;
- when to bus teammate;
- how to avoid detection.

**Training cost:** medium.  
**Implementation cost:** high.  
**Win-rate impact:** high.

---

### 19. Add full 6-player special roles: Merlin, Assassin, Percival, Morgana
The assets and role names already exist in places, but the simplified game uses Servants and Minions.

**Feature work:**

- role assignment;
- private information rules;
- Merlin knowledge;
- Percival ambiguity;
- Assassin endgame;
- Morgana deception;
- UI/tutorial updates;
- server/game rules.

**Agent work:**

- role-specific prompts;
- role-specific action policies;
- altered belief graph variables;
- assassin target selection.

**Training cost:** medium to high.  
**Implementation cost:** high.  
**Win-rate impact:** mixed initially, high once stable.  
**Feature impact:** very high.

---

## Cost Tier 4: High Training, High Implementation

### 20. Generalize beyond 6 players
Current model is deeply 6-player-specific:

- six role variables;
- exactly two Evil players;
- fixed party composition categories;
- fixed vote composition categories;
- fixed neural input size;
- hard-coded player loops.

**Improvement:**

Support 5-10 player Avalon/Resistance configurations.

**Requires:**

- dynamic game rules;
- dynamic factor graph construction;
- dynamic party/vote encodings;
- different Evil-count constraints;
- retraining or architecture redesign;
- UI/tutorial updates.

**Training cost:** high.  
**Implementation cost:** high.  
**Feature impact:** very high.

---

### 21. Replace neural factors with permutation-equivariant networks
Instead of circular permutation hacks, use an architecture naturally suited to players as exchangeable entities.

Options:

- DeepSets;
- graph neural networks;
- transformer over player/event tokens;
- equivariant role-belief model.

**Benefits:**

- cleaner support for variable player counts;
- less positional bias;
- richer event histories;
- easier extension to rejected proposals and chat-derived events.

**Training cost:** high.  
**Implementation cost:** high.  
**Win-rate impact:** high if done well.

---

### 22. Replace the factor graph with a learned belief-state model
A more radical approach: keep GRAIL’s philosophy but replace explicit factor graph inference.

**Possible model:**

```text
event sequence -> transformer/RNN/GNN -> posterior over roles
```

Inputs could include all proposals, votes, outcomes, chat embeddings, and private role info.

**Pros:**

- more expressive;
- easier to include language;
- can learn complex deception patterns.

**Cons:**

- less interpretable;
- more data-hungry;
- harder to debug;
- loses clean Bayesian constraint handling unless added explicitly.

**Training cost:** high.  
**Implementation cost:** high.  
**Win-rate impact:** potentially high.

---

### 23. End-to-end self-play reinforcement learning
Train agents by playing many games against themselves or mixtures of baselines.

**Could train:**

- proposal policy;
- vote policy;
- sabotage policy;
- dialogue strategy, if using constrained language actions;
- belief updates.

**Training cost:** very high.  
**Implementation cost:** very high.  
**Win-rate impact:** potentially very high.  
**Risk:** high complexity, unstable results, hard evaluation.

---

### 24. Multi-agent debate/planning for actions
Instead of a simple heuristic, ask one or more models to evaluate candidate actions.

Example:

```text
Generate candidate parties.
Score each party under Good objective.
Score each party under Evil objective.
Check consistency with public statements.
Select action.
```

**Training cost:** none, but inference cost high.  
**Implementation cost:** high.  
**Win-rate impact:** medium to high.  
**Risk:** expensive, slower games, more hallucination surface.

---

## Game Feature Expansion Ideas

Ranked by practicality:

### Easy

- GRAIL can play Minion.
- Add configurable all-agent matchups.
- Add better game logs and belief visualizations.
- Add “explain this decision” debug mode.
- Add adjustable heuristic thresholds in config.

### Medium

- Add Evil-specific prompts and policies.
- Add role-aware agent classes.
- Add rejected proposal history to belief/action logic.
- Add runtime hallucination checking.
- Add self-play evaluation suite.

### Hard

- Add Merlin/Assassin/Percival/Morgana.
- Add variable player counts.
- Add full Avalon rules for different player counts and quest fail thresholds.
- Add learned action policy.
- Add richer belief model with proposal/chat history.

### Very Hard

- Variable-player neural belief model.
- End-to-end self-play training.
- Language-grounded deception strategy training.
- Fully learned multi-role Avalon agent.

## My Recommended Roadmap

### Phase 1: Make Evil GRAIL viable
Do this first.

1. Implement `GoodPolicy` and `EvilPolicy`.
2. Replace Good-oriented Minion behavior and build a real Evil/Minion action heuristic.
3. Add quest-fail strategy for evil agents.
4. Add Evil-specific prompts.
5. Fix first-quest party behavior.
6. Add simple evaluation: GRAIL Good vs GRAIL Evil, GRAIL Good vs DeepSeek Evil, etc.

This should be relatively cheap and directly addresses the biggest observed limitation.

### Phase 2: Improve win rate without retraining the graph
1. Tune thresholds through self-play.
2. Add score-aware and quest-aware policy.
3. Add proposal/vote behavior features.
4. Add consistency checks between beliefs, actions, and messages.

This improves play quality while preserving the current model.

### Phase 3: Improve the belief model
1. Add rejected proposal history.
2. Add proposer identity.
3. Add richer vote/proposal sequence features.
4. Retrain neural factor functions.
5. Compare against current GRAIL.

This is the first serious model upgrade.

### Phase 4: Expand Avalon features
1. Add Assassin/Merlin first.
2. Then Percival/Morgana.
3. Then variable player counts.

Special roles are easier than variable player counts because the current implementation is already deeply hard-coded around six players.

### Phase 5: Consider replacing the architecture
Only after the above:

- learned action policy;
- transformer/GNN belief model;
- self-play RL;
- language-aware belief modeling.

These are exciting, but they are not the cheapest path to better gameplay.