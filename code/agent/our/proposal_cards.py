# proposal_cards.py
#
# Phase-3 shared proposal-card featurizer (Workstream 1) — the single source
# of truth for turning a game's full ordered proposal history (including
# rejected proposals) into the fixed-shape tensor consumed by both model
# tracks. The dataset pipeline (GameRecord events), the offline replay
# harness, AND the live agent (GameInfo proposal history) all build cards
# through here, so train-time and run-time inputs cannot skew.
#
# Canonical card: a dict with 0-based seat indices
#   quest:    int 1..5
#   prop_idx: int 1..N  (clamped to MAX_PROPOSALS_PER_QUEST when encoded)
#   proposer: int 0..5 or None when unknown
#   team:     frozenset of int 0..5
#   votes:    frozenset of int 0..5 (players who approved)
#   accepted: bool
#   forced:   bool (forced-fifth hammer variant: vote skipped, recorded as
#             unanimous approval with this flag set)
#   outcome:  None (unknown) | False (quest failed) | True (quest succeeded)
#
# Tensor layout per card (CARD_DIM columns), all seat fields rolled into the
# target player's egocentric frame (target at index 0, p -> (p - target) % 6):
#   [0:6]   proposer one-hot (all zero when unknown)
#   [6:12]  team membership
#   [12:18] approve votes
#   [18]    accepted flag
#   [19]    forced flag (Workstream 5 variant)
#   [20]    present flag (1 on every real card; mask-free consumers)
#   [21:26] quest number one-hot (1..5)
#   [26:31] proposal index one-hot (1..5)
#   [31:34] quest outcome one-hot (unknown, fail, success)

import torch

NUM_PLAYERS = 6
NUM_QUESTS = 5
MAX_PROPOSALS_PER_QUEST = 5
NUM_SLOTS = NUM_QUESTS * MAX_PROPOSALS_PER_QUEST  # 25
CARD_DIM = 34


def make_card(quest, prop_idx, proposer, team, votes, accepted, forced=False,
              outcome=None):
    return {
        "quest": int(quest),
        "prop_idx": int(prop_idx),
        "proposer": None if proposer is None else int(proposer),
        "team": frozenset(int(p) for p in team),
        "votes": frozenset(int(p) for p in votes),
        "accepted": bool(accepted),
        "forced": bool(forced),
        "outcome": outcome,
    }


def roll_index(player, target):
    """Egocentric frame: the target player sits at index 0."""
    return (player - target) % NUM_PLAYERS


def card_to_features(card, target):
    """Encode one canonical card as a CARD_DIM-long feature row for `target`."""
    row = [0.0] * CARD_DIM
    if card["proposer"] is not None:
        row[roll_index(card["proposer"], target)] = 1.0
    for p in card["team"]:
        row[6 + roll_index(p, target)] = 1.0
    for p in card["votes"]:
        row[12 + roll_index(p, target)] = 1.0
    if card["accepted"]:
        row[18] = 1.0
    if card.get("forced"):
        row[19] = 1.0
    row[20] = 1.0  # present
    quest = min(max(card["quest"], 1), NUM_QUESTS)
    row[21 + (quest - 1)] = 1.0
    prop_idx = min(max(card["prop_idx"], 1), MAX_PROPOSALS_PER_QUEST)
    row[26 + (prop_idx - 1)] = 1.0
    if card["outcome"] is None:
        row[31] = 1.0
    elif card["outcome"] is False:
        row[32] = 1.0
    else:
        row[33] = 1.0
    return row


def cards_to_tensor(cards, target, n_slots=NUM_SLOTS, dtype=torch.float32):
    """Build the [n_slots, CARD_DIM] card tensor and [n_slots] presence mask."""
    if len(cards) > n_slots:
        raise ValueError(f"{len(cards)} cards exceed the {n_slots} fixed slots")
    feats = torch.zeros((n_slots, CARD_DIM), dtype=dtype)
    mask = torch.zeros(n_slots, dtype=dtype)
    for i, card in enumerate(cards):
        feats[i] = torch.tensor(card_to_features(card, target), dtype=dtype)
        mask[i] = 1.0
    return feats, mask


# ---------------------------------------------------------------------------
# Source adapters (both must produce identical canonical cards — see the
# parity test in code/agent/test_proposal_cards.py)
# ---------------------------------------------------------------------------

def cards_from_events(events):
    """Canonical cards from GameRecord events (event_schema.py).

    Pairs each ProposalEvent with its VoteEvent; a QuestEvent attaches the
    outcome to that quest's accepted card. Works on any chronological prefix
    of an event stream that does not split a proposal/vote pair.
    """
    cards = []
    pending = None  # the proposal awaiting its vote
    for ev in events:
        kind = ev.get("type")
        if kind == "proposal":
            if pending is not None:
                raise ValueError("proposal event while another is unvoted")
            pending = ev
        elif kind == "vote":
            if pending is None or (pending["quest"], pending["round_idx"]) != (
                    ev["quest"], ev["round_idx"]):
                raise ValueError(
                    f"vote {ev['quest']}.{ev['round_idx']} without its proposal")
            votes = [s for s in range(NUM_PLAYERS) if ev["approves"][s]]
            cards.append(make_card(
                quest=ev["quest"],
                prop_idx=ev["round_idx"],
                proposer=pending.get("leader_seat"),
                team=pending["team_seats"],
                votes=votes,
                accepted=ev["accepted"],
                forced=ev.get("forced", False),
                outcome=None,
            ))
            pending = None
        elif kind == "quest":
            for card in reversed(cards):
                if card["quest"] == ev["quest"] and card["accepted"]:
                    card["outcome"] = bool(ev["success"])
                    break
            else:
                raise ValueError(
                    f"quest {ev['quest']} result without an accepted proposal")
        else:
            raise ValueError(f"unknown event type {kind!r}")
    if pending is not None:
        raise ValueError("trailing unvoted proposal")
    return cards


def cards_from_history(history, name_to_index0):
    """Canonical cards from GameInfo.policy_proposal_history() output.

    `history` entries: {quest, leader, party, votes: {name: bool}, accepted,
    quest_outcome, forced?}. `name_to_index0` maps lowercase player name ->
    0-based seat index (GameInfo.players_to_index is 1-based; subtract first).
    """
    cards = []
    prop_counters = {}
    for entry in history:
        quest = entry["quest"]
        prop_counters[quest] = prop_counters.get(quest, 0) + 1
        leader = (entry.get("leader") or "").lower()
        cards.append(make_card(
            quest=quest,
            prop_idx=prop_counters[quest],
            proposer=name_to_index0.get(leader),
            team=[name_to_index0[n.lower()] for n in entry["party"]],
            votes=[name_to_index0[n.lower()]
                   for n, v in entry["votes"].items() if v],
            accepted=entry["accepted"],
            forced=entry.get("forced", False),
            outcome=entry.get("quest_outcome"),
        ))
    return cards


def history_from_cards(cards, seats):
    """Canonical cards -> GameInfo.policy_proposal_history() dict shape.

    The inverse of cards_from_history, used by the offline harness to feed
    the real HeuristicOracle/GoodPolicy with replayed proposal history.
    `seats` maps seat index -> player name (lowercased on output).
    """
    names = [str(n).lower() for n in seats]
    history = []
    for card in cards:
        history.append({
            "quest": card["quest"],
            "leader": "" if card["proposer"] is None else names[card["proposer"]],
            "party": [names[s] for s in sorted(card["team"])],
            "votes": {names[s]: (s in card["votes"]) for s in range(NUM_PLAYERS)},
            "accepted": card["accepted"],
            "forced": card.get("forced", False),
            "quest_outcome": card["outcome"],
        })
    return history


def running_scores(cards):
    """Per-card (good_wins, evil_wins) BEFORE the card was voted.

    Derived from the outcomes attached to earlier accepted cards; used by the
    Track-B sequence input so the score channel stays in this module.
    """
    scores = []
    good = evil = 0
    for card in cards:
        scores.append((good, evil))
        if card["accepted"] and card["outcome"] is not None:
            if card["outcome"]:
                good += 1
            else:
                evil += 1
    return scores


# ---------------------------------------------------------------------------
# Training snapshots
# ---------------------------------------------------------------------------

def enumerate_snapshots(cards):
    """All belief-relevant prefixes of a game's card stream.

    Yields (snapshot_cards, is_quest_boundary) in chronological order:
      - the empty state,
      - after every voted proposal (a just-accepted card shows outcome=None),
      - after every quest resolution (the accepted card's outcome filled in).
    A quest-resolution snapshot doubles as the start-of-next-quest state, so
    it is emitted once with is_quest_boundary=True.
    """
    snapshots = [([], True)]
    seen = []
    for card in cards:
        if card["accepted"] and card["outcome"] is not None:
            pending = dict(card)
            pending["outcome"] = None
            seen.append(pending)
            snapshots.append((list(seen), False))
            seen[-1] = card
            snapshots.append((list(seen), True))
        else:
            seen.append(card)
            snapshots.append((list(seen), False))
    return snapshots


# ---------------------------------------------------------------------------
# Feature ablations (evaluable increments for the attribution checkpoint)
# ---------------------------------------------------------------------------

# tensor column ranges, must match card_to_features above
COL_PROPOSER = slice(0, 6)
COL_ACCEPTED = 18
COL_PROP_IDX = slice(26, 31)

FEATURE_ABLATIONS = ("none", "no_rejected", "no_proposer")


def apply_feature_ablation(cards, mask, mode):
    """Ablate encoded card tensors for the goal-attribution runs.

    cards: [..., S, CARD_DIM], mask: [..., S] (not modified in place). Modes:
      none        — unchanged.
      no_proposer — zero the proposer one-hot (isolates proposer identity).
      no_rejected — mask out rejected-proposal cards entirely and strip the
                    within-quest proposal index from the survivors (which
                    would otherwise leak the rejection count); the model then
                    sees only what v2-style accepted-proposal streams carry
                    (isolates the rejected-history signal).
    """
    if mode == "none":
        return cards, mask
    cards = cards.clone()
    mask = mask.clone()
    if mode == "no_proposer":
        cards[..., COL_PROPOSER] = 0
    elif mode == "no_rejected":
        accepted = cards[..., COL_ACCEPTED] > 0
        keep = accepted & (mask > 0)
        mask = keep.to(mask.dtype)
        cards = cards * mask.unsqueeze(-1)
        cards[..., COL_PROP_IDX] = 0
    else:
        raise ValueError(f"unknown feature ablation '{mode}'")
    return cards, mask


# ---------------------------------------------------------------------------
# v2 compatibility (legacy 21-int vector for the factor_v2 replay adapter and
# controlled comparisons)
# ---------------------------------------------------------------------------

def v2_vector_from_cards(cards):
    """Project a card stream onto the legacy 21-int v2 state vector.

    Only resolved quests' accepted proposals are representable in v2 — this is
    exactly the information loss the card representation removes. Roles (the
    first 6 ints) are left at 0; FactorGraphModelV2.predict_probs fills in the
    ego role itself. Matches GameInfo.get_state_vector (which reads proposal
    [-1] of each resolved quest — always the accepted one, since acceptance
    ends the quest's proposal sequence).
    """
    from itertools import combinations

    players = list(range(1, NUM_PLAYERS + 1))
    vote_compositions = []
    for size in range(4, NUM_PLAYERS + 1):
        for subset in combinations(players, size):
            vote_compositions.append(subset)

    vector = [0] * (NUM_PLAYERS + 3 * NUM_QUESTS)
    for card in cards:
        if not (card["accepted"] and card["outcome"] is not None):
            continue
        q = card["quest"]
        team = tuple(sorted(p + 1 for p in card["team"]))
        votes = tuple(sorted(p + 1 for p in card["votes"]))
        party_index = list(combinations(players, len(team))).index(team)
        vote_index = vote_compositions.index(votes)
        vector[6 + (q - 1) * 3] = party_index + 1
        vector[6 + (q - 1) * 3 + 1] = vote_index + 1
        vector[6 + (q - 1) * 3 + 2] = int(card["outcome"]) + 1
    return vector
