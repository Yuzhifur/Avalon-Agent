# test_proposal_cards.py
#
# Phase-3 Workstream-1 unit tests: train/runtime featurizer parity, the
# egocentric roll, snapshot enumeration, forced-fifth encoding, running
# scores, and legacy v2-vector compatibility. The real-log parity test drives
# the actual GameInfo class with a committed phase-2 server log and compares
# against the dataset pipeline — the train/serve skew guarantee on real data.
#
# Runnable with pytest or directly:  python test_proposal_cards.py

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "our"))
sys.path.insert(0, os.path.join(HERE, "our", "training", "dataset"))

from proposal_cards import (  # noqa: E402
    CARD_DIM,
    NUM_SLOTS,
    apply_feature_ablation,
    cards_from_events,
    cards_from_history,
    cards_to_tensor,
    card_to_features,
    enumerate_snapshots,
    make_card,
    running_scores,
    v2_vector_from_cards,
)
from event_schema import GameRecord  # noqa: E402
import parse_avalonlogs  # noqa: E402
import parse_selfplay  # noqa: E402

PLAYERS = ["ALICE", "BOB", "CARA", "DAVE", "EVE", "FRED"]
NAME_TO_INDEX0 = {n.lower(): i for i, n in enumerate(PLAYERS)}

REAL_SERVER_LOG = os.path.join(
    HERE, "..", "evaluation", "policy_runs_overnight", "20260608T180435Z",
    "evil_baseline", "run_001", "server", "VFTG.json")


def synthetic_avalonlogs_game():
    """Quest 1 (size 2): rejected then approved+success.
    Quest 2 (size 3): approved, failed.
    Quest 3 (size 4): one rejected proposal, then approved, unresolved."""
    return {
        "players": [{"name": n} for n in PLAYERS],
        "missions": [
            {
                "teamSize": 2, "state": "SUCCESS",
                "proposals": [
                    {"state": "REJECTED", "proposer": "ALICE",
                     "team": ["ALICE", "BOB"],
                     "votes": ["ALICE", "BOB"]},
                    {"state": "APPROVED", "proposer": "BOB",
                     "team": ["BOB", "CARA"],
                     "votes": ["BOB", "CARA", "DAVE", "EVE"]},
                ],
            },
            {
                "teamSize": 3, "state": "FAIL",
                "proposals": [
                    {"state": "APPROVED", "proposer": "CARA",
                     "team": ["CARA", "DAVE", "FRED"],
                     "votes": ["ALICE", "CARA", "DAVE", "FRED"]},
                ],
            },
            {
                "teamSize": 4, "state": "PENDING",
                "proposals": [
                    {"state": "REJECTED", "proposer": "DAVE",
                     "team": ["DAVE", "EVE", "FRED", "ALICE"],
                     "votes": ["DAVE", "EVE"]},
                    {"state": "APPROVED", "proposer": "EVE",
                     "team": ["BOB", "CARA", "DAVE", "EVE"],
                     "votes": ["BOB", "CARA", "DAVE", "EVE"]},
                ],
            },
        ],
        "outcome": {
            "state": "EVIL_WIN",
            "message": "Three failed missions",
            "roles": [
                {"name": "ALICE", "role": "LOYAL FOLLOWER"},
                {"name": "BOB", "role": "EVIL MINION"},
                {"name": "CARA", "role": "LOYAL FOLLOWER"},
                {"name": "DAVE", "role": "EVIL MINION"},
                {"name": "EVE", "role": "LOYAL FOLLOWER"},
                {"name": "FRED", "role": "LOYAL FOLLOWER"},
            ],
        },
    }


def synthetic_runtime_history():
    """The same game expressed as GameInfo.policy_proposal_history() output
    (lowercase names, full vote dicts, quest_outcome on accepted cards)."""
    def votes(approve, reject):
        v = {n.lower(): True for n in approve}
        v.update({n.lower(): False for n in reject})
        return v

    return [
        {"quest": 1, "leader": "alice", "party": ["alice", "bob"],
         "votes": votes(["alice", "bob"], ["cara", "dave", "eve", "fred"]),
         "accepted": False, "quest_outcome": None},
        {"quest": 1, "leader": "bob", "party": ["bob", "cara"],
         "votes": votes(["bob", "cara", "dave", "eve"], ["alice", "fred"]),
         "accepted": True, "quest_outcome": True},
        {"quest": 2, "leader": "cara", "party": ["cara", "dave", "fred"],
         "votes": votes(["alice", "cara", "dave", "fred"], ["bob", "eve"]),
         "accepted": True, "quest_outcome": False},
        {"quest": 3, "leader": "dave", "party": ["dave", "eve", "fred", "alice"],
         "votes": votes(["dave", "eve"], ["alice", "bob", "cara", "fred"]),
         "accepted": False, "quest_outcome": None},
        {"quest": 3, "leader": "eve", "party": ["bob", "cara", "dave", "eve"],
         "votes": votes(["bob", "cara", "dave", "eve"], ["alice", "fred"]),
         "accepted": True, "quest_outcome": None},
    ]


def pipeline_cards():
    record = parse_avalonlogs.parse_game(synthetic_avalonlogs_game(), "synthetic")
    return record, cards_from_events(record.events)


def test_train_runtime_parity():
    """The dataset pipeline (parser -> events -> cards) and the runtime
    adapter (GameInfo history -> cards) must build identical cards and
    identical tensors — the train/serve skew guarantee."""
    record, cards_train = pipeline_cards()
    cards_runtime = cards_from_history(synthetic_runtime_history(), NAME_TO_INDEX0)
    assert cards_train == cards_runtime
    for target in range(6):
        feats_a, mask_a = cards_to_tensor(cards_train, target)
        feats_b, mask_b = cards_to_tensor(cards_runtime, target)
        assert torch.equal(feats_a, feats_b)
        assert torch.equal(mask_a, mask_b)
    assert record.evil == [0, 1, 0, 1, 0, 0]
    assert record.winner == "evil" and record.win_reason == "three_fails"


def test_card_layout_and_roll():
    _, cards = pipeline_cards()
    assert len(cards) == 5
    first = cards[0]
    assert first["quest"] == 1 and first["prop_idx"] == 1
    assert first["proposer"] == 0 and not first["accepted"]
    assert first["team"] == frozenset({0, 1})

    # accepted quest-1 card carries the success outcome; quest-3 accepted
    # card is unresolved
    assert cards[1]["outcome"] is True
    assert cards[2]["outcome"] is False
    assert cards[4]["accepted"] and cards[4]["outcome"] is None

    # egocentric roll: target 1 (BOB) sees itself at index 0
    feats, mask = cards_to_tensor(cards, target=1)
    assert mask.sum().item() == 5
    # card 0: proposer ALICE (seat 0) -> rolled index 5 for target 1
    assert feats[0, 5].item() == 1.0
    # card 0 team {ALICE, BOB} -> rolled {5, 0}
    assert feats[0, 6 + 0].item() == 1.0 and feats[0, 6 + 5].item() == 1.0
    # accepted/forced/present flags
    assert feats[0, 18].item() == 0.0   # rejected
    assert feats[0, 19].item() == 0.0   # not forced
    assert feats[0, 20].item() == 1.0   # present
    # quest 1 one-hot, prop_idx 1 one-hot, outcome unknown
    assert feats[0, 21].item() == 1.0
    assert feats[0, 26].item() == 1.0
    assert feats[0, 31].item() == 1.0

    assert feats.shape == (NUM_SLOTS, CARD_DIM)


def shift_card_seats(card, k):
    """Rotate every seat field of a canonical card by +k (mod 6)."""
    return make_card(
        quest=card["quest"], prop_idx=card["prop_idx"],
        proposer=None if card["proposer"] is None else (card["proposer"] + k) % 6,
        team=[(p + k) % 6 for p in card["team"]],
        votes=[(p + k) % 6 for p in card["votes"]],
        accepted=card["accepted"], forced=card["forced"],
        outcome=card["outcome"],
    )


def test_roll_identity():
    """Plan verification: rolling seats by 1 six times is the identity, and
    the egocentric encoding is rotation-equivariant — shifting all seats by k
    while shifting the target by k leaves the features unchanged."""
    _, cards = pipeline_cards()
    shifted = cards
    for _ in range(6):
        shifted = [shift_card_seats(c, 1) for c in shifted]
    assert shifted == cards

    base = [card_to_features(c, target=0) for c in cards]
    for k in range(6):
        rolled = [card_to_features(shift_card_seats(c, k), target=k)
                  for c in cards]
        assert rolled == base


def test_forced_bit():
    card = make_card(quest=3, prop_idx=5, proposer=2, team=[2, 3, 4, 5],
                     votes=range(6), accepted=True, forced=True)
    row = card_to_features(card, target=0)
    assert row[18] == 1.0 and row[19] == 1.0
    # forced survives the events round trip
    events = [
        {"type": "proposal", "quest": 3, "round_idx": 5, "leader_seat": 2,
         "team_seats": [2, 3, 4, 5]},
        {"type": "vote", "quest": 3, "round_idx": 5,
         "approves": [True] * 6, "accepted": True, "forced": True},
    ]
    assert cards_from_events(events) == [card]
    # history side: prop_idx is inferred by counting voted proposals within
    # the quest, so stage four rejected proposals before the forced fifth
    reject = {"quest": 3, "leader": "bob", "party": ["bob", "cara"],
              "votes": {n.lower(): n == "BOB" for n in PLAYERS},
              "accepted": False, "quest_outcome": None}
    history = [dict(reject) for _ in range(4)] + [
        {"quest": 3, "leader": "cara",
         "party": ["cara", "dave", "eve", "fred"],
         "votes": {n.lower(): True for n in PLAYERS},
         "accepted": True, "forced": True, "quest_outcome": None}]
    assert cards_from_history(history, NAME_TO_INDEX0)[-1] == card


def test_snapshot_enumeration():
    _, cards = pipeline_cards()
    snapshots = enumerate_snapshots(cards)
    # 1 empty + 5 proposal states + 2 quest resolutions = 8
    assert len(snapshots) == 8
    assert snapshots[0] == ([], True)
    # the state right after quest-1 acceptance shows outcome unknown ...
    after_accept = snapshots[2][0]
    assert after_accept[-1]["accepted"] and after_accept[-1]["outcome"] is None
    assert snapshots[2][1] is False
    # ... and the next snapshot resolves it as a quest boundary
    resolved = snapshots[3][0]
    assert resolved[-1]["outcome"] is True
    assert snapshots[3][1] is True
    # final snapshot: all 5 cards, quest 3 accepted but unresolved
    assert len(snapshots[-1][0]) == 5
    assert snapshots[-1][1] is False


def test_running_scores():
    _, cards = pipeline_cards()
    assert running_scores(cards) == [
        (0, 0), (0, 0), (1, 0), (1, 1), (1, 1)]


def test_v2_vector_compatibility():
    """v2_vector_from_cards must reproduce the legacy vectorizer's output on
    resolved quests (the only states v2 can express)."""
    from generate_dataset_1 import vectorize_game_history

    game = synthetic_avalonlogs_game()
    legacy = vectorize_game_history(game, circular=False, partial=False)[0]
    _, cards = pipeline_cards()
    ours = v2_vector_from_cards(cards)
    # observable fields only (legacy fills roles into the first 6 slots)
    assert ours[6:] == legacy[6:]


def test_event_schema_roundtrip():
    record, _ = pipeline_cards()
    clone = GameRecord.from_json_line(record.to_json_line())
    assert clone.to_dict() == record.to_dict()
    assert cards_from_events(clone.events) == cards_from_events(record.events)


def test_gameinfo_replay_parity_real_log():
    """Drive the real GameInfo with a committed phase-2 server log (the same
    inputs ACLAgent.addMessage feeds it) and compare its card stream against
    the dataset pipeline's. This is the parity test the plan gates on."""
    from agent_acl import GameInfo

    full = parse_selfplay.load_full_state(REAL_SERVER_LOG)
    record = parse_selfplay.parse_full_state(full, "VFTG")
    cards_pipeline = cards_from_events(record.events)

    game = GameInfo()
    game.players_to_index = {n.lower(): i + 1 for i, n in enumerate(record.seats)}
    game.index_to_players = {i + 1: n.lower() for i, n in enumerate(record.seats)}

    # Replay the same decoded messages the agent would have seen.
    pending_team, pending_leader, pending_votes = None, None, None
    for msg in full["messages"]:
        if msg.get("player") != "system":
            continue
        text = msg["msg"]
        if " proposed a party: " in text:
            leader, team_str = text.split(" proposed a party: ", 1)
            pending_leader = leader.strip()
            pending_team = [n.strip() for n in team_str.split(",")]
        elif text.startswith("Party vote summary:"):
            body = text.split("Party vote summary:", 1)[1].strip()
            pending_votes = {}
            for part in body.split(", "):
                name, vote = part.rsplit(": ", 1)
                pending_votes[name.strip()] = vote.strip().lower() == "yes"
        elif text.startswith("The party has been"):
            game.current_proposed_party = pending_team
            game.party_leader = pending_leader
            game.add_party_proposal(
                pending_team, pending_votes,
                len(game.quest_results) + 1, leader=pending_leader)
        elif text.startswith("The quest has"):
            game.add_quest_result("succeeded" in text)

    name_to_index0 = {n: i - 1 for n, i in game.players_to_index.items()}
    cards_runtime = cards_from_history(
        game.policy_proposal_history(), name_to_index0)

    assert cards_runtime == cards_pipeline
    for target in range(6):
        feats_a, _ = cards_to_tensor(cards_pipeline, target)
        feats_b, _ = cards_to_tensor(cards_runtime, target)
        assert torch.equal(feats_a, feats_b)

    # and the legacy 21-vector projection matches GameInfo.get_state_vector
    legacy = game.get_state_vector()
    assert v2_vector_from_cards(cards_pipeline)[6:] == legacy[6:]


def test_forced_fifth_server_messages():
    """Workstream-5 parser check: the marker -> unanimous summary -> approval
    sequence the server emits under AVALON_HAMMER_RULE=forced_fifth must
    produce a forced, accepted VoteEvent and a forced card."""
    def sysmsg(text, quest):
        return {"player": "system", "msg": text, "quest": quest, "turn": 1}

    names = [n.capitalize() for n in PLAYERS]
    yes_summary = ", ".join(f"{n}: yes" for n in names)
    reject_summary = ", ".join(
        f"{n}: {'yes' if i < 2 else 'no'}" for i, n in enumerate(names))
    messages = []
    for _ in range(4):  # four rejected proposals
        messages += [
            sysmsg("Alice proposed a party: Alice, Bob", 1),
            sysmsg(f"Party vote summary: {reject_summary}", 1),
            sysmsg("The party has been rejected!", 1),
        ]
    messages += [
        sysmsg("Bob proposed a party: Bob, Cara", 1),
        sysmsg(parse_selfplay.FORCED_FIFTH_MARKER, 1),
        sysmsg(f"Party vote summary: {yes_summary}", 1),
        sysmsg("The party has been approved!", 1),
        sysmsg("Voting for the quest has started...", 1),
        sysmsg("The quest has succeeded!", 1),
        sysmsg("Good wins by succeeding three quests!", 1),
    ]
    full = {
        "all_players": [
            {"id": i + 1, "name": n,
             "role": ("Minion-1" if n in ("BOB", "DAVE") else "Servant-1")}
            for i, n in enumerate(PLAYERS)
        ],
        "winner": "good",
        "messages": messages,
    }
    record = parse_selfplay.parse_full_state(full, "forced_test")
    votes = [e for e in record.events if e["type"] == "vote"]
    assert len(votes) == 5
    assert [v["forced"] for v in votes] == [False] * 4 + [True]
    assert votes[-1]["accepted"] and all(votes[-1]["approves"])
    cards = cards_from_events(record.events)
    assert cards[-1]["forced"] and cards[-1]["prop_idx"] == 5
    assert record.win_reason == "three_successes"


def test_empty_state_tensor():
    feats, mask = cards_to_tensor([], target=0)
    assert mask.sum().item() == 0
    assert feats.abs().sum().item() == 0


def test_feature_ablation_no_proposer():
    _, cards = pipeline_cards()
    feats, mask = cards_to_tensor(cards, target=0)
    feats_b, mask_b = apply_feature_ablation(
        feats.unsqueeze(0), mask.unsqueeze(0), "no_proposer")
    assert feats_b[0, :, 0:6].abs().sum().item() == 0
    # everything else untouched
    assert torch.equal(feats_b[0, :, 6:], feats[:, 6:])
    assert torch.equal(mask_b[0], mask)
    # original not mutated
    assert feats[0, 0:6].abs().sum().item() > 0


def test_feature_ablation_no_rejected():
    _, cards = pipeline_cards()
    feats, mask = cards_to_tensor(cards, target=0)
    feats_b, mask_b = apply_feature_ablation(
        feats.unsqueeze(0), mask.unsqueeze(0), "no_rejected")
    # 5 cards, 2 of them rejected -> 3 survive
    assert mask_b.sum().item() == 3
    # rejected slots (0 and 3) fully zeroed, accepted slots keep their team
    assert feats_b[0, 0].abs().sum().item() == 0
    assert feats_b[0, 3].abs().sum().item() == 0
    assert feats_b[0, 1, 6:12].sum().item() == 2  # quest-1 accepted team of 2
    # proposal-index field stripped from the survivors (leaks reject count)
    assert feats_b[0, :, 26:31].abs().sum().item() == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
