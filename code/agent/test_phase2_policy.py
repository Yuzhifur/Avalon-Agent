"""Focused unit checks for the Phase 2 policy and consistency layer.

Run from code/agent/: python3 test_phase2_policy.py
"""

import importlib.util
import os
import sys
import types
from enum import Enum


try:
    from agent_base import ATEAM
except ModuleNotFoundError:
    class ATEAM(Enum):
        GOOD = 1
        EVIL = 2

    stub = types.ModuleType("agent_base")
    stub.ATEAM = ATEAM
    sys.modules["agent_base"] = stub


ROOT = os.path.dirname(__file__)


def load_module(name, relative_path):
    path = os.path.join(ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heuristic = load_module(
    "phase2_heuristic",
    os.path.join("our", "policy_models", "heuristic.py"),
)
consistency = load_module(
    "phase2_consistency",
    os.path.join("our", "policy_models", "consistency.py"),
)
HeuristicOracle = heuristic.HeuristicOracle
MessageConsistencyChecker = consistency.MessageConsistencyChecker


def probs(**evil_probabilities):
    return {
        name: {"evil": value, "good": round(1 - value, 4)}
        for name, value in evil_probabilities.items()
    }


def make(
    role,
    *,
    self_name="sam",
    teammates=("paul",),
    failed=0,
    quest=2,
    good_wins=0,
    evil_wins=0,
    config=None,
    history=None,
):
    oracle = HeuristicOracle(role, policy_config=config)
    oracle.set_identity(self_name, set(teammates))
    oracle.update_context(
        failed_party_votes=failed,
        quest_number=quest,
        good_wins=good_wins,
        evil_wins=evil_wins,
        party_size=3,
        leader="jane",
        proposal_history=history or [],
        quest_history=[],
    )
    return oracle


passed = 0
failed_cases = []


def check(name, condition):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed_cases.append(name)
        print(f"  FAIL  {name}")


P = probs(sam=0.20, paul=0.68, luca=0.50, jane=0.15, kira=0.25, mia=0.35)

print("== configurable Good policy ==")
good = make(ATEAM.GOOD, teammates=())
check(
    "Good includes self in proposals",
    "sam" in good.propose_party(3, P),
)
check(
    "Good approves final proposal to avoid automatic Evil win",
    make(ATEAM.GOOD, teammates=(), failed=4).vote_for_party(
        ["paul", "luca", "mia"], P
    ) is True,
)
strict = make(
    ATEAM.GOOD,
    teammates=(),
    config={"good": {"reject_thresholds": {"2": 0.45}}},
)
check(
    "config override changes Good rejection threshold",
    strict.vote_for_party(["luca", "jane", "kira"], P) is False,
)
match_point = make(ATEAM.GOOD, teammates=(), quest=4, good_wins=2)
check(
    "Good becomes stricter at match point",
    match_point.vote_for_party(["luca", "jane", "kira"], P) is False,
)

print("== mechanical behavior features ==")
failed_history = [{
    "quest": 1,
    "leader": "jane",
    "party": ["luca", "jane"],
    "votes": {
        "sam": True,
        "paul": True,
        "luca": True,
        "jane": True,
        "kira": False,
        "mia": False,
    },
    "accepted": True,
    "quest_outcome": False,
}]
history_good = make(
    ATEAM.GOOD,
    teammates=(),
    history=failed_history,
)
check(
    "failed-quest membership raises policy risk",
    history_good.vote_for_party(["luca", "kira", "mia"], P) is False,
)
rejected_history = [{
    "quest": 2,
    "leader": "luca",
    "party": ["luca", "jane", "mia"],
    "votes": {
        "sam": False,
        "paul": True,
        "luca": True,
        "jane": False,
        "kira": False,
        "mia": True,
    },
    "accepted": False,
    "quest_outcome": None,
}]
rejected_signal = make(
    ATEAM.GOOD,
    teammates=(),
    history=rejected_history,
)
P_REJECTED = probs(sam=0.20, paul=0.40, luca=0.58, jane=0.15, kira=0.25, mia=0.35)
check(
    "rejected proposal leadership and approval affect policy risk",
    rejected_signal.vote_for_party(["luca", "jane", "kira"], P_REJECTED) is False,
)
decision = history_good.decision_log_payload()
check(
    "decision log records alternatives and scores",
    decision.get("alternatives") == [True, False] and "party_risk" in decision.get("scores", {}),
)

print("== Evil vote theater and score safety ==")
distance_config = {
    "evil": {
        "teammate_distance_rate": 1.0,
        "teammate_distance_threshold": 0.60,
        "clean_team_cover_approve_rate": 0.0,
        "evil_vote_split_rate": 0.0,
    }
}
distance = make(ATEAM.EVIL, config=distance_config)
check(
    "Evil distances from exposed teammate when score is safe",
    distance.vote_for_party(["paul", "jane", "kira"], P) is False,
)
urgent = make(ATEAM.EVIL, config=distance_config, good_wins=2)
check(
    "score pressure overrides teammate distancing",
    urgent.vote_for_party(["paul", "jane", "kira"], P) is True,
)
split_config = {
    "evil": {
        "teammate_distance_rate": 0.0,
        "evil_vote_split_rate": 1.0,
    }
}
split_sam = make(ATEAM.EVIL, config=split_config, self_name="sam", teammates=("paul",))
split_paul = make(ATEAM.EVIL, config=split_config, self_name="paul", teammates=("sam",))
check(
    "two Evil agents deterministically split a cover vote",
    split_sam.vote_for_party(["sam", "paul", "jane"], P) is False
    and split_paul.vote_for_party(["sam", "paul", "jane"], P) is True,
)
final_reject = make(ATEAM.EVIL, config=distance_config, failed=4)
check(
    "Evil rejects the fifth proposal for the automatic win",
    final_reject.vote_for_party(["jane", "kira", "mia"], P) is False,
)
social = make(ATEAM.EVIL, config=distance_config)
social.propose_party(3, P)
plan = social.public_plan_summary().lower()
check(
    "fake Evil hypothesis excludes known Evil identities",
    "sam deserve" not in plan and "paul deserve" not in plan,
)

print("== consistency repair ==")
checker = MessageConsistencyChecker()
reaction = checker.repair(
    "i'm voting no, this team is bad",
    mode="party_reaction",
    party=["sam", "jane", "kira"],
    planned_vote=True,
)
check(
    "vote contradiction is repaired",
    reaction["changed"] and "approving" in reaction["message"],
)
proposal = checker.repair(
    "i like this group",
    mode="proposal",
    party=["sam", "jane", "kira"],
)
check(
    "proposal message is aligned to selected party",
    all(name in proposal["message"].lower() for name in ("sam", "jane", "kira")),
)
history = checker.repair(
    "quest 1 succeeded so i trust that team",
    mode="party_reaction",
    party=["sam", "jane", "kira"],
    planned_vote=True,
    quest_history=[(["sam", "jane"], "fail")],
)
check(
    "incorrect quest-history claim is corrected",
    "quest 1 failed" in history["message"].lower(),
)

print(f"\n{passed} passed, {len(failed_cases)} failed")
if failed_cases:
    print("FAILED:", failed_cases)
    raise SystemExit(1)
print("All Phase 2 policy checks passed.")
