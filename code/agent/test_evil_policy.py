"""Fast unit driver for the evil HeuristicOracle logic (no Docker / no LLM).

Run from code/agent/:   python test_evil_policy.py
Exercises propose_party / vote_for_party / vote_for_quest for an evil agent on
crafted belief dicts. Belief dicts are keyed by lowercase player name.
"""

import sys
import types
from enum import Enum

# The full agent_base imports pydantic (only present in the Docker image). The
# heuristic only needs ATEAM, so stub a minimal agent_base module to run the
# pure policy logic standalone.
try:
    from agent_base import ATEAM
except ModuleNotFoundError:
    class ATEAM(Enum):
        GOOD = 1
        EVIL = 2
    _stub = types.ModuleType("agent_base")
    _stub.ATEAM = ATEAM
    sys.modules["agent_base"] = _stub

# Load heuristic.py directly by path so we skip our/__init__.py (which imports torch).
import importlib.util
import os
_hpath = os.path.join(os.path.dirname(__file__), "our", "policy_models", "heuristic.py")
_spec = importlib.util.spec_from_file_location("heuristic_standalone", _hpath)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HeuristicOracle = _mod.HeuristicOracle


def probs(**kw):
    """Build a probabilities dict from name=evil_prob pairs."""
    return {name: {"evil": p, "good": round(1 - p, 4)} for name, p in kw.items()}


def make(role=ATEAM.EVIL, self_name="sam", teammates=("paul",),
         failed=0, quest=2, good_wins=0, evil_wins=0):
    o = HeuristicOracle(role)
    o.set_identity(self_name, set(teammates))
    o.update_context(failed_party_votes=failed, quest_number=quest,
                     good_wins=good_wins, evil_wins=evil_wins)
    return o


passed = 0
failed_cases = []

def check(name, cond):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed_cases.append(name)
        print(f"  FAIL  {name}")


# Beliefs: paul (teammate) looks clean (0.2), sam (self) looks clean (0.3),
# luca burned-evil-looking good 0.6, others good.
P = probs(sam=0.3, paul=0.2, luca=0.6, jane=0.1, kira=0.15, mia=0.25)

print("== propose_party (evil) ==")
o = make(quest=2)
party = o._evil_propose_party(3, P)
n_evil = sum(1 for m in party if o._is_evil_name(m))
check("proposes exactly one evil", n_evil == 1)
check("party has correct size", len(party) == 3)
# of {sam, paul}, paul looks cleaner (0.2 < 0.3) -> paul chosen
check("prefers cleanest-looking evil (paul)", "paul" in party and "sam" not in party)

# If teammate paul is 'burned' (0.8) but self sam clean, with two evils it should
# skip the burned one and still include the other evil (sam).
P2 = probs(sam=0.3, paul=0.85, luca=0.6, jane=0.1, kira=0.15, mia=0.25)
o2 = make(quest=2)
party2 = o2._evil_propose_party(3, P2)
check("skips burned evil, includes cleaner evil (sam)", "sam" in party2 and "paul" not in party2)
# If both evils are above EVIL_BURNED_THRESHOLD, still include the least-burned
# evil. The threshold documents exposure; it must not create an all-good party.
P3 = probs(sam=0.75, paul=0.85, luca=0.6, jane=0.1, kira=0.15, mia=0.25)
o3 = make(quest=2)
party3 = o3._evil_propose_party(3, P3)
check("both evils burned: still includes least-burned evil", "sam" in party3 and "paul" not in party3)

print("== vote_for_party (evil) ==")
# Team containing a teammate -> approve
o = make(quest=2)
check("approves team containing teammate", o.vote_for_party(["paul", "jane", "kira"], P) is True)
# All-good team, safe (early, quest>=2, low failed) -> reject
o = make(quest=2, failed=0)
check("rejects clean team when safe", o.vote_for_party(["jane", "kira", "mia"], P) is False)
# Under rejection pressure -> approve even a clean team
o = make(quest=2, failed=3)
check("approves under reject pressure (failed>=3)", o.vote_for_party(["jane", "kira", "mia"], P) is True)
# evil_wins==2: never let a clean team run
o = make(quest=4, failed=0, evil_wins=2)
check("blocks clean team when one fail from win", o.vote_for_party(["jane", "kira", "mia"], P) is False)
# evil_wins==2 should override the cover-preserving failed>=3 approval.
o = make(quest=4, failed=3, evil_wins=2)
check("blocks clean team near reject limit when one fail from win", o.vote_for_party(["jane", "kira", "mia"], P) is False)

print("== vote_for_quest (evil) ==")
# Quest 1 -> pass to build trust (record party first)
o = make(quest=1)
o.vote_for_party(["sam", "jane"], P)
check("passes quest 1 (build trust)", o.vote_for_quest() is True)
# Mid-game, self on quest, sole evil -> fail
o = make(quest=2)
o.vote_for_party(["sam", "jane", "kira"], P)
check("fails mid-game when sole evil on quest", o.vote_for_quest() is False)
# Two evils on quest: only min-name fails. self=sam, teammate=paul -> paul < sam, so sam passes.
o = make(self_name="sam", teammates=("paul",), quest=3)
o.vote_for_party(["sam", "paul", "jane", "kira"], P)
check("two evils: non-designated (sam) passes", o.vote_for_quest() is True)
# Same but self is the min name: self=paul
o = make(self_name="paul", teammates=("sam",), quest=3)
o.vote_for_party(["sam", "paul", "jane", "kira"], P)
check("two evils: designated (paul) fails", o.vote_for_quest() is False)
# good_wins==2 -> must fail now
o = make(quest=4, good_wins=2)
o.vote_for_party(["sam", "jane", "kira"], P)
check("fails when good one win away", o.vote_for_quest() is False)
# evil_wins==2 even on quest 1 -> fail to win
o = make(quest=1, evil_wins=2)
o.vote_for_party(["sam", "jane"], P)
check("fails when evil one fail from win (overrides quest-1 trust)", o.vote_for_quest() is False)

print("== casing ==")
# named_knowledge gives capitalized names; set_identity must lowercase-match.
o = HeuristicOracle(ATEAM.EVIL)
o.set_identity("Sam", {"Paul"})
check("matches capitalized teammate name", o._is_evil_name("paul") and o._is_evil_name("Paul"))
check("matches capitalized self name", o._is_evil_name("SAM"))

print("== good path unchanged (sanity) ==")
g = HeuristicOracle(ATEAM.GOOD)
g.set_identity("sam", set())
g.update_context(0, 2, 0, 0)
check("good passes quest", g.vote_for_quest() is True)
check("good rejects party with likely-evil member",
      g.vote_for_party(["luca", "jane", "kira"], P) is False)
gp = g.propose_party(3, P)
check("good proposes top-good players (no luca)", "luca" not in gp and len(gp) == 3)

print(f"\n{passed} passed, {len(failed_cases)} failed")
if failed_cases:
    print("FAILED:", failed_cases)
    raise SystemExit(1)
print("All evil-policy checks passed.")
