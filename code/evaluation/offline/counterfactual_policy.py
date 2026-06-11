"""Counterfactual GoodPolicy replay (Workstream 0).

Instantiates the real HeuristicOracle(GOOD) — the exact policy code the
agents run — against replayed beliefs, feeding it proposal history in the
same dict shape as GameInfo.policy_proposal_history(). Used for:

  - the phase-2 reproduction gate (recomputing behavior risk / thresholds
    that classify evil-team approvals), and
  - the counterfactual flip metric: would the policy, given a candidate
    model's beliefs, have rejected historically-approved evil teams (and at
    what false-rejection cost on clean teams)?

Penalty ablations map to the same GRAIL_POLICY_OVERRIDES_GOOD knobs the live
stack uses (Workstream 4), so offline winners carry over verbatim.
"""

import sys

from . import add_agent_paths

add_agent_paths()

from agent_enums import ATEAM                      # noqa: E402
from our.policy_models.heuristic import HeuristicOracle  # noqa: E402
from our.proposal_cards import (                   # noqa: E402
    cards_from_events, history_from_cards,
)

# Named penalty ablations (plan, Workstream 4). Values override the "good"
# section of DEFAULT_POLICY_CONFIG.
PENALTY_PRESETS = {
    "default": {},
    "off": {
        "good": {
            "failed_quest_member_penalty": 0.0,
            "approved_failed_quest_penalty": 0.0,
            "proposed_failed_quest_penalty": 0.0,
            "rejected_successful_quest_penalty": 0.0,
            "rejected_proposal_leader_penalty": 0.0,
            "approved_rejected_proposal_penalty": 0.0,
        }
    },
    "no_rejected": {
        "good": {
            "rejected_proposal_leader_penalty": 0.0,
            "approved_rejected_proposal_penalty": 0.0,
        }
    },
}


def iter_vote_points(record):
    """Yield one dict per party vote of a GameRecord, with everything the
    policy needs to be replayed at that decision point:

      proposal, vote: the raw events
      cards_before:   canonical cards of the state the voters saw
      fpv:            failed_party_votes at the vote (= round_idx - 1)
      good_wins/evil_wins: resolved quests so far
      quest, round_idx, team_seats, approves, accepted, forced
    """
    events = record.events
    for i, ev in enumerate(events):
        if ev.get("type") != "vote":
            continue
        proposal = events[i - 1]
        cards_before = cards_from_events(events[:i - 1])
        outcomes = [c["outcome"] for c in cards_before
                    if c["accepted"] and c["outcome"] is not None]
        yield {
            "proposal": proposal,
            "vote": ev,
            "cards_before": cards_before,
            "quest": ev["quest"],
            "round_idx": ev["round_idx"],
            "fpv": ev["round_idx"] - 1,
            "good_wins": sum(1 for o in outcomes if o),
            "evil_wins": sum(1 for o in outcomes if not o),
            "team_seats": list(proposal["team_seats"]),
            "leader_seat": proposal.get("leader_seat"),
            "approves": list(ev["approves"]),
            "accepted": ev["accepted"],
            "forced": ev.get("forced", False),
        }


class GoodPolicyReplayer:
    """The real GoodPolicy, driven by replayed context and beliefs."""

    def __init__(self, penalties="default", policy_config=None):
        if policy_config is None:
            policy_config = PENALTY_PRESETS[penalties]
        self.oracle = HeuristicOracle(ATEAM.GOOD, policy_config=policy_config)

    def decide(self, record, point, beliefs_by_seat, ego_seat):
        """Replay one party vote for one good ego.

        beliefs_by_seat: {seat: P(evil)} as the ego's belief state.
        Returns {vote, threshold, party_risk: {seat: risk}}.
        """
        seats = [str(n).lower() for n in record.seats]
        self.oracle.set_identity(seats[ego_seat], set())
        self.oracle.update_context(
            failed_party_votes=point["fpv"],
            quest_number=point["quest"],
            good_wins=point["good_wins"],
            evil_wins=point["evil_wins"],
            party_size=len(point["team_seats"]),
            leader=(None if point["leader_seat"] is None
                    else seats[point["leader_seat"]]),
            proposal_history=history_from_cards(point["cards_before"], seats),
            quest_history=[],
        )
        probabilities = {
            seats[s]: {"evil": float(p), "good": 1.0 - float(p)}
            for s, p in beliefs_by_seat.items()
        }
        party = [seats[s] for s in point["team_seats"]]
        vote = self.oracle.vote_for_party(party, probabilities)
        decision = self.oracle.decision_log_payload()
        scores = decision.get("scores", {})
        risk_by_seat = {
            seats.index(name): risk
            for name, risk in scores.get("party_risk", {}).items()
        }
        return {
            "vote": bool(vote),
            "threshold": scores.get("threshold"),
            "party_risk": risk_by_seat,
        }


def counterfactual_flip_metrics(decisions):
    """Summarize replayed-vs-historical good votes.

    decisions: list of {historical_approve, replay_approve, team_has_evil,
    forced_pressure (fpv >= 4)}. Reports the plan's two headline numbers:
    reject-flip rate on historically-approved evil teams, and new false
    rejections of clean teams — both excluding fpv >= 4 votes, where the
    policy auto-approves regardless of beliefs.
    """
    flip_base = [d for d in decisions
                 if d["historical_approve"] and d["team_has_evil"]
                 and not d["forced_pressure"]]
    flipped = [d for d in flip_base if not d["replay_approve"]]

    clean_base = [d for d in decisions
                  if d["historical_approve"] and not d["team_has_evil"]
                  and not d["forced_pressure"]]
    clean_flipped = [d for d in clean_base if not d["replay_approve"]]

    return {
        "approved_evil_teams": len(flip_base),
        "evil_team_reject_flip_rate": (
            len(flipped) / len(flip_base) if flip_base else None),
        "approved_clean_teams": len(clean_base),
        "clean_team_new_false_reject_rate": (
            len(clean_flipped) / len(clean_base) if clean_base else None),
    }
