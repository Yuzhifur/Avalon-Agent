"""Configurable action policies layered on top of GRAIL beliefs.

The factor graph remains responsible for role probabilities. This module turns
those probabilities plus mechanical game history into actions. HeuristicOracle
keeps the original public API while delegating to separate Good and Evil
policies.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os

from agent_base import ATEAM


DEFAULT_POLICY_CONFIG = {
    "common": {
        "final_proposal_rejects": 4,
    },
    "good": {
        "include_self_in_proposals": True,
        "reject_thresholds": {
            "1": 0.65,
            "2": 0.60,
            "3": 0.60,
            "4": 0.57,
            "5": 0.55,
        },
        "good_match_point_penalty": 0.08,
        "evil_match_point_penalty": 0.05,
        "failed_quest_member_penalty": 0.12,
        "approved_failed_quest_penalty": 0.06,
        "proposed_failed_quest_penalty": 0.08,
        "rejected_successful_quest_penalty": 0.04,
        "rejected_proposal_leader_penalty": 0.02,
        "approved_rejected_proposal_penalty": 0.01,
    },
    "evil": {
        "pass_first_quest": True,
        "burned_threshold": 0.70,
        "clean_team_reject_through_quest": 3,
        "clean_team_cover_approve_rate": 0.25,
        "teammate_distance_rate": 0.35,
        "teammate_distance_threshold": 0.65,
        "teammate_soft_defense_threshold": 0.30,
        "evil_vote_split_rate": 0.25,
        "midgame_fail_rate": 1.0,
    },
}

# Backward-compatible constants used by the Phase 1 test driver.
PASS_FIRST_QUEST = DEFAULT_POLICY_CONFIG["evil"]["pass_first_quest"]
EVIL_BURNED_THRESHOLD = DEFAULT_POLICY_CONFIG["evil"]["burned_threshold"]


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_policy_config(config=None, role=None):
    """Merge defaults, agent JSON config, and optional experiment overrides."""
    merged = _deep_merge(DEFAULT_POLICY_CONFIG, config or {})
    raw_override = os.environ.get("GRAIL_POLICY_OVERRIDES", "").strip()
    if raw_override:
        try:
            merged = _deep_merge(merged, json.loads(raw_override))
        except json.JSONDecodeError as exc:
            raise ValueError("GRAIL_POLICY_OVERRIDES must be valid JSON") from exc
    side = "EVIL" if role == ATEAM.EVIL else "GOOD"
    raw_side_override = os.environ.get(f"GRAIL_POLICY_OVERRIDES_{side}", "").strip()
    if raw_side_override:
        try:
            merged = _deep_merge(merged, json.loads(raw_side_override))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GRAIL_POLICY_OVERRIDES_{side} must be valid JSON"
            ) from exc
    return merged


@dataclass
class PolicyContext:
    failed_party_votes: int = 0
    quest_number: int = 1
    good_wins: int = 0
    evil_wins: int = 0
    party_size: int = 0
    leader: str = ""
    proposal_history: list = field(default_factory=list)
    quest_history: list = field(default_factory=list)


class EvilSocialState:
    """Compact, persistent public cover plan for an Evil agent."""

    SUSPICION_CUES = ("sus", "don't trust", "do not trust", "worried", "risky", "not sold")
    TRUST_CUES = ("trust", "looks good", "seems good", "comfortable with", "fine with")

    def __init__(self, self_name="", evil_teammates=None):
        self.self_name = (self_name or "").lower()
        self.evil_teammates = {n.lower() for n in (evil_teammates or set())}
        self.stances = {}
        self.accused = set()
        self.defended = set()
        self.fake_evil_team = []
        self.last_vote_reason = ""
        self.action_history = []

    def set_identity(self, self_name, evil_teammates):
        self.self_name = (self_name or "").lower()
        self.evil_teammates = {n.lower() for n in (evil_teammates or set())}

    def refresh(
        self,
        probabilities,
        behavior_risk,
        distance_threshold,
        soft_defense_threshold,
    ):
        if not probabilities:
            return
        evil_set = set(self.evil_teammates)
        if self.self_name:
            evil_set.add(self.self_name)

        public_candidates = [name for name in probabilities if name.lower() not in evil_set]
        ranked = sorted(
            public_candidates,
            key=lambda name: behavior_risk.get(name.lower(), probabilities[name]["evil"]),
            reverse=True,
        )
        self.fake_evil_team = [name.lower() for name in ranked[:2]]

        for name in probabilities:
            lowered = name.lower()
            risk = behavior_risk.get(lowered, probabilities[name]["evil"])
            if lowered in self.fake_evil_team:
                self.stances[lowered] = "suspect"
            elif lowered in self.evil_teammates:
                if probabilities[name]["evil"] >= distance_threshold:
                    self.stances[lowered] = "cautious"
                elif probabilities[name]["evil"] <= soft_defense_threshold:
                    self.stances[lowered] = "trust"
                else:
                    self.stances[lowered] = "neutral"
            elif risk <= 0.30:
                self.stances[lowered] = "trust"
            else:
                self.stances[lowered] = "neutral"

    def record_action(self, action, selected, reason):
        self.action_history.append(
            {"action": action, "selected": selected, "reason": reason}
        )
        self.action_history = self.action_history[-12:]
        if action == "vote_party":
            self.last_vote_reason = reason

    def record_message(self, message, player_names):
        text = (message or "").lower()
        for name in player_names:
            lowered = name.lower()
            position = text.find(lowered)
            if position < 0:
                continue
            window = text[max(0, position - 45): position + len(lowered) + 45]
            if any(cue in window for cue in self.SUSPICION_CUES):
                self.accused.add(lowered)
                self.stances[lowered] = "suspect"
            if any(cue in window for cue in self.TRUST_CUES):
                self.defended.add(lowered)
                if self.stances.get(lowered) != "suspect":
                    self.stances[lowered] = "trust"

    def prompt_summary(self):
        suspects = ", ".join(name.capitalize() for name in self.fake_evil_team) or "no one strongly"
        cautious = [
            name.capitalize()
            for name, stance in self.stances.items()
            if stance == "cautious"
        ]
        cautious_text = ", ".join(cautious) if cautious else "no specific player"
        trusted = [
            name.capitalize()
            for name, stance in self.stances.items()
            if stance == "trust" and name in self.evil_teammates
        ]
        support_text = (
            f"If useful, give only light, evidence-based support to {', '.join(trusted)}. "
            if trusted
            else "Do not make a point of defending any specific player. "
        )
        return (
            f"Your current public story is that {suspects} deserve the most scrutiny. "
            f"Be cautious rather than protective around {cautious_text}. "
            f"{support_text}"
            "Keep vote explanations tied to public quest and proposal history, and avoid "
            "forming an obvious voting bloc."
        )


class BaseSidePolicy:
    def __init__(self, config, context, owner):
        self.config = config
        self.context = context
        self.owner = owner

    @staticmethod
    def order_players(probabilities):
        return sorted(
            probabilities.keys(),
            key=lambda name: probabilities[name]["good"],
            reverse=True,
        )

    def _behavior_risk(self, probabilities):
        """Return adjusted public risk using only mechanical history."""
        risk = {
            name.lower(): float(values["evil"])
            for name, values in (probabilities or {}).items()
        }
        good_config = self.owner.config["good"]

        for record in self.context.proposal_history:
            party = [name.lower() for name in record.get("party", [])]
            votes = {
                name.lower(): bool(vote)
                for name, vote in record.get("votes", {}).items()
            }
            proposer = str(record.get("leader") or "").lower()
            outcome = record.get("quest_outcome")

            if outcome is False:
                for name in party:
                    risk[name] = risk.get(name, 0.5) + good_config["failed_quest_member_penalty"]
                for name, approved in votes.items():
                    if approved:
                        risk[name] = risk.get(name, 0.5) + good_config["approved_failed_quest_penalty"]
                if proposer:
                    risk[proposer] = risk.get(proposer, 0.5) + good_config["proposed_failed_quest_penalty"]
            elif outcome is True:
                for name, approved in votes.items():
                    if not approved:
                        risk[name] = risk.get(name, 0.5) + good_config["rejected_successful_quest_penalty"]
            elif record.get("accepted") is False:
                if proposer:
                    risk[proposer] = (
                        risk.get(proposer, 0.5)
                        + good_config["rejected_proposal_leader_penalty"]
                    )
                for name, approved in votes.items():
                    if approved:
                        risk[name] = (
                            risk.get(name, 0.5)
                            + good_config["approved_rejected_proposal_penalty"]
                        )

        return {name: max(0.0, min(1.0, value)) for name, value in risk.items()}

    def _stable_choice(self, label, rate):
        if rate <= 0:
            return False
        if rate >= 1:
            return True
        seed = (
            f"{self.owner.self_name}|{label}|{self.context.quest_number}|"
            f"{self.context.failed_party_votes}|{len(self.context.proposal_history)}"
        )
        bucket = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < rate

    def _shared_choice(self, label, rate):
        if rate <= 0:
            return False
        if rate >= 1:
            return True
        seed = (
            f"{label}|{self.context.quest_number}|{self.context.failed_party_votes}|"
            f"{len(self.context.proposal_history)}"
        )
        bucket = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < rate

    def _record(self, action, selected, alternatives, reason, scores=None):
        self.owner.last_decision = {
            "action": action,
            "selected": selected,
            "alternatives": alternatives,
            "reason": reason,
            "scores": scores or {},
        }


class GoodPolicy(BaseSidePolicy):
    def _threshold(self):
        thresholds = self.config["reject_thresholds"]
        threshold = float(
            thresholds.get(str(self.context.quest_number), thresholds.get("default", 0.60))
        )
        if self.context.good_wins >= 2:
            threshold -= self.config["good_match_point_penalty"]
        if self.context.evil_wins >= 2:
            threshold -= self.config["evil_match_point_penalty"]
        return max(0.40, min(0.90, threshold))

    def opinion_on_party(self, party, probabilities):
        risk = self._behavior_risk(probabilities)
        return all(risk.get(name.lower(), 0.5) < self._threshold() for name in party)

    def vote_for_party(self, party, probabilities, record=True):
        final_rejects = self.owner.config["common"]["final_proposal_rejects"]
        risk = self._behavior_risk(probabilities)
        threshold = self._threshold()

        if self.context.failed_party_votes >= final_rejects:
            vote = True
            reason = "approve the final proposal to avoid an automatic Evil win"
        else:
            risky = [name for name in party if risk.get(name.lower(), 0.5) >= threshold]
            vote = not risky
            reason = (
                "all party members are below the current risk threshold"
                if vote
                else f"party exceeds the risk threshold through {', '.join(risky)}"
            )

        if record:
            self._record(
                "vote_party",
                vote,
                [True, False],
                reason,
                {"threshold": threshold, "party_risk": {n: risk.get(n.lower(), 0.5) for n in party}},
            )
        return vote

    def propose_party(self, party_size, probabilities):
        risk = self._behavior_risk(probabilities)
        ordered = sorted(probabilities, key=lambda name: (risk.get(name.lower(), 0.5), name))
        party = []

        if (
            self.config["include_self_in_proposals"]
            and self.owner.self_name
            and self.owner.self_name in {name.lower() for name in ordered}
        ):
            self_key = next(name for name in ordered if name.lower() == self.owner.self_name)
            party.append(self_key)

        for name in ordered:
            if len(party) >= party_size:
                break
            if name not in party:
                party.append(name)

        self._record(
            "propose_party",
            party[:party_size],
            ordered[: min(len(ordered), party_size + 2)],
            "select the lowest mechanically adjusted risk players",
            {"player_risk": risk},
        )
        return party[:party_size]

    def vote_for_quest(self):
        self._record("vote_quest", True, [True], "Good players must pass quests")
        return True


class EvilPolicy(BaseSidePolicy):
    def __init__(self, config, context, owner):
        super().__init__(config, context, owner)
        self.social_state = EvilSocialState()

    def set_identity(self, self_name, evil_teammates):
        self.social_state.set_identity(self_name, evil_teammates)

    def _refresh_social_state(self, probabilities):
        risk = self._behavior_risk(probabilities)
        self.social_state.refresh(
            probabilities,
            risk,
            self.config["teammate_distance_threshold"],
            self.config["teammate_soft_defense_threshold"],
        )
        return risk

    def opinion_on_party(self, party, probabilities):
        return any(self.owner._is_evil_name(name) for name in party)

    def vote_for_party(self, party, probabilities, record=True):
        risk = self._refresh_social_state(probabilities)
        final_rejects = self.owner.config["common"]["final_proposal_rejects"]
        evil_members = [name for name in party if self.owner._is_evil_name(name)]
        clean_team = not evil_members
        direct_pressure = self.context.good_wins >= 2 or self.context.evil_wins >= 2

        if self.context.failed_party_votes >= final_rejects:
            vote = False
            reason = "rejecting the fifth proposal wins immediately for Evil"
        elif clean_team and self.context.good_wins >= 2:
            vote = False
            reason = "block a clean team when Good is one success from winning"
        elif clean_team and self.context.evil_wins >= 2:
            vote = False
            reason = "seek a team containing Evil to secure the final failed quest"
        elif clean_team:
            cover_approve = self._stable_choice(
                "approve-clean-team",
                self.config["clean_team_cover_approve_rate"],
            )
            early_reject = (
                self.context.quest_number <= self.config["clean_team_reject_through_quest"]
                and self.context.failed_party_votes <= 1
            )
            vote = cover_approve or not early_reject
            reason = (
                "approve a clean team occasionally to preserve public cover"
                if vote
                else "reject a clean team while proposal pressure is low"
            )
        else:
            teammate_only = all(name.lower() != self.owner.self_name for name in evil_members)
            probabilities_by_name = {
                name.lower(): values for name, values in probabilities.items()
            }
            exposed_teammate = any(
                probabilities_by_name.get(name.lower(), {"evil": 0.5})["evil"]
                >= self.config["teammate_distance_threshold"]
                for name in evil_members
            )
            distance = (
                teammate_only
                and exposed_teammate
                and not direct_pressure
                and self._stable_choice("distance-teammate", self.config["teammate_distance_rate"])
            )
            split_vote = (
                len(evil_members) >= 2
                and not direct_pressure
                and self._shared_choice("split-evil-votes", self.config["evil_vote_split_rate"])
            )
            designated_rejector = max(name.lower() for name in evil_members)
            should_split_reject = split_vote and self.owner.self_name == designated_rejector
            vote = not (distance or should_split_reject)
            if distance:
                reason = "distance from an exposed teammate while the score is safe"
            elif should_split_reject:
                reason = "split the Evil voting bloc while preserving a fail opportunity"
            else:
                reason = "approve a team that preserves an Evil fail opportunity"

        if record:
            self._record(
                "vote_party",
                vote,
                [True, False],
                reason,
                {"party_risk": {n: risk.get(n.lower(), 0.5) for n in party}},
            )
            self.social_state.record_action("vote_party", vote, reason)
        return vote

    def propose_party(self, party_size, probabilities):
        risk = self._refresh_social_state(probabilities)
        ordered = sorted(probabilities, key=lambda name: (risk.get(name.lower(), 0.5), name))
        evils = [name for name in ordered if self.owner._is_evil_name(name)]
        goods = [name for name in ordered if not self.owner._is_evil_name(name)]

        party = []
        if evils:
            candidate = min(evils, key=lambda name: probabilities[name]["evil"])
            party.append(candidate)

        for name in goods:
            if len(party) >= party_size:
                break
            party.append(name)
        for name in ordered:
            if len(party) >= party_size:
                break
            if name not in party:
                party.append(name)

        selected = party[:party_size]
        reason = "include the least-exposed Evil player with credible Good-looking cover"
        self._record(
            "propose_party",
            selected,
            ordered[: min(len(ordered), party_size + 2)],
            reason,
            {"player_risk": risk},
        )
        self.social_state.record_action("propose_party", selected, reason)
        return selected

    def vote_for_quest(self):
        evils_on_quest = [
            name for name in self.owner._last_party if self.owner._is_evil_name(name)
        ]
        if len(evils_on_quest) >= 2:
            designated = min(name.lower() for name in evils_on_quest)
            if self.owner.self_name != designated:
                vote = True
                reason = "coordinate so the designated Evil teammate casts the only fail"
                self._record("vote_quest", vote, [True, False], reason)
                self.social_state.record_action("vote_quest", vote, reason)
                return vote

        if self.context.evil_wins >= 2:
            vote = False
            reason = "fail to secure the third failed quest"
        elif self.context.good_wins >= 2:
            vote = False
            reason = "fail because Good is one success from winning"
        elif self.context.quest_number == 1 and self.config["pass_first_quest"]:
            vote = True
            reason = "pass Quest 1 to build cover"
        else:
            should_fail = self._stable_choice(
                "midgame-fail",
                self.config["midgame_fail_rate"],
            )
            vote = not should_fail
            reason = (
                "fail the midgame quest while an Evil player is present"
                if should_fail
                else "pass this quest to preserve cover"
            )

        self._record("vote_quest", vote, [True, False], reason)
        self.social_state.record_action("vote_quest", vote, reason)
        return vote


class HeuristicOracle:
    """Backward-compatible policy facade used by ACLAgent."""

    def __init__(self, role, policy_config=None):
        self.role = role
        self.config = load_policy_config(policy_config, role=role)
        self.context = PolicyContext()
        self.self_name = None
        self.evil_teammates = set()
        self._last_party = []
        self.last_decision = {}

        self.good_policy = GoodPolicy(self.config["good"], self.context, self)
        self.evil_policy = EvilPolicy(self.config["evil"], self.context, self)

    @property
    def failed_party_votes(self):
        return self.context.failed_party_votes

    @property
    def quest_number(self):
        return self.context.quest_number

    @property
    def good_wins(self):
        return self.context.good_wins

    @property
    def evil_wins(self):
        return self.context.evil_wins

    def set_identity(self, self_name, evil_teammates):
        self.self_name = (self_name or "").lower()
        self.evil_teammates = {name.lower() for name in (evil_teammates or set())}
        self.evil_policy.set_identity(self.self_name, self.evil_teammates)

    def update_context(
        self,
        failed_party_votes,
        quest_number,
        good_wins,
        evil_wins,
        party_size=0,
        leader=None,
        proposal_history=None,
        quest_history=None,
    ):
        self.context.failed_party_votes = failed_party_votes or 0
        self.context.quest_number = quest_number or 1
        self.context.good_wins = good_wins or 0
        self.context.evil_wins = evil_wins or 0
        self.context.party_size = party_size or 0
        self.context.leader = (leader or "").lower()
        self.context.proposal_history = list(proposal_history or [])
        self.context.quest_history = list(quest_history or [])

    def _active_policy(self):
        return self.evil_policy if self.role == ATEAM.EVIL else self.good_policy

    def _evil_set(self):
        names = set(self.evil_teammates)
        if self.self_name:
            names.add(self.self_name)
        return names

    def _is_evil_name(self, name):
        return name.lower() in self._evil_set()

    def get_action(self, state):
        return {}

    def opinion_on_party(self, party, probabilities):
        return self._active_policy().opinion_on_party(party, probabilities)

    def vote_for_party(self, party, probabilities):
        self._last_party = list(party)
        return self._active_policy().vote_for_party(party, probabilities, record=True)

    def preview_vote_for_party(self, party, probabilities):
        return self._active_policy().vote_for_party(party, probabilities, record=False)

    def propose_party(self, party_size, probabilities):
        return self._active_policy().propose_party(party_size, probabilities)

    # Backward-compatible helpers used by Phase 1 tests.
    def _evil_vote_for_party(self, party, probabilities):
        self._last_party = list(party)
        return self.evil_policy.vote_for_party(party, probabilities, record=True)

    def _evil_propose_party(self, party_size, probabilities):
        return self.evil_policy.propose_party(party_size, probabilities)

    def vote_for_quest(self):
        return self._active_policy().vote_for_quest()

    def chose_assassin_target(self, probabilities):
        return self.order_players(probabilities)[0]

    def order_players(self, probabilities):
        return self._active_policy().order_players(probabilities)

    def public_plan_summary(self):
        if self.role != ATEAM.EVIL:
            return ""
        return self.evil_policy.social_state.prompt_summary()

    def public_stances(self):
        if self.role != ATEAM.EVIL:
            return {}
        return dict(self.evil_policy.social_state.stances)

    def record_public_message(self, message, player_names):
        if self.role == ATEAM.EVIL:
            self.evil_policy.social_state.record_message(message, player_names)

    def decision_log_payload(self):
        return deepcopy(self.last_decision)
