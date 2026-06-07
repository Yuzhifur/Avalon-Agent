"""Runtime consistency checks for GRAIL chat messages."""

import re


class MessageConsistencyChecker:
    REJECT_CUES = (
        "vote no",
        "voting no",
        "reject",
        "not approving",
        "won't approve",
        "dont approve",
        "don't approve",
    )
    APPROVE_CUES = (
        "vote yes",
        "voting yes",
        "approve",
        "looks good to me",
        "fine with this team",
    )

    def repair(
        self,
        message,
        *,
        mode,
        party=None,
        planned_vote=None,
        quest_history=None,
        public_stances=None,
    ):
        original = (message or "").strip()
        repaired = original
        conflicts = []
        party = [name.lower() for name in (party or [])]
        quest_history = list(quest_history or [])
        public_stances = public_stances or {}

        repaired, history_conflicts = self._repair_quest_facts(repaired, quest_history)
        conflicts.extend(history_conflicts)

        if mode == "proposal" and party:
            missing = [name for name in party if name not in repaired.lower()]
            if missing:
                conflicts.append(f"proposal message omitted party members: {missing}")
                names = ", ".join(name.capitalize() for name in party)
                repaired = f"{repaired.rstrip('.')} i'm sticking with {names}.".strip()

        if mode == "party_reaction" and planned_vote is not None:
            lower = repaired.lower()
            contradicts = (
                planned_vote and any(cue in lower for cue in self.REJECT_CUES)
            ) or (
                not planned_vote and any(cue in lower for cue in self.APPROVE_CUES)
            )
            if contradicts:
                conflicts.append("message contradicted the planned party vote")
                names = ", ".join(name.capitalize() for name in party)
                if planned_vote:
                    repaired = (
                        f"i'm okay approving {names} for now based on the public quest "
                        "and vote history."
                    )
                else:
                    repaired = (
                        f"i'm not comfortable approving {names} yet based on the public "
                        "quest and vote history."
                    )

        stance_conflict = self._find_stance_conflict(repaired, public_stances)
        if stance_conflict:
            conflicts.append(stance_conflict)
            if mode == "party_reaction" and planned_vote is not None:
                names = ", ".join(name.capitalize() for name in party)
                action = "approve" if planned_vote else "reject"
                repaired = (
                    f"i'm leaning to {action} {names} based on the proposal and quest "
                    "history, but i'm still keeping my reads flexible."
                )

        return {
            "message": repaired,
            "changed": repaired != original,
            "conflicts": conflicts,
        }

    def _repair_quest_facts(self, message, quest_history):
        outcomes = {
            index + 1: str(item[1]).lower()
            for index, item in enumerate(quest_history)
            if len(item) >= 2
        }
        conflicts = []

        def replace(match):
            quest = int(match.group(1))
            claimed = match.group(2).lower()
            actual = outcomes.get(quest)
            if actual not in ("success", "fail", "failed", "succeeded"):
                return match.group(0)
            normalized_actual = "failed" if actual in ("fail", "failed") else "succeeded"
            normalized_claimed = "failed" if claimed in ("fail", "failed") else "succeeded"
            if normalized_actual == normalized_claimed:
                return match.group(0)
            conflicts.append(
                f"corrected Quest {quest} outcome from {normalized_claimed} to {normalized_actual}"
            )
            return f"quest {quest} {normalized_actual}"

        repaired = re.sub(
            r"\bquest\s+(\d+)\s+(failed|fail|succeeded|success)\b",
            replace,
            message,
            flags=re.IGNORECASE,
        )
        return repaired, conflicts

    def _find_stance_conflict(self, message, public_stances):
        lower = message.lower()
        for name, stance in public_stances.items():
            if name.lower() not in lower:
                continue
            if stance == "suspect" and any(cue in lower for cue in self.APPROVE_CUES):
                return f"message strongly endorsed publicly suspected player {name}"
        return None
