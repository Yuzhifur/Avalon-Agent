"""Unified event schema for Avalon game records (phase 3, Workstream 1).

One `GameRecord` per game, independent of the source (avalonlogs scrape,
ProAvalon dump, or our self-play server logs). All downstream consumers — the
card featurizer, both model tracks, and the offline replay harness — read only
this schema, so the three parsers are the only places where source formats
matter.

Seats are 0-based indices into `seats` (the per-game player-name order).
Events are plain dicts (JSONL-friendly) in chronological order:

  ProposalEvent {type: "proposal", quest, round_idx, leader_seat, team_seats}
  VoteEvent     {type: "vote", quest, round_idx, approves[6], accepted, forced}
  QuestEvent    {type: "quest", quest, success}

`round_idx` is the 1-based proposal counter within the quest (resets on
acceptance — under standard rules it never exceeds 5). A VoteEvent always
follows its ProposalEvent. `forced` marks the forced-fifth hammer variant
(Workstream 5), where the fifth proposal skips the party vote; such votes are
recorded as unanimous approval with forced=True. QuestEvents appear only for
resolved quests.

Serialization is JSON Lines, one game per line — required for the ~101k-game
ProAvalon corpus. Always pass encoding="utf-8" (Windows defaults to a locale
codec).
"""

from dataclasses import dataclass, field, asdict
import json

NUM_PLAYERS = 6
NUM_QUESTS = 5

SOURCES = ("avalonlogs", "proavalon", "selfplay")
WINNERS = ("good", "evil")
# Normalized end-of-game reasons. "hammer" = five rejected proposals in a row
# (Evil auto-win under the standard rule).
WIN_REASONS = (
    "three_successes",
    "three_fails",
    "hammer",
    "assassination",
    "unknown",
)


def proposal_event(quest, round_idx, leader_seat, team_seats):
    return {
        "type": "proposal",
        "quest": int(quest),
        "round_idx": int(round_idx),
        "leader_seat": None if leader_seat is None else int(leader_seat),
        "team_seats": sorted(int(s) for s in team_seats),
    }


def vote_event(quest, round_idx, approves, accepted, forced=False):
    approves = [bool(a) for a in approves]
    if len(approves) != NUM_PLAYERS:
        raise ValueError(f"approves must have {NUM_PLAYERS} entries")
    return {
        "type": "vote",
        "quest": int(quest),
        "round_idx": int(round_idx),
        "approves": approves,
        "accepted": bool(accepted),
        "forced": bool(forced),
    }


def quest_event(quest, success):
    return {
        "type": "quest",
        "quest": int(quest),
        "success": bool(success),
    }


@dataclass
class GameRecord:
    game_id: str
    source: str                      # one of SOURCES
    seats: list                      # 6 player names, seat order
    evil: list                       # 6 ints, 1 = evil
    winner: str                      # "good" | "evil" | None (unfinished)
    win_reason: str                  # one of WIN_REASONS
    variant_roles: list = field(default_factory=list)  # e.g. ["MERLIN", ...]
    events: list = field(default_factory=list)

    def validate(self):
        if self.source not in SOURCES:
            raise ValueError(f"{self.game_id}: bad source {self.source!r}")
        if len(self.seats) != NUM_PLAYERS or len(self.evil) != NUM_PLAYERS:
            raise ValueError(f"{self.game_id}: needs exactly {NUM_PLAYERS} seats")
        if sum(self.evil) != 2:
            raise ValueError(f"{self.game_id}: expected exactly 2 evil, got {sum(self.evil)}")
        if self.winner is not None and self.winner not in WINNERS:
            raise ValueError(f"{self.game_id}: bad winner {self.winner!r}")
        if self.win_reason not in WIN_REASONS:
            raise ValueError(f"{self.game_id}: bad win_reason {self.win_reason!r}")
        pending = None
        for ev in self.events:
            kind = ev.get("type")
            if kind == "proposal":
                if pending is not None:
                    raise ValueError(f"{self.game_id}: proposal without a vote before it")
                pending = (ev["quest"], ev["round_idx"])
            elif kind == "vote":
                if pending != (ev["quest"], ev["round_idx"]):
                    raise ValueError(
                        f"{self.game_id}: vote {ev['quest']}.{ev['round_idx']} "
                        f"does not follow its proposal (pending={pending})")
                pending = None
            elif kind == "quest":
                if pending is not None:
                    raise ValueError(f"{self.game_id}: quest event amid open proposal")
            else:
                raise ValueError(f"{self.game_id}: unknown event type {kind!r}")
        if pending is not None:
            raise ValueError(f"{self.game_id}: trailing unvoted proposal")
        return self

    def to_dict(self):
        return asdict(self)

    def to_json_line(self):
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, d):
        return cls(
            game_id=d["game_id"],
            source=d["source"],
            seats=list(d["seats"]),
            evil=[int(x) for x in d["evil"]],
            winner=d.get("winner"),
            win_reason=d.get("win_reason", "unknown"),
            variant_roles=list(d.get("variant_roles", [])),
            events=list(d.get("events", [])),
        )

    @classmethod
    def from_json_line(cls, line):
        return cls.from_dict(json.loads(line))


def read_jsonl(path):
    """Yield GameRecords from a JSONL corpus file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield GameRecord.from_json_line(line)


def write_jsonl(records, path):
    """Write an iterable of GameRecords to a JSONL file. Returns the count."""
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json_line())
            fh.write("\n")
            count += 1
    return count
