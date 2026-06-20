"""Self-play server log JSON -> GameRecord (phase 3, Workstream 1).

Input is the Colyseus server log written by our stack (one JSON per game,
e.g. policy run directories' server/<GAMEID>.json). The authoritative state
is the last message with msgtype == "game"; its `full` payload carries
`all_players` (id/name/role) and the ordered system `messages` stream, whose
formats are shared with code/evaluation/log_analyzer.py and ACLAgent:

  "<Leader> proposed a party: A, B, C"
  "Party vote summary: Name: yes, Name: no, ..."
  "The party has been approved!" / "The party has been rejected!"
  "The quest has succeeded!" / "The quest has failed!"
  endings: "Evil wins by failing three quests!",
           "Good wins by succeeding three quests!",
           "Evil wins by rejecting five parties!",
           "Evil wins by assassinating Merlin!",
           "Good wins as Evil assassinated the wrong Merlin!"

Under the forced-fifth hammer variant (Workstream 5) the server emits
FORCED_FIFTH_MARKER followed by a unanimous vote summary and the standard
approval message; the resulting VoteEvent gets forced=True.

CLI:  python parse_selfplay.py <runs-root|server.json> [--out corpus.jsonl]
      runs-root is searched recursively for */server/*.json
"""

import argparse
import glob
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from event_schema import (
        GameRecord, NUM_PLAYERS, proposal_event, quest_event, vote_event,
        write_jsonl,
    )
else:
    from .event_schema import (
        GameRecord, NUM_PLAYERS, proposal_event, quest_event, vote_event,
        write_jsonl,
    )

# Single source of truth for the forced-fifth marker message (Workstream 5).
# AvalonGame.ts emits exactly this string; GameInfo and this parser match it.
FORCED_FIFTH_MARKER = "The fifth proposal is forced through!"

EVIL_ROLE_PREFIXES = ("minion", "morgana", "assassin", "mordred", "oberon", "evil")

WIN_REASON_BY_ENDING = {
    "Evil wins by failing three quests!": "three_fails",
    "Good wins by succeeding three quests!": "three_successes",
    "Evil wins by rejecting five parties!": "hammer",
    "Evil wins by assassinating Merlin!": "assassination",
    "Good wins as Evil assassinated the wrong Merlin!": "three_successes",
}


class SkipGame(Exception):
    pass


def _is_evil_role(role):
    return str(role).lower().startswith(EVIL_ROLE_PREFIXES)


def load_full_state(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    games = [m for m in data.get("logs", []) if m.get("msgtype") == "game"]
    if not games:
        raise SkipGame("no game message in log")
    return games[-1]["full"]


def parse_full_state(full, game_id):
    players = sorted(full.get("all_players", []), key=lambda p: p["id"])
    if len(players) != NUM_PLAYERS:
        raise SkipGame(f"{len(players)} players")
    seats = [p["name"] for p in players]
    seat_of = {name.lower(): i for i, name in enumerate(seats)}
    evil = [1 if _is_evil_role(p.get("role", "")) else 0 for p in players]
    variant_roles = sorted(
        {p["role"].split("-")[0].upper() for p in players}
        - {"SERVANT", "MINION"}
    )

    winner = full.get("winner") or None
    if winner not in ("good", "evil"):
        winner = None

    events = []
    win_reason = "unknown"
    pending_team = None     # seats of the currently proposed party
    pending_leader = None
    pending_votes = None    # {seat: bool}
    pending_forced = False
    rounds_in_quest = {}    # quest -> voted proposal count

    for msg in full.get("messages", []):
        if msg.get("player") != "system":
            continue
        text = msg.get("msg", "")
        quest = msg.get("quest")

        if " proposed a party: " in text:
            leader_name, team_str = text.split(" proposed a party: ", 1)
            pending_leader = seat_of.get(leader_name.strip().lower())
            try:
                pending_team = [
                    seat_of[n.strip().lower()] for n in team_str.split(",")
                ]
            except KeyError:
                raise SkipGame(f"unknown player in proposal {text!r}")
            pending_votes = None
        elif text.startswith("Party vote summary:"):
            body = text.split("Party vote summary:", 1)[1].strip()
            votes = {}
            for part in body.split(", "):
                if ": " not in part:
                    continue
                name, vote = part.rsplit(": ", 1)
                seat = seat_of.get(name.strip().lower())
                if seat is None:
                    raise SkipGame(f"unknown voter in summary {text!r}")
                votes[seat] = vote.strip().lower() == "yes"
            pending_votes = votes
        elif text.startswith(FORCED_FIFTH_MARKER):
            pending_forced = True
        elif text.startswith("The party has been"):
            accepted = "approved" in text
            if pending_team is None or pending_votes is None:
                raise SkipGame(f"party result without proposal/votes at {text!r}")
            if len(pending_votes) != NUM_PLAYERS:
                raise SkipGame(f"vote summary covers {len(pending_votes)} players")
            rounds_in_quest[quest] = rounds_in_quest.get(quest, 0) + 1
            round_idx = rounds_in_quest[quest]
            approves = [pending_votes[s] for s in range(NUM_PLAYERS)]
            if not pending_forced and accepted != (sum(approves) > NUM_PLAYERS / 2):
                raise SkipGame(f"vote/result mismatch in quest {quest}")
            events.append(
                proposal_event(quest, round_idx, pending_leader, pending_team))
            events.append(
                vote_event(quest, round_idx, approves, accepted, pending_forced))
            pending_team = pending_leader = pending_votes = None
            pending_forced = False
        elif text.startswith("The quest has"):
            events.append(quest_event(quest, "succeeded" in text))
        elif text in WIN_REASON_BY_ENDING:
            win_reason = WIN_REASON_BY_ENDING[text]

    if winner is None:
        raise SkipGame("no winner (incomplete game)")

    record = GameRecord(
        game_id=game_id,
        source="selfplay",
        seats=seats,
        evil=evil,
        winner=winner,
        win_reason=win_reason,
        variant_roles=variant_roles,
        events=events,
    )
    try:
        record.validate()
    except ValueError as exc:
        raise SkipGame(str(exc))
    return record


def parse_file(path):
    game_id = os.path.splitext(os.path.basename(path))[0]
    return parse_full_state(load_full_state(path), game_id)


def find_server_logs(root):
    """All server game logs under a policy-run root directory."""
    pattern = os.path.join(root, "**", "server", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def parse_path(path, on_skip=None):
    if os.path.isdir(path):
        paths = find_server_logs(path)
    else:
        paths = [path]
    for p in paths:
        try:
            yield parse_file(p)
        except SkipGame as exc:
            if on_skip:
                on_skip(p, str(exc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="server log JSON or a runs root directory")
    ap.add_argument("--out", help="write GameRecords to this JSONL file")
    args = ap.parse_args(argv)

    skips = []
    records = list(parse_path(args.path, on_skip=lambda p, r: skips.append((p, r))))
    print(f"parsed {len(records)} games, skipped {len(skips)}")
    for p, reason in skips[:10]:
        print(f"  skip {os.path.basename(p)}: {reason}")
    if args.out:
        write_jsonl(records, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
