"""avalonlogs scrape -> GameRecord (phase 3, Workstream 1).

Source format (one JSON file per game, e.g. E:/Local/avalonlogs_6p):

  players:  [{name, ...} x6]
  missions: [{state: SUCCESS|FAIL|PENDING, teamSize, failsRequired, numFails,
              proposals: [{proposer, team: [names], votes: [approving names],
                           state: APPROVED|REJECTED}]}]
  outcome:  {state: EVIL_WIN|GOOD_WIN, message,
             roles: [{name, role, assassin}], votes}

Every proposal (including rejected ones and those of the PENDING hammer
mission) becomes a ProposalEvent + VoteEvent — this is exactly the signal the
legacy vectorizer discarded. Quest results only exist for resolved missions.

Role vocabulary seen in the 3,317-game 6p extract: LOYAL FOLLOWER, EVIL
MINION, MERLIN, PERCIVAL, MORGANA, MORDRED, OBERON, ASSASSIN. The parser
records `variant_roles` (everything beyond the vanilla pair, plus ASSASSIN
when carried as a flag); corpus-level filtering (e.g. dropping MORDRED/OBERON
games per plan risk #2) happens in build_corpus.py, not here.

CLI:  python parse_avalonlogs.py <dir-or-file> [--out corpus.jsonl]
"""

import argparse
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

GOOD_ROLES = {"LOYAL FOLLOWER", "MERLIN", "PERCIVAL", "SERVANT"}
VANILLA_ROLES = {"LOYAL FOLLOWER", "EVIL MINION"}

WIN_REASON_BY_MESSAGE = {
    "Merlin assassinated": "assassination",
    "Three failed missions": "three_fails",
    "Five team proposals in a row rejected": "hammer",
    "Three successful missions": "three_successes",
    "Three missions succeeded": "three_successes",
}


class SkipGame(Exception):
    """Raised when a record cannot be represented as a 6p GameRecord."""


def parse_game(json_data, game_id):
    players = [p["name"] for p in json_data.get("players", [])]
    if len(players) != NUM_PLAYERS:
        raise SkipGame(f"{len(players)} players")
    seat_of = {name: i for i, name in enumerate(players)}

    outcome = json_data.get("outcome") or {}
    state = outcome.get("state")
    if state not in ("EVIL_WIN", "GOOD_WIN"):
        raise SkipGame(f"unfinished/odd outcome state {state!r}")
    winner = "evil" if state == "EVIL_WIN" else "good"
    win_reason = WIN_REASON_BY_MESSAGE.get(outcome.get("message"), "unknown")

    role_entries = outcome.get("roles") or []
    role_of = {r["name"]: r.get("role", "") for r in role_entries}
    if set(role_of) != set(players):
        raise SkipGame("roles do not cover the player list")
    evil = [0 if role_of[name] in GOOD_ROLES else 1 for name in players]

    variant_roles = sorted(
        {role for role in role_of.values() if role not in VANILLA_ROLES}
        | ({"ASSASSIN"} if any(r.get("assassin") for r in role_entries) else set())
    )

    events = []
    missions = json_data.get("missions") or []
    if len(missions) > 5:
        raise SkipGame(f"{len(missions)} missions")
    for q, mission in enumerate(missions, start=1):
        proposals = mission.get("proposals") or []
        for round_idx, prop in enumerate(proposals, start=1):
            try:
                team_seats = [seat_of[n] for n in prop["team"]]
                approver_seats = {seat_of[n] for n in prop.get("votes", [])}
            except KeyError as exc:
                raise SkipGame(f"unknown player name {exc} in quest {q}")
            leader_seat = seat_of.get(prop.get("proposer"))
            accepted = prop.get("state") == "APPROVED"
            approves = [s in approver_seats for s in range(NUM_PLAYERS)]
            # sanity: APPROVED iff strict majority approved
            if accepted != (sum(approves) > NUM_PLAYERS / 2):
                raise SkipGame(
                    f"vote/state mismatch in quest {q} proposal {round_idx}")
            events.append(proposal_event(q, round_idx, leader_seat, team_seats))
            events.append(vote_event(q, round_idx, approves, accepted))
        if mission.get("state") in ("SUCCESS", "FAIL"):
            events.append(quest_event(q, mission["state"] == "SUCCESS"))

    record = GameRecord(
        game_id=game_id,
        source="avalonlogs",
        seats=players,
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
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    game_id = os.path.splitext(os.path.basename(path))[0]
    return parse_game(data, game_id)


def parse_path(path, on_skip=None):
    """Yield GameRecords from a single file or a directory of game JSONs."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
        paths = [os.path.join(path, n) for n in names]
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
    ap.add_argument("path", help="game JSON file or directory of them")
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
