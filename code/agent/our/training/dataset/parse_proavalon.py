"""ProAvalon dump -> GameRecord (phase 3, Workstream 1).

Written ahead of data arrival against the voteHistory format already decoded
by generate_dataset_2.py, and to be validated on the real file. Per game:

  numberOfPlayers: 6
  voteHistory[player][mission_idx][proposal_idx] = string containing tokens
      VHleader / VHpicked / VHapprove / VHreject
  playerRoles[player] = {alliance: "Resistance"|"Spy", role: ...}
  missionHistory = ["succeeded"|"failed", ...]
  howTheGameWasWon = free text; contains "Hammer" when five rejected
      proposals ended the game (that final mission has no quest result)
  playerUsernamesOrderedReversed = fallback seat rotation

Unlike generate_dataset_2.py (which read only proposal [-1] of each mission),
this parser iterates ALL proposal indices, emitting rejected proposals as
real ProposalEvent/VoteEvent pairs — including the final hammer mission the
legacy vectorizer dropped entirely.

Seat order keeps the legacy convention: the leader rotation observed in
voteHistory (get_leader_order_one_cycle), so records are directly comparable
with the v2 pipeline.

CLI:  python parse_proavalon.py <dump.json|dump.jsonl> [--out corpus.jsonl]
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

VANILLA_ROLES = {"resistance", "spy", ""}


class SkipGame(Exception):
    pass


def get_leader_order_one_cycle(data):
    """Seat order = leader rotation. Ported from generate_dataset_2.py."""
    vote_history = data["voteHistory"]
    players = list(vote_history.keys())
    n_players = data["numberOfPlayers"]
    player_order = list(data.get("playerUsernamesOrderedReversed") or [])

    leader_order = []
    found = set()
    num_missions = len(vote_history[players[0]])
    for mission_index in range(num_missions):
        num_proposals = len(vote_history[players[0]][mission_index])
        for proposal_index in range(num_proposals):
            for p in players:
                if "VHleader" in vote_history[p][mission_index][proposal_index]:
                    if p not in found:
                        leader_order.append(p)
                        found.add(p)
                    if len(found) == n_players:
                        return leader_order
                    break

    if not leader_order:
        raise SkipGame("no VHleader tokens found")
    if len(leader_order) < n_players:
        if sorted(player_order) != sorted(players):
            raise SkipGame("cannot complete leader rotation (bad player order)")
        while player_order[0] != leader_order[0]:
            player_order.insert(0, player_order.pop())
        leader_order = player_order
    if len(leader_order) != n_players:
        raise SkipGame("leader rotation does not cover all players")
    return leader_order


def normalize_win_reason(how_won, mission_history):
    text = str(how_won or "").lower()
    if "hammer" in text:
        return "hammer"
    if "assassin" in text or "merlin" in text:
        return "assassination"
    fails = sum(1 for m in mission_history if m == "failed")
    successes = sum(1 for m in mission_history if m == "succeeded")
    if fails >= 3:
        return "three_fails"
    if successes >= 3:
        return "three_successes"
    return "unknown"


def parse_game(data, game_id):
    if data.get("numberOfPlayers") != NUM_PLAYERS:
        raise SkipGame(f"{data.get('numberOfPlayers')} players")
    vote_history = data.get("voteHistory") or {}
    if len(vote_history) != NUM_PLAYERS:
        raise SkipGame(f"voteHistory covers {len(vote_history)} players")

    seats = get_leader_order_one_cycle(data)
    seat_of = {name: i for i, name in enumerate(seats)}

    player_roles = data.get("playerRoles") or {}
    if set(player_roles) != set(seats):
        raise SkipGame("playerRoles does not cover the players")
    evil = [
        1 if player_roles[name].get("alliance") == "Spy" else 0
        for name in seats
    ]
    variant_roles = sorted(
        {
            str(player_roles[name].get("role", "")).upper()
            for name in seats
            if str(player_roles[name].get("role", "")).lower() not in VANILLA_ROLES
        }
    )

    mission_history = list(data.get("missionHistory") or [])
    how_won = data.get("howTheGameWasWon", "")
    win_reason = normalize_win_reason(how_won, mission_history)

    winner = data.get("winningTeam")
    if winner == "Resistance":
        winner = "good"
    elif winner == "Spy":
        winner = "evil"
    else:
        # Field name varies across dump vintages; infer from the end state.
        winner = {
            "three_fails": "evil",
            "hammer": "evil",
            "assassination": "evil",
            "three_successes": "good",
        }.get(win_reason)

    events = []
    num_missions = len(vote_history[seats[0]])
    if num_missions > 5:
        raise SkipGame(f"{num_missions} missions")
    for mission_index in range(num_missions):
        proposal_counts = {len(vote_history[p][mission_index]) for p in seats}
        if len(proposal_counts) != 1:
            raise SkipGame(f"ragged voteHistory in mission {mission_index + 1}")
        num_proposals = proposal_counts.pop()
        mission_has_accepted = False
        for proposal_index in range(num_proposals):
            leader_seat = None
            team_seats = []
            approves = [False] * NUM_PLAYERS
            for name in seats:
                tokens = vote_history[name][mission_index][proposal_index]
                seat = seat_of[name]
                if "VHleader" in tokens:
                    if leader_seat is not None:
                        raise SkipGame(
                            f"two leaders in mission {mission_index + 1}")
                    leader_seat = seat
                if "VHpicked" in tokens:
                    team_seats.append(seat)
                if "VHapprove" in tokens:
                    approves[seat] = True
            if not team_seats:
                raise SkipGame(f"empty team in mission {mission_index + 1}")
            accepted = sum(approves) > NUM_PLAYERS / 2
            mission_has_accepted = mission_has_accepted or accepted
            quest = mission_index + 1
            round_idx = proposal_index + 1
            events.append(
                proposal_event(quest, round_idx, leader_seat, team_seats))
            events.append(vote_event(quest, round_idx, approves, accepted))
        # A quest only resolves when a team was actually sent. Hammer games
        # (howTheGameWasWon == "Hammer rejected.") record the final mission as
        # "failed" in missionHistory even though its five proposals were all
        # rejected and the quest was never played — emitting a QuestEvent there
        # would be a result with no accepted proposal to attach it to. The
        # rejected proposals themselves are kept as ProposalEvent/VoteEvents:
        # they are real votes and exactly the rejected-history signal phase 3
        # is built to use.
        if (mission_has_accepted
                and mission_index < len(mission_history)
                and mission_history[mission_index] in ("succeeded", "failed")):
            events.append(
                quest_event(mission_index + 1,
                            mission_history[mission_index] == "succeeded"))

    record = GameRecord(
        game_id=game_id,
        source="proavalon",
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


def iter_raw_games(path):
    """Yield (game_dict, game_id) from a JSON array dump or a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(64).lstrip()
        fh.seek(0)
        if head.startswith("["):
            data = json.load(fh)
            for i, game in enumerate(data):
                yield game, str(game.get("_id", f"proavalon_{i:06d}"))
        else:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    game = json.loads(line)
                    yield game, str(game.get("_id", f"proavalon_{i:06d}"))


def parse_path(path, on_skip=None):
    for game, game_id in iter_raw_games(path):
        try:
            yield parse_game(game, game_id)
        except SkipGame as exc:
            if on_skip:
                on_skip(game_id, str(exc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="ProAvalon dump (.json array or .jsonl)")
    ap.add_argument("--out", help="write GameRecords to this JSONL file")
    args = ap.parse_args(argv)

    skips = []
    records = []
    for rec in parse_path(args.path, on_skip=lambda g, r: skips.append((g, r))):
        records.append(rec)
    print(f"parsed {len(records)} 6p games, skipped {len(skips)}")
    for g, reason in skips[:10]:
        print(f"  skip {g}: {reason}")
    if args.out:
        write_jsonl(records, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
