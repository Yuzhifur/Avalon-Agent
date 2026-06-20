"""Loaders for the committed phase-2 run artifacts (Workstream 0).

A phase-2 policy-run directory (e.g. policy_runs_overnight/20260608T180435Z)
contains per run:

  run_xxx/server/<GAMEID>.json       committed server log
  run_xxx/agent/CSV_(Name)_<GAMEID>.csv   committed live-belief traces
  run_xxx/agent/DEBUG_(Name)_<GAMEID>.log POLICY_DECISION lines (gitignored;
                                     present only on the machine that ran the
                                     games — used for cross-validation and
                                     for extracting the committed
                                     policy_decisions.jsonl archives)

CSV format (ACLAgent.initialize_csv/data_csv): row 1 = "round,<names...>"
(sorted names), row 2 = roles, then one row per vote_party action with the
agent's current P(evil) per player — exactly the beliefs its policy consumed
for that vote, vibes included.
"""

import ast
import csv
import glob
import json
import os
import re

from . import add_agent_paths

add_agent_paths()

import parse_selfplay  # noqa: E402


def find_runs(root):
    """Yield (run_dir, server_log_path, game_id) for every run with a server log."""
    for server_log in sorted(
            glob.glob(os.path.join(root, "**", "server", "*.json"),
                      recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(server_log))
        game_id = os.path.splitext(os.path.basename(server_log))[0]
        yield run_dir, server_log, game_id


def load_games(root, on_skip=None):
    """Parse every completed game under root.

    Returns a list of dicts: {record, run_dir, game_id}.
    """
    games = []
    for run_dir, server_log, game_id in find_runs(root):
        try:
            record = parse_selfplay.parse_file(server_log)
        except parse_selfplay.SkipGame as exc:
            if on_skip:
                on_skip(server_log, str(exc))
            continue
        games.append({"record": record, "run_dir": run_dir, "game_id": game_id})
    return games


def load_belief_csv(path):
    """One agent's belief trace.

    Returns {names: [lowercase...], roles: {name: role}, rows: [{round, beliefs:
    {name: p_evil}}]}.
    """
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        roles_row = next(reader)
        names = [n.strip().lower() for n in header[1:]]
        roles = {n: r for n, r in zip(names, roles_row[1:])}
        rows = []
        for row in reader:
            if not row or not row[0]:
                continue
            rows.append({
                "round": int(row[0]),
                "beliefs": {n: float(v) for n, v in zip(names, row[1:])},
            })
    return {"names": names, "roles": roles, "rows": rows}


def load_run_belief_csvs(run_dir, game_id):
    """{lowercase agent name: parsed CSV} for one run."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "agent", f"CSV_(*)_{game_id}.csv")):
        m = re.search(r"CSV_\((.+)\)_", os.path.basename(path))
        if m:
            out[m.group(1).lower()] = load_belief_csv(path)
    return out


# ---------------------------------------------------------------------------
# DEBUG log decision extraction (authoritative phase-2 source; logs are
# gitignored so these helpers also export committed .jsonl archives)
# ---------------------------------------------------------------------------

POLICY_LINE = re.compile(
    r"^POLICY_DECISION (?P<action>\w+) team=(?P<team>\w+) quest=(?P<quest>\d+) "
    r"score_good=(?P<good>\d+) score_evil=(?P<evil>\d+) "
    r"failed_party_votes=(?P<fpv>\d+) known_evil=(?P<known>\[.*?\])"
    r"(?: party=(?P<party>\[.*?\]) vote=(?P<vote>\w+))?"
    r"(?: last_party=(?P<last_party>\[.*?\]))?"
    r"(?: selected=(?P<selected>\[.*?\]))?")

DETAIL_PREFIX = "POLICY_DECISION_DETAIL "


def parse_debug_log(path):
    """Yield {action, team, quest, fpv, party, vote, detail} decisions."""
    decisions = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = POLICY_LINE.match(line)
            if m:
                d = m.groupdict()
                decisions.append({
                    "action": d["action"],
                    "team": d["team"],
                    "quest": int(d["quest"]),
                    "good_wins": int(d["good"]),
                    "evil_wins": int(d["evil"]),
                    "fpv": int(d["fpv"]),
                    "party": ast.literal_eval(d["party"]) if d["party"] else None,
                    "vote": (None if d["vote"] is None
                             else d["vote"] == "True"),
                    "detail": None,
                })
            elif line.startswith(DETAIL_PREFIX) and decisions:
                payload = json.loads(line[len(DETAIL_PREFIX):])
                if decisions[-1]["detail"] is None and (
                        payload.get("action") == {
                            "vote_party": "vote_party",
                            "vote_quest": "vote_quest",
                            "propose_party": "propose_party",
                        }.get(decisions[-1]["action"])):
                    decisions[-1]["detail"] = payload
    return decisions


def load_run_debug_decisions(run_dir, game_id):
    """{lowercase agent name: [decisions]} from DEBUG logs, or {} if absent."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "agent", f"DEBUG_(*)_{game_id}.log")):
        m = re.search(r"DEBUG_\((.+)\)_", os.path.basename(path))
        if m:
            out[m.group(1).lower()] = parse_debug_log(path)
    return out


def extract_decisions_jsonl(run_dir, game_id):
    """Archive POLICY_DECISION streams as committed-friendly JSONL.

    Writes run_dir/agent/policy_decisions_<GAMEID>.jsonl (one line per agent
    decision, with the agent name attached) and returns the path, or None if
    no DEBUG logs are present. This satisfies the plan's log-retention rule
    without committing the multi-GB raw logs.
    """
    decisions = load_run_debug_decisions(run_dir, game_id)
    if not decisions:
        return None
    out_path = os.path.join(run_dir, "agent", f"policy_decisions_{game_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for agent, items in sorted(decisions.items()):
            for item in items:
                row = dict(item)
                row["agent"] = agent
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")
    return out_path
