"""Corpus builder (Workstream 1): GameRecord JSONLs + stats + splits.

  python build_corpus.py --avalonlogs E:\\Local\\avalonlogs_6p \\
      [--selfplay <runs-root>] [--proavalon <dump>] \\
      [--out-dir ../../../../../data/corpora] [--splits-dir <offline/splits>]

Per source this writes data/corpora/<source>.jsonl plus a stats report
(counts, role-variant breakdown, winner/win-reason mix, proposals-per-quest
histogram, rejected-count and proposal-depth coverage histograms — the
coverage evidence the plan's self-play-augmentation decision keys off — and
a chat-presence check for ProAvalon).

Role-variant filter (plan risk #2): keep games whose special roles are within
{MERLIN, PERCIVAL, ASSASSIN, MORGANA}; label by alliance only. Games with
MORDRED/OBERON (or parse-level defects) are dropped and counted.

Game-level train/val/test splits (fixed seed) are written to
code/evaluation/offline/splits/<source>.json for the training and offline
eval code to share. The self-play phase-2 set gets a single eval-only list.
"""

import argparse
import collections
import json
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from event_schema import read_jsonl, write_jsonl
    import parse_avalonlogs
    import parse_proavalon
    import parse_selfplay
else:
    from .event_schema import read_jsonl, write_jsonl
    from . import parse_avalonlogs, parse_proavalon, parse_selfplay

REPO_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "corpora")
DEFAULT_SPLITS_DIR = os.path.join(
    REPO_ROOT, "code", "evaluation", "offline", "splits")

SPLIT_SEED = 20260610  # matches the prior card-dataset split convention
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

ALLOWED_VARIANT_ROLES = {"MERLIN", "PERCIVAL", "ASSASSIN", "MORGANA"}


def variant_ok(record):
    return set(record.variant_roles) <= ALLOWED_VARIANT_ROLES


def corpus_stats(records, source):
    stats = {
        "source": source,
        "n_games": len(records),
        "winner": dict(collections.Counter(r.winner for r in records)),
        "win_reason": dict(collections.Counter(r.win_reason for r in records)),
        "variant_roles": dict(collections.Counter(
            "+".join(r.variant_roles) or "vanilla" for r in records)),
        "proposals_per_quest": {},
        "rejected_per_game": {},
        "proposal_depth": {},   # round_idx distribution over all proposals
        "voted_proposals_total": 0,
    }
    per_quest = collections.defaultdict(list)
    depth = collections.Counter()
    rejected_per_game = collections.Counter()
    for r in records:
        rejected = 0
        quest_counts = collections.Counter()
        for ev in r.events:
            if ev["type"] == "vote":
                stats["voted_proposals_total"] += 1
                quest_counts[ev["quest"]] += 1
                depth[ev["round_idx"]] += 1
                if not ev["accepted"]:
                    rejected += 1
        for q, c in quest_counts.items():
            per_quest[q].append(c)
        rejected_per_game[rejected] += 1
    stats["proposals_per_quest"] = {
        q: round(sum(v) / len(v), 3) for q, v in sorted(per_quest.items())}
    stats["proposal_depth"] = {k: depth[k] for k in sorted(depth)}
    stats["rejected_per_game"] = {
        k: rejected_per_game[k] for k in sorted(rejected_per_game)}
    return stats


def make_splits(records, seed=SPLIT_SEED):
    ids = [r.game_id for r in records]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_val = int(n * SPLIT_FRACTIONS["val"])
    return {
        "seed": seed,
        "fractions": SPLIT_FRACTIONS,
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def proavalon_chat_check(path, sample=200):
    """Plan requirement: does the ProAvalon dump carry chat? Scans raw games
    for plausible chat fields."""
    keys = collections.Counter()
    checked = 0
    for game, _gid in parse_proavalon.iter_raw_games(path):
        for key in game:
            if any(tok in key.lower() for tok in ("chat", "message", "log")):
                keys[key] += 1
        checked += 1
        if checked >= sample:
            break
    return {"games_scanned": checked, "chat_like_fields": dict(keys)}


def has_forced_votes(record):
    return any(ev.get("forced") for ev in record.events
               if ev["type"] == "vote")


def build_source(name, records_iter, out_dir, splits_dir, eval_only=False):
    skipped = collections.Counter()
    records = []
    for record in records_iter:
        if not variant_ok(record):
            skipped["variant_roles"] += 1
            continue
        # forced-fifth variant games stay out of training corpora (no human
        # data has the rule); eval-only sets keep them for the C3/C4 cells
        if not eval_only and has_forced_votes(record):
            skipped["forced_fifth_variant"] += 1
            continue
        records.append(record)

    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, f"{name}.jsonl")
    write_jsonl(records, corpus_path)

    stats = corpus_stats(records, name)
    stats["skipped"] = dict(skipped)
    stats_path = os.path.join(out_dir, f"stats_{name}.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    os.makedirs(splits_dir, exist_ok=True)
    split_path = os.path.join(splits_dir, f"{name}.json")
    if eval_only:
        payload = {
            "eval_only": True,
            "games": [
                {"game_id": r.game_id, "winner": r.winner} for r in records],
        }
    else:
        payload = make_splits(records)
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print(f"{name}: {len(records)} games -> {corpus_path}")
    print(f"  stats -> {stats_path}; splits -> {split_path}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return records, stats


def load_split_records(corpus_path, split_path, split):
    """Helper for trainers: records of one split of a built corpus."""
    with open(split_path, "r", encoding="utf-8") as fh:
        splits = json.load(fh)
    wanted = set(splits[split])
    return [r for r in read_jsonl(corpus_path) if r.game_id in wanted]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--avalonlogs", help="directory of avalonlogs 6p JSONs")
    ap.add_argument("--selfplay", help="policy-runs root of server logs")
    ap.add_argument("--proavalon", help="ProAvalon dump (validated on arrival)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    args = ap.parse_args(argv)

    if not (args.avalonlogs or args.selfplay or args.proavalon):
        ap.error("give at least one of --avalonlogs/--selfplay/--proavalon")

    skips = []
    if args.avalonlogs:
        build_source(
            "avalonlogs",
            parse_avalonlogs.parse_path(
                args.avalonlogs, on_skip=lambda p, r: skips.append((p, r))),
            args.out_dir, args.splits_dir)
    if args.selfplay:
        build_source(
            "selfplay90",
            parse_selfplay.parse_path(
                args.selfplay, on_skip=lambda p, r: skips.append((p, r))),
            args.out_dir, args.splits_dir, eval_only=True)
    if args.proavalon:
        chat = proavalon_chat_check(args.proavalon)
        print(f"proavalon chat check: {chat}")
        records, stats = build_source(
            "proavalon",
            parse_proavalon.parse_path(
                args.proavalon, on_skip=lambda g, r: skips.append((g, r))),
            args.out_dir, args.splits_dir)
        stats["chat_check"] = chat
        with open(os.path.join(args.out_dir, "stats_proavalon.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)

    if skips:
        print(f"parser skips: {len(skips)} (first 5: {skips[:5]})")


if __name__ == "__main__":
    main()
