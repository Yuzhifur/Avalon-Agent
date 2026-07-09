#!/usr/bin/env python3
"""Language-fingerprint evaluation over live-game chat logs (stdlib only).

docs/limitations_and_next_steps.md §6 layer 1: can a simple classifier
identify an agent's ROLE (good/evil) or its PROMPT PERSONA from public chat
alone? If yes, the speech layer carries an exploitable fingerprint that is
part of the self-play environment rather than of Avalon reasoning.

Tasks (all classifiers are multinomial naive Bayes over word 1-2 grams +
char 3-4 grams, evaluated with game-grouped k-fold cross-validation so no
game contributes to both train and test):

  role      good-vs-evil from a player's messages — per message and per
            player-game (all of one player's messages in one game pooled).
            The per-player-game number is the practically relevant one:
            "can you spot the spy from their chat?" Chance = the majority
            class (4/6 good).
  persona   which persona-bank voice produced the message (diversified runs
            only; labels parsed from the agents' PROMPT_PERSONA log lines).
            Here HIGH accuracy is the goal — it verifies the personas are
            actually distinct — while role accuracy dropping toward chance
            is the fingerprint-reduction goal.
  condition baseline-vs-diversified (sanity check that the intervention
            changed the text distribution at all).

Also reports style statistics per condition (message length, hedging/
question markers, lexical diversity) — the §6 layer-1 descriptive view.

Usage:
  python3 fingerprint_classifier.py --baseline ROOT [ROOT ...] \
      [--diversified ROOT [ROOT ...]] [--cell NAME] [--folds 5] [--json OUT]

A ROOT is a run-output root (e.g. evaluation/phase3_live_runs/<ts>); games
are discovered as */run_*/server/*.json with a completed status.json.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict


# ---------------------------------------------------------------- extraction

def _terminal_full(payload):
    for entry in reversed(payload.get("logs", [])):
        full = entry.get("full", {})
        if full.get("winner"):
            return full
    return {}


def _persona_labels(run_dir):
    """{lowercase player: persona} from agent LOG_(Name)_<gid>.log lines."""
    labels = {}
    for path in glob.glob(os.path.join(run_dir, "agent", "LOG_*.log")):
        name = os.path.basename(path).split("(")[1].split(")")[0].lower()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "PROMPT_PERSONA " in line:
                        labels[name] = line.rsplit("PROMPT_PERSONA ", 1)[1].strip()
                        break
        except OSError:
            continue
    return labels


def extract_games(roots, cell=None):
    """-> list of {game, cell, winner, roles: {name: 'good'|'evil'},
    personas: {name: key}, messages: [(player, text)]} — players only."""
    games = []
    for root in roots:
        for status_path in sorted(glob.glob(
                os.path.join(root, "*", "run_*", "status.json"))):
            run_dir = os.path.dirname(status_path)
            cand = os.path.basename(os.path.dirname(run_dir))
            if cell and cand != cell:
                continue
            try:
                with open(status_path, "r", encoding="utf-8") as fh:
                    status = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") != "completed":
                continue
            server_logs = glob.glob(os.path.join(run_dir, "server", "*.json"))
            if not server_logs:
                continue
            try:
                with open(server_logs[0], "r", encoding="utf-8") as fh:
                    full = _terminal_full(json.load(fh))
            except (OSError, json.JSONDecodeError):
                continue
            if not full:
                continue
            roles = {
                p["name"].lower(): ("evil" if str(p.get("role", "")).lower()
                                    .startswith("minion") else "good")
                for p in full.get("all_players", [])
            }
            messages = [
                (m["player"].lower(), m.get("msg", ""))
                for m in full.get("messages", [])
                if m.get("player") and m["player"] != "system" and m.get("msg")
            ]
            games.append({
                "game": os.path.splitext(os.path.basename(server_logs[0]))[0]
                         + ":" + os.path.basename(run_dir) + ":" + cand,
                "cell": cand,
                "winner": (full.get("winner") or "").lower(),
                "roles": roles,
                "personas": _persona_labels(run_dir),
                "messages": messages,
            })
    return games


# ------------------------------------------------------------------ features

_WORD_RE = re.compile(r"[a-z']+|[?!.,]")


def mask_names(text, names):
    """Replace roster names with a placeholder so the role classifier can't
    key on player identities (who-mentions-whom), only on style/content."""
    out = text
    for name in names:
        out = re.sub(rf"\b{re.escape(name)}\b", "player", out, flags=re.I)
    return out


def featurize(text):
    """Sparse count features: word unigrams+bigrams and char 3-4 grams."""
    words = _WORD_RE.findall(text.lower())
    feats = Counter()
    for w in words:
        feats["w:" + w] += 1
    for a, b in zip(words, words[1:]):
        feats["b:" + a + "_" + b] += 1
    compact = " " + " ".join(words) + " "
    for n in (3, 4):
        for i in range(len(compact) - n + 1):
            feats["c:" + compact[i:i + n]] += 1
    return feats


# ------------------------------------------------- naive Bayes + evaluation

class NaiveBayes:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, featsets, labels):
        self.classes = sorted(set(labels))
        self.prior = {c: math.log(labels.count(c) / len(labels))
                      for c in self.classes}
        self.counts = {c: Counter() for c in self.classes}
        self.totals = {c: 0 for c in self.classes}
        vocab = set()
        for feats, label in zip(featsets, labels):
            self.counts[label].update(feats)
            self.totals[label] += sum(feats.values())
            vocab.update(feats)
        self.vocab_size = len(vocab) or 1
        return self

    def scores(self, feats):
        out = {}
        for c in self.classes:
            s = self.prior[c]
            denom = math.log(self.totals[c] + self.alpha * self.vocab_size)
            for f, n in feats.items():
                s += n * (math.log(self.counts[c][f] + self.alpha) - denom)
            out[c] = s
        return out

    def predict(self, feats):
        s = self.scores(feats)
        return max(s, key=s.get)


def group_folds(groups, k):
    """Round-robin games into k folds (deterministic by sorted game id)."""
    fold_of = {}
    for i, g in enumerate(sorted(set(groups))):
        fold_of[g] = i % k
    return [fold_of[g] for g in groups]


def auc_from_scores(scores, labels, positive):
    """Rank-based (Mann-Whitney) AUC for the `positive` class."""
    pos = sorted(s for s, l in zip(scores, labels) if l == positive)
    neg = sorted(s for s, l in zip(scores, labels) if l != positive)
    if not pos or not neg:
        return float("nan")
    ranked = sorted((s, 1 if i else 0) for i, group in enumerate((neg, pos))
                    for s in group)
    # average ranks with tie handling
    rank_sum, i, n = 0.0, 0, len(ranked)
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum += avg_rank * sum(lab for _, lab in ranked[i:j + 1])
        i = j + 1
    u = rank_sum - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def cross_validate(featsets, labels, groups, k, positive=None):
    """-> dict(accuracy, majority, auc, n). Game-grouped k-fold CV."""
    folds = group_folds(groups, k)
    correct = total = 0
    all_scores, all_labels = [], []
    for fold in range(k):
        train = [i for i, f in enumerate(folds) if f != fold]
        test = [i for i, f in enumerate(folds) if f == fold]
        if not train or not test:
            continue
        train_labels = [labels[i] for i in train]
        if len(set(train_labels)) < 2:
            continue
        model = NaiveBayes().fit([featsets[i] for i in train], train_labels)
        for i in test:
            scores = model.scores(featsets[i])
            pred = max(scores, key=scores.get)
            correct += int(pred == labels[i])
            total += 1
            if positive is not None and positive in scores:
                others = [v for c, v in scores.items() if c != positive]
                all_scores.append(scores[positive] - max(others))
                all_labels.append(labels[i])
    majority = max(Counter(labels).values()) / len(labels) if labels else 0.0
    return {
        "n": total,
        "accuracy": correct / total if total else float("nan"),
        "majority_baseline": majority,
        "auc": (auc_from_scores(all_scores, all_labels, positive)
                if positive is not None else None),
    }


# --------------------------------------------------------------- style stats

_HEDGES = ("maybe", "idk", "not sure", "i think", "could be", "probably",
           "i guess", "might", "kinda", "unsure")


def style_stats(games):
    msgs = [(p, t) for g in games for p, t in g["messages"]]
    if not msgs:
        return {}
    words_per = [len(t.split()) for _, t in msgs]
    joined = " ".join(t.lower() for _, t in msgs)
    tokens = joined.split()
    return {
        "games": len(games),
        "messages": len(msgs),
        "messages_per_game": round(len(msgs) / len(games), 1),
        "words_per_message_mean": round(sum(words_per) / len(words_per), 2),
        "words_per_message_p90": sorted(words_per)[int(0.9 * len(words_per))],
        "hedge_rate": round(sum(any(h in t.lower() for h in _HEDGES)
                                for _, t in msgs) / len(msgs), 3),
        "question_rate": round(sum("?" in t for _, t in msgs) / len(msgs), 3),
        "type_token_ratio": round(len(set(tokens)) / len(tokens), 3) if tokens else 0.0,
    }


# --------------------------------------------------------------------- tasks

def role_tasks(games, k, masked=True):
    """Per-message and per-player-game role classification. Roster names are
    masked by default so identity mentions can't stand in for style."""
    m_feats, m_labels, m_groups = [], [], []
    pg_feats, pg_labels, pg_groups = [], [], []
    for g in games:
        names = list(g["roles"])
        per_player = defaultdict(list)
        for player, text in g["messages"]:
            role = g["roles"].get(player)
            if not role:
                continue
            if masked:
                text = mask_names(text, names)
            m_feats.append(featurize(text))
            m_labels.append(role)
            m_groups.append(g["game"])
            per_player[player].append(text)
        for player, texts in per_player.items():
            pg_feats.append(featurize(" ".join(texts)))
            pg_labels.append(g["roles"][player])
            pg_groups.append(g["game"])
    return (cross_validate(m_feats, m_labels, m_groups, k, positive="evil"),
            cross_validate(pg_feats, pg_labels, pg_groups, k, positive="evil"))


def persona_task(games, k):
    feats, labels, groups = [], [], []
    for g in games:
        for player, text in g["messages"]:
            key = g["personas"].get(player)
            if not key:
                continue
            feats.append(featurize(text))
            labels.append(key)
            groups.append(g["game"])
    if len(set(labels)) < 2:
        return None
    return cross_validate(feats, labels, groups, k)


def condition_task(base_games, div_games, k):
    feats, labels, groups = [], [], []
    for cond, games in (("baseline", base_games), ("diversified", div_games)):
        for g in games:
            for player, text in g["messages"]:
                feats.append(featurize(text))
                labels.append(cond)
                groups.append(g["game"])
    return cross_validate(feats, labels, groups, k, positive="diversified")


def fmt(result):
    if result is None:
        return "n/a (missing labels)"
    auc = ("" if result["auc"] is None or math.isnan(result["auc"])
           else f"  AUC {result['auc']:.3f}")
    return (f"acc {result['accuracy']:.3f} vs majority "
            f"{result['majority_baseline']:.3f} (n={result['n']}){auc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="run roots with the default template (e.g. the Gate-2 dirs)")
    ap.add_argument("--diversified", nargs="*", default=[],
                    help="run roots with GRAIL_PROMPT_PERSONA_BANK games")
    ap.add_argument("--cell", default=None,
                    help="only use this candidate cell within --diversified roots")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base_games = extract_games(args.baseline)
    div_games = extract_games(args.diversified, cell=args.cell)

    report = {"baseline": {}, "diversified": {}}
    print("=" * 68)
    print("LANGUAGE-FINGERPRINT EVALUATION (naive Bayes, game-grouped CV)")
    print("=" * 68)

    print(f"\nBASELINE (templated): {len(base_games)} games")
    report["baseline"]["style"] = style_stats(base_games)
    print("  style:", json.dumps(report["baseline"]["style"]))
    msg, pg = role_tasks(base_games, args.folds)
    report["baseline"]["role_per_message"] = msg
    report["baseline"]["role_per_player_game"] = pg
    print("  role per-message:     ", fmt(msg))
    print("  role per-player-game: ", fmt(pg))

    if div_games:
        print(f"\nDIVERSIFIED (persona bank): {len(div_games)} games")
        report["diversified"]["style"] = style_stats(div_games)
        print("  style:", json.dumps(report["diversified"]["style"]))
        msg_d, pg_d = role_tasks(div_games, args.folds)
        report["diversified"]["role_per_message"] = msg_d
        report["diversified"]["role_per_player_game"] = pg_d
        print("  role per-message:     ", fmt(msg_d))
        print("  role per-player-game: ", fmt(pg_d))
        per = persona_task(div_games, args.folds)
        report["diversified"]["persona"] = per
        print("  persona id (want HIGH):", fmt(per))
        cond = condition_task(base_games, div_games, args.folds)
        report["condition_baseline_vs_diversified"] = cond
        print("\n  condition (baseline vs diversified):", fmt(cond))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
