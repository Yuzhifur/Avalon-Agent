"""Belief-quality metrics for the offline replay harness (Workstream 0).

All functions are pure and operate on flat arrays/lists collected by
replay_eval.py. Conventions:

  - label: 1 = the target player is truly evil
  - score/prob: the evaluated model's P(target is evil) from a good ego's
    perspective; self-pairs (target == ego) are excluded upstream
  - pair hypotheses: the 15 unordered seat pairs from itertools.combinations
"""

from itertools import combinations
import math

import numpy as np

NUM_PLAYERS = 6
EVIL_PAIRS = list(combinations(range(NUM_PLAYERS), 2))


def roc_auc(labels, scores):
    """Pooled ROC-AUC; None when only one class is present."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def brier(labels, probs):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return None
    return float(np.mean((probs - labels) ** 2))


def ece(labels, probs, n_bins=10):
    """Expected calibration error with equal-width probability bins."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        err += (count / total) * abs(labels[in_bin].mean() - probs[in_bin].mean())
    return float(err)


def nll(labels, probs, eps=1e-9):
    labels = np.asarray(labels, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    if len(labels) == 0:
        return None
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def pair_scores_from_marginals(p_evil):
    """{pair: score} from per-seat P(evil) assuming independence across seats,
    restricted to the 15 exactly-2-evil hypotheses and renormalized.

    Exact for backends that emit a pair distribution natively (they should
    bypass this); a documented approximation for marginal-only backends like
    factor_v2.
    """
    eps = 1e-12
    weights = {}
    for pair in EVIL_PAIRS:
        w = 1.0
        for seat in range(NUM_PLAYERS):
            p = min(max(p_evil.get(seat, 0.5), eps), 1.0 - eps)
            w *= p if seat in pair else (1.0 - p)
        weights[pair] = w
    total = sum(weights.values())
    if total <= 0:
        return {pair: 1.0 / len(EVIL_PAIRS) for pair in EVIL_PAIRS}
    return {pair: w / total for pair, w in weights.items()}


def pair_topk_hit(pair_scores, true_pair, k):
    """1.0 if the true evil pair is among the k highest-scoring hypotheses."""
    ranked = sorted(pair_scores, key=lambda p: -pair_scores[p])
    return 1.0 if tuple(sorted(true_pair)) in [tuple(sorted(p)) for p in ranked[:k]] else 0.0


def summarize_beliefs(samples):
    """Aggregate a list of per-(state, ego, target) samples.

    Each sample: {label, prob, quest, is_quest_boundary}. Returns the metric
    block used in replay reports: pooled and by-quest AUC, calibration, and
    mean P(evil) on true evil by quest.
    """
    out = {
        "n_samples": len(samples),
        "auc_pooled": roc_auc([s["label"] for s in samples],
                              [s["prob"] for s in samples]),
        "brier": brier([s["label"] for s in samples],
                       [s["prob"] for s in samples]),
        "ece": ece([s["label"] for s in samples],
                   [s["prob"] for s in samples]),
        "nll": nll([s["label"] for s in samples],
                   [s["prob"] for s in samples]),
        "auc_by_quest": {},
        "mean_p_evil_on_true_evil_by_quest": {},
        "mean_p_evil_on_true_good_by_quest": {},
    }
    for quest in range(1, 6):
        subset = [s for s in samples if s["quest"] == quest]
        out["auc_by_quest"][quest] = roc_auc(
            [s["label"] for s in subset], [s["prob"] for s in subset])
        evil = [s["prob"] for s in subset if s["label"] == 1]
        good = [s["prob"] for s in subset if s["label"] == 0]
        out["mean_p_evil_on_true_evil_by_quest"][quest] = (
            float(np.mean(evil)) if evil else None)
        out["mean_p_evil_on_true_good_by_quest"][quest] = (
            float(np.mean(good)) if good else None)
    evil_all = [s["prob"] for s in samples if s["label"] == 1]
    out["mean_p_evil_on_true_evil"] = float(np.mean(evil_all)) if evil_all else None
    out["median_p_evil_on_true_evil"] = (
        float(np.median(evil_all)) if evil_all else None)
    out["share_true_evil_rated_good"] = (
        float(np.mean([p < 0.5 for p in evil_all])) if evil_all else None)
    return out


def summarize_pairs(samples):
    """Aggregate per-state pair-hypothesis samples:
    {pair_scores: {pair: p}, true_pair, quest}."""
    if not samples:
        return {"n_states": 0, "pair_top1": None, "pair_top3": None}
    top1 = [pair_topk_hit(s["pair_scores"], s["true_pair"], 1) for s in samples]
    top3 = [pair_topk_hit(s["pair_scores"], s["true_pair"], 3) for s in samples]
    out = {
        "n_states": len(samples),
        "pair_top1": float(np.mean(top1)),
        "pair_top3": float(np.mean(top3)),
        "pair_top1_by_quest": {},
    }
    for quest in range(1, 6):
        subset = [pair_topk_hit(s["pair_scores"], s["true_pair"], 1)
                  for s in samples if s["quest"] == quest]
        out["pair_top1_by_quest"][quest] = (
            float(np.mean(subset)) if subset else None)
    return out
