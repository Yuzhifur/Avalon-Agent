# test_model_v3.py
#
# Track A inference tests: exact constrained posterior under the
# exactly-2-evil constraint, ego conditioning, prior reweighting, and the
# pair posterior. Runnable with pytest or directly:
#   python test_model_v3.py

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "our"))

from model_v3 import EVIL_PAIRS, FactorGraphModelV3  # noqa: E402


def test_constrained_posterior_uniform():
    post = FactorGraphModelV3.constrained_posterior([0.5] * 6, algorithm="sum")
    for t in range(1, 7):
        assert abs(post[t]["evil"] - 1 / 3) < 1e-9


def test_constrained_posterior_sums_to_two():
    evidence = [0.9, 0.1, 0.4, 0.6, 0.2, 0.7]
    post = FactorGraphModelV3.constrained_posterior(evidence, algorithm="sum")
    total = sum(post[t]["evil"] for t in range(1, 7))
    assert abs(total - 2.0) < 1e-9


def test_constrained_posterior_self_conditioning():
    evidence = [0.5] * 6
    post = FactorGraphModelV3.constrained_posterior(
        evidence, self_role="good", self_index=0, algorithm="sum")
    assert post[1]["evil"] == 0.0
    for t in range(2, 7):
        assert abs(post[t]["evil"] - 2 / 5) < 1e-9

    post = FactorGraphModelV3.constrained_posterior(
        evidence, self_role="evil", self_index=0, algorithm="sum")
    assert post[1]["evil"] == 1.0
    for t in range(2, 7):
        assert abs(post[t]["evil"] - 1 / 5) < 1e-9


def test_strong_evidence_dominates():
    evidence = [0.95, 0.9, 0.05, 0.05, 0.05, 0.05]
    for alg in ("sum", "max"):
        post = FactorGraphModelV3.constrained_posterior(evidence, algorithm=alg)
        ranked = sorted(range(1, 7), key=lambda t: -post[t]["evil"])
        assert set(ranked[:2]) == {1, 2}


def test_update_priors_shifts_posterior():
    model = FactorGraphModelV3()
    model.update_priors({1: {"evil": 0.9, "good": 0.1}})
    post = model.constrained_posterior([0.5] * 6, priors=model._priors,
                                       algorithm="sum")
    assert post[1]["evil"] > 1 / 3


def test_pair_weights_match_posterior():
    evidence = [0.8, 0.3, 0.6, 0.2, 0.4, 0.55]
    weights = FactorGraphModelV3._pair_weights(evidence)
    assert set(weights) == set(EVIL_PAIRS)
    total = sum(weights.values())
    post = FactorGraphModelV3.constrained_posterior(evidence, algorithm="sum")
    for t in range(6):
        marginal = sum(w for pair, w in weights.items() if t in pair) / total
        assert abs(marginal - post[t + 1]["evil"]) < 1e-9


def test_pair_weights_ego_mask():
    evidence = [0.5] * 6
    weights = FactorGraphModelV3._pair_weights(
        evidence, self_role="good", self_index=2)
    assert all(2 not in pair for pair in weights)
    assert len(weights) == 10
    weights = FactorGraphModelV3._pair_weights(
        evidence, self_role="evil", self_index=2)
    assert all(2 in pair for pair in weights)
    assert len(weights) == 5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
