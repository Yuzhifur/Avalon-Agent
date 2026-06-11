# model_v3.py
#
# Track A belief model (Workstream 2): FactorGraphModelV3.
#
# The card encoder produces one evil-evidence probability per player; the
# exactly-2-evil constraint is then applied by exact enumeration over the 15
# valid evil pairs. With v2's party/vote/outcome nodes always observed, its
# factor graph reduces to 6 unary evidence factors plus the joint constraint
# — a tree — so this closed form computes the same posterior family that v2
# approximated with loopy BP, exactly and in microseconds. No pomegranate,
# no edge-insertion-order hazard.
#
# The external interface (construct / load_from_file / predict_probs /
# update_priors) matches FactorGraphModelV2 so ACLAgent and the LLM-vibes
# prior blend work unchanged. game_state here is the canonical card list from
# proposal_cards (the model factory adapts GameInfo at runtime).
#
# algorithm="sum" is the default: the prior phase-3 Stage-A evaluation found
# sum-product beats max-product on calibration for this posterior.

from itertools import combinations
import os

import torch

try:
    from .card_encoder import ProposalSetDistribution, load_card_model
    from .proposal_cards import NUM_PLAYERS, cards_to_tensor
except ImportError:  # imported top-level by training/eval scripts
    from card_encoder import ProposalSetDistribution, load_card_model
    from proposal_cards import NUM_PLAYERS, cards_to_tensor

EVIL_PAIRS = list(combinations(range(NUM_PLAYERS), 2))  # 15 valid assignments


def _role_name(self_role):
    """Accept the agent ATEAM enum or a plain 'good'/'evil' string."""
    name = getattr(self_role, "name", self_role)
    return str(name).upper()


class FactorGraphModelV3:
    def __init__(self):
        self.encoder = None
        self.temperature = 1.0
        self._priors = self._neutral_priors()

    @staticmethod
    def _neutral_priors():
        return {i: {"good": 0.5, "evil": 0.5} for i in range(1, NUM_PLAYERS + 1)}

    def construct(self, **encoder_kwargs):
        self.encoder = ProposalSetDistribution(**encoder_kwargs)
        self.encoder.eval()

    def load_from_file(self, folder_path="v4_trackA/"):
        """Load weights. Mirrors v2's path convention ('our/models/<folder>')
        but also accepts a direct directory or file path."""
        candidates = [
            folder_path,
            os.path.join("our/models", folder_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", folder_path),
        ]
        for base in candidates:
            path = base if base.endswith(".pth") else os.path.join(
                base, "card_model_calibrated.pth")
            if os.path.exists(path):
                self.encoder, self.temperature = load_card_model(path)
                return
        raise ValueError(f"No card model checkpoint found for '{folder_path}'")

    def evidence_probs(self, cards):
        """Per-player calibrated P(evil) from the card encoder alone
        (no constraint, no priors). Returns a [6] float list."""
        if self.encoder is None:
            raise ValueError("Model not constructed/loaded")
        batch_cards, batch_mask = [], []
        for target in range(NUM_PLAYERS):
            feats, mask = cards_to_tensor(cards, target)
            batch_cards.append(feats)
            batch_mask.append(mask)
        with torch.no_grad():
            logits = self.encoder(torch.stack(batch_cards),
                                  torch.stack(batch_mask))
            probs = torch.sigmoid(logits.squeeze(-1) / self.temperature)
        return probs.tolist()

    def predict_probs(self, game_state, self_role=None, self_index=None,
                      algorithm="sum"):
        """game_state: canonical card list (see proposal_cards).

        self_role/self_index condition on the agent's own known alignment
        (pass None for an outside-observer posterior). Returns
        {1: {'good': g, 'evil': e}, ...} like FactorGraphModelV2.
        """
        assert algorithm in ("max", "sum")
        evidence = self.evidence_probs(game_state)
        return self.constrained_posterior(
            evidence, self_role=self_role, self_index=self_index,
            algorithm=algorithm, priors=self._priors)

    def pair_posterior(self, game_state, self_role=None, self_index=None):
        """{(i, j): P(pair)} over the valid evil pairs, ego-conditioned."""
        evidence = self.evidence_probs(game_state)
        weights = self._pair_weights(
            evidence, self_role=self_role, self_index=self_index,
            priors=self._priors)
        total = sum(weights.values())
        if total <= 0:
            return {pair: 1.0 / len(weights) for pair in weights}
        return {pair: w / total for pair, w in weights.items()}

    @staticmethod
    def _pair_weights(evidence, self_role=None, self_index=None, priors=None):
        if priors is None:
            priors = FactorGraphModelV3._neutral_priors()
        known_evil = known_good = None
        if self_role is not None and self_index is not None:
            if _role_name(self_role) == "EVIL":
                known_evil = self_index
            elif _role_name(self_role) == "GOOD":
                known_good = self_index
            else:
                raise ValueError("Role must be either good or evil")

        weights = {}
        for pair in EVIL_PAIRS:
            if known_evil is not None and known_evil not in pair:
                continue
            if known_good is not None and known_good in pair:
                continue
            w = 1.0
            for t in range(NUM_PLAYERS):
                p_evil = max(min(evidence[t], 1.0 - 1e-9), 1e-9)
                prior = priors[t + 1]
                if t in pair:
                    w *= p_evil * prior["evil"]
                else:
                    w *= (1.0 - p_evil) * prior["good"]
            weights[pair] = w
        return weights

    @staticmethod
    def constrained_posterior(evidence, self_role=None, self_index=None,
                              algorithm="sum", priors=None):
        """Exact per-player posterior under the exactly-2-evil constraint.

        evidence: [6] per-player P(evil) values from any detector — the
        replay harness also pushes other models' outputs through this same
        constraint layer for controlled comparisons.
        """
        weights = FactorGraphModelV3._pair_weights(
            evidence, self_role=self_role, self_index=self_index, priors=priors)

        results = {}
        for t in range(NUM_PLAYERS):
            in_t = [w for pair, w in weights.items() if t in pair]
            out_t = [w for pair, w in weights.items() if t not in pair]
            if algorithm == "sum":
                evil_mass, good_mass = sum(in_t), sum(out_t)
            else:
                evil_mass = max(in_t) if in_t else 0.0
                good_mass = max(out_t) if out_t else 0.0
            total = evil_mass + good_mass
            if total <= 0.0:
                evil_prob = 0.5
            else:
                evil_prob = evil_mass / total
            results[t + 1] = {"good": 1.0 - evil_prob, "evil": evil_prob}
        return results

    def update_priors(self, priors):
        """priors: {1: {'evil': e, 'good': g}, ...} — same contract as v2
        (the LLM-vibes prior reweighting); persists until overwritten."""
        for index, probs in priors.items():
            self._priors[index] = {"good": probs["good"], "evil": probs["evil"]}
