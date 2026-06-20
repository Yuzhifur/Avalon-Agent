# seq_belief_model.py
#
# Track B runtime wrapper (Workstream 3): SeqBeliefModel implements the same
# surface as FactorGraphModelV2/V3 (construct / load_from_file /
# predict_probs / update_priors) over the CardSeqGRU 15-way pair head, so
# the model factory can swap it in without touching ACLAgent.
#
# game_state is the canonical card list from proposal_cards. The ego role
# enters both as an input feature and as an inference-time pair mask;
# update_priors (LLM vibes) reweights the pair distribution by the member
# prior odds — the same mechanism as Track A's prior reweighting.

import os

import torch

try:
    from .proposal_cards import NUM_PLAYERS, cards_to_tensor
    from .seq_model.model import (
        EVIL_PAIRS, load_seq_model, steps_from_encoded, unroll_pair,
    )
except ImportError:  # imported top-level by training/eval scripts
    from proposal_cards import NUM_PLAYERS, cards_to_tensor
    from seq_model.model import (
        EVIL_PAIRS, load_seq_model, steps_from_encoded, unroll_pair,
    )


def _role_name(self_role):
    name = getattr(self_role, "name", self_role)
    return str(name).upper()


class SeqBeliefModel:
    def __init__(self):
        self.model = None
        self.temperature = 1.0
        self._priors = self._neutral_priors()

    @staticmethod
    def _neutral_priors():
        return {i: {"good": 0.5, "evil": 0.5} for i in range(1, NUM_PLAYERS + 1)}

    def construct(self, **kwargs):
        from .seq_model.model import CardSeqGRU
        self.model = CardSeqGRU(**kwargs)
        self.model.eval()

    def load_from_file(self, folder_path="seq_v1/"):
        candidates = [
            folder_path,
            os.path.join("our/models", folder_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", folder_path),
        ]
        for base in candidates:
            path = base if base.endswith(".pth") else os.path.join(
                base, "seq_model_calibrated.pth")
            if os.path.exists(path):
                self.model, self.temperature = load_seq_model(path)
                return
        raise ValueError(f"No seq model checkpoint found for '{folder_path}'")

    def _pair_distribution(self, cards, self_role=None, self_index=None):
        """{original-frame pair: probability} after role mask and priors."""
        if self.model is None:
            raise ValueError("Model not constructed/loaded")
        ego = self_index or 0
        role = None if self_role is None else _role_name(self_role)
        ego_is_evil = 1.0 if role == "EVIL" else 0.0

        feats, mask = cards_to_tensor(cards, ego)
        steps, lengths = steps_from_encoded(
            feats.unsqueeze(0), mask.unsqueeze(0),
            torch.tensor([ego_is_evil]))
        with torch.no_grad():
            logits = self.model(steps, lengths).squeeze(0) / self.temperature
            probs = torch.softmax(logits, dim=-1)

        dist = {}
        for i, rolled in enumerate(EVIL_PAIRS):
            pair = unroll_pair(rolled, ego)
            weight = float(probs[i])
            # inference-time role mask
            if role == "GOOD" and self_index in pair:
                weight = 0.0
            elif role == "EVIL" and self_index not in pair:
                weight = 0.0
            # LLM-vibes prior reweighting by member prior odds
            for member in pair:
                prior = self._priors[member + 1]
                weight *= max(prior["evil"], 1e-9) / max(prior["good"], 1e-9)
            dist[pair] = dist.get(pair, 0.0) + weight
        total = sum(dist.values())
        if total <= 0:
            valid = [p for p in dist]
            return {p: 1.0 / len(valid) for p in valid}
        return {p: w / total for p, w in dist.items()}

    def predict_probs(self, game_state, self_role=None, self_index=None,
                      algorithm="sum"):
        """Per-player marginals from the pair distribution (algorithm kept
        for interface compatibility; the pair head is already joint)."""
        dist = self._pair_distribution(game_state, self_role, self_index)
        results = {}
        for t in range(NUM_PLAYERS):
            evil = sum(w for pair, w in dist.items() if t in pair)
            results[t + 1] = {"good": 1.0 - evil, "evil": evil}
        return results

    def pair_posterior(self, game_state, self_role=None, self_index=None):
        return self._pair_distribution(game_state, self_role, self_index)

    def update_priors(self, priors):
        """Same contract as FactorGraphModelV2 — persists until overwritten."""
        for index, probs in priors.items():
            self._priors[index] = {"good": probs["good"], "evil": probs["evil"]}
