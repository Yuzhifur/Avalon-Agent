# card_encoder.py
#
# Track A encoder (Workstream 2): ProposalSetDistribution — a shared per-card
# MLP, masked mean-pooled over the 25 proposal slots, followed by a small head
# that emits one logit of evidence that the target player (index 0 of the
# egocentric frame) is evil.
#
# One instance is shared across all six per-player factors; the egocentric
# roll in proposal_cards.cards_to_tensor provides the player-specific view,
# mirroring the v2 EgoNeuralDistribution design — but as a separate class so
# the frozen v2 stack (whose model is a class-level singleton) stays intact
# for A/B comparison. Position fields inside each card (quest, proposal index)
# preserve order information under pooling; Track B is the deliberate
# recurrence upgrade, so anything order-related beyond position features is
# out of scope here.

import torch
import torch.nn as nn

try:
    from .proposal_cards import CARD_DIM
except ImportError:  # imported top-level by training/eval scripts
    from proposal_cards import CARD_DIM


class ProposalSetDistribution(nn.Module):
    def __init__(self, card_dim=CARD_DIM, card_hidden=64, card_embed=64,
                 head_hidden=32):
        super().__init__()
        self.card_mlp = nn.Sequential(
            nn.Linear(card_dim, card_hidden),
            nn.ReLU(),
            nn.Linear(card_hidden, card_embed),
            nn.ReLU(),
        )
        # +1: the fraction of filled slots, so mean pooling does not erase
        # how deep into the game (and how reject-heavy) the history is
        self.head = nn.Sequential(
            nn.Linear(card_embed + 1, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, cards, mask):
        """cards: [B, S, CARD_DIM] float, mask: [B, S] float -> [B, 1] logit."""
        h = self.card_mlp(cards) * mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True)
        pooled = h.sum(dim=1) / count.clamp(min=1.0)
        density = count / cards.shape[1]
        return self.head(torch.cat([pooled, density], dim=1))


class TemperatureScaledModel(nn.Module):
    """Temperature wrapper matching the two-argument forward.

    (The legacy ModelWithTemperature wraps single-input models, so the card
    net gets its own thin equivalent.)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, cards, mask):
        return self.model(cards, mask) / self.temperature

    def set_temperature(self, val_loader):
        """Fit T by minimizing binary NLL on a validation loader that yields
        (cards, mask, label) batches."""
        criterion = nn.BCEWithLogitsLoss()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for cards, mask, labels in val_loader:
                logits_list.append(self.model(cards, mask).squeeze(-1))
                labels_list.append(labels.float())
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        before = criterion(logits, labels).item()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        after = criterion(logits / self.temperature, labels).item()
        print(f"Temperature scaling: T={self.temperature.item():.4f} "
              f"NLL {before:.4f} -> {after:.4f}")
        return self


def save_card_model(model, path, temperature=1.0, config=None):
    torch.save({
        "state_dict": model.state_dict(),
        "temperature": float(temperature),
        "config": config or {},
    }, path)


def load_card_model(path):
    """Returns (ProposalSetDistribution, temperature). The checkpoint stores
    the bare state dict plus the fitted temperature scalar."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model = ProposalSetDistribution(
        card_dim=config.get("card_dim", CARD_DIM),
        card_hidden=config.get("card_hidden", 64),
        card_embed=config.get("card_embed", 64),
        head_hidden=config.get("head_hidden", 32),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("temperature", 1.0)
