# seq_model/model.py
#
# Track B architecture (Workstream 3): each proposal card (ego-rolled, the
# same encoding Track A consumes) plus a running-score field and the ego-role
# flag is linearly projected to a 128-d step embedding; a 2-layer GRU
# (hidden 128, ~250k params — sized for a tens-of-thousands-of-games regime)
# encodes the sequence; the final hidden state feeds a 15-way softmax over
# the evil-pair hypotheses (structural exactly-2-evil; per-player marginal =
# sum of pairs containing the player).
#
# Step features are derived FROM the encoded card tensor (accepted/outcome
# bits), so the dataset cache built for Track A serves both tracks and the
# runtime wrapper builds identical inputs from the live card stream.

from itertools import combinations
import os

import torch
import torch.nn as nn

try:
    from ..proposal_cards import CARD_DIM, NUM_PLAYERS
except ImportError:  # imported top-level by training/eval scripts
    from proposal_cards import CARD_DIM, NUM_PLAYERS

EVIL_PAIRS = list(combinations(range(NUM_PLAYERS), 2))  # rolled-frame pairs
PAIR_INDEX = {pair: i for i, pair in enumerate(EVIL_PAIRS)}

# step layout: card (CARD_DIM) + [good_wins/3, evil_wins/3] + [ego_good, ego_evil]
STEP_DIM = CARD_DIM + 4
COL_ACCEPTED = 18
COL_OUTCOME_FAIL = 32
COL_OUTCOME_SUCCESS = 33


def rolled_pair_label(evil_seats, ego):
    """True evil pair as a 15-way class index in the ego-rolled frame."""
    rolled = tuple(sorted((s - ego) % NUM_PLAYERS for s in evil_seats))
    return PAIR_INDEX[rolled]


def unroll_pair(pair, ego):
    """Rolled-frame pair -> original seat numbering."""
    return tuple(sorted((p + ego) % NUM_PLAYERS for p in pair))


def steps_from_encoded(cards, mask, ego_is_evil):
    """Build GRU step tensors from encoded card tensors.

    cards: [..., S, CARD_DIM] float (ego-rolled, slot-padded)
    mask:  [..., S] float
    ego_is_evil: scalar/tensor broadcastable to [...] (1.0 = ego evil)

    Returns (steps [..., S+1, STEP_DIM], lengths [...]) — a leading all-zero
    BOS step keeps the empty history well-defined; real steps carry the
    running score BEFORE the card, derived from the accepted/outcome bits.
    """
    single = cards.dim() == 2
    if single:
        cards = cards.unsqueeze(0)
        mask = mask.unsqueeze(0)
    B, S, _ = cards.shape

    resolved_success = cards[..., COL_OUTCOME_SUCCESS] * mask
    resolved_fail = cards[..., COL_OUTCOME_FAIL] * mask
    # score before card i = outcomes attached to cards 0..i-1
    good_wins = torch.cumsum(resolved_success, dim=1) - resolved_success
    evil_wins = torch.cumsum(resolved_fail, dim=1) - resolved_fail

    ego_evil = torch.as_tensor(ego_is_evil, dtype=cards.dtype,
                               device=cards.device).expand(B)
    role = torch.stack([1.0 - ego_evil, ego_evil], dim=-1)  # [B, 2]
    role_steps = role.unsqueeze(1).expand(B, S, 2)

    feats = torch.cat([
        cards,
        (good_wins / 3.0).unsqueeze(-1),
        (evil_wins / 3.0).unsqueeze(-1),
        role_steps,
    ], dim=-1) * mask.unsqueeze(-1)

    bos = torch.zeros((B, 1, STEP_DIM), dtype=cards.dtype, device=cards.device)
    steps = torch.cat([bos, feats], dim=1)
    lengths = mask.sum(dim=1).long() + 1
    if single:
        return steps.squeeze(0), lengths.squeeze(0)
    return steps, lengths


class CardSeqGRU(nn.Module):
    def __init__(self, step_dim=STEP_DIM, hidden=128, layers=2, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(step_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, len(EVIL_PAIRS))

    def forward(self, steps, lengths):
        """steps: [B, T, STEP_DIM], lengths: [B] -> [B, 15] pair logits."""
        x = torch.relu(self.proj(steps))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])


def save_seq_model(model, path, temperature=1.0, config=None):
    torch.save({
        "state_dict": model.state_dict(),
        "temperature": float(temperature),
        "config": config or {},
    }, path)


def load_seq_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model = CardSeqGRU(
        step_dim=config.get("step_dim", STEP_DIM),
        hidden=config.get("hidden", 128),
        layers=config.get("layers", 2),
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("temperature", 1.0)
