"""Snapshot enumeration and training-example materialization (Workstream 1).

Both model tracks train on the same proposal-boundary snapshots: one example
after every VoteEvent (future cards masked out) plus the post-quest-resolution
state — produced by proposal_cards.enumerate_snapshots from the shared card
stream. This supersedes the legacy partialize_vector: mid-quest rejected
proposals are exactly the new signal.

Seat augmentation: each snapshot yields 6 examples, one per target player
(label = that player's true alignment), encoded in the target-centric frame.
Rolling the target across all six seats IS the legacy ×6 cyclic-permutation
augmentation — rotating every seat by k and putting the target at index 0 is
identical to keeping seats fixed and choosing target k — so no separate
record-rotation pass exists.

`build_card_dataset` materializes the uint8 tensor dict consumed by
train_track_a.py (and whose snapshot table Track B's sequence trainer reuses).
"""

import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _OUR = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if _OUR not in sys.path:
        sys.path.insert(0, _OUR)
    from proposal_cards import (
        NUM_PLAYERS, NUM_SLOTS, cards_from_events, cards_to_tensor,
        enumerate_snapshots, v2_vector_from_cards,
    )
else:
    from ...proposal_cards import (
        NUM_PLAYERS, NUM_SLOTS, cards_from_events, cards_to_tensor,
        enumerate_snapshots, v2_vector_from_cards,
    )

EXPECTED_TEAM_SIZES = {2, 3, 4}


def record_to_cards(record):
    """GameRecord -> canonical cards, with corpus-level sanity checks."""
    cards = cards_from_events(record.events)
    if len(cards) > NUM_SLOTS:
        raise ValueError("more voted proposals than fixed slots")
    for card in cards:
        if len(card["team"]) not in EXPECTED_TEAM_SIZES:
            raise ValueError(f"unexpected team size {len(card['team'])}")
    if sum(record.evil) != 2:
        raise ValueError(f"expected exactly 2 evil players, got {sum(record.evil)}")
    return cards


def snapshots_from_record(record):
    """[(snapshot_cards, is_quest_boundary)] for one GameRecord."""
    return enumerate_snapshots(record_to_cards(record))


def build_card_dataset(records, desc=None, progress=True):
    """Materialize the per-target card tensors for a list of GameRecords.

    Returns a dict of stacked tensors (uint8 where possible):
      cards [N, 25, CARD_DIM], mask [N, 25], label [N], game_idx [N],
      target_idx [N], n_cards [N], quest_boundary [N], v2_vector [N, 21]
    The v2 vector of each snapshot is stored so controlled same-split v2
    comparisons can score identical states.
    """
    iterator = enumerate(records)
    if progress:
        from tqdm import tqdm
        iterator = enumerate(tqdm(records, desc=desc or "vectorizing"))

    feats_list, mask_list = [], []
    labels, game_idx, target_idx = [], [], []
    n_cards_list, quest_boundary, v2_vectors = [], [], []

    for g_i, record in iterator:
        for snapshot, is_boundary in snapshots_from_record(record):
            v2_vec = v2_vector_from_cards(snapshot)
            for target in range(NUM_PLAYERS):
                feats, mask = cards_to_tensor(snapshot, target)
                feats_list.append((feats > 0.5).to(torch.uint8))
                mask_list.append((mask > 0.5).to(torch.uint8))
                labels.append(record.evil[target])
                game_idx.append(g_i)
                target_idx.append(target)
                n_cards_list.append(len(snapshot))
                quest_boundary.append(int(is_boundary))
                v2_vectors.append(v2_vec)

    return {
        "cards": torch.stack(feats_list),
        "mask": torch.stack(mask_list),
        "label": torch.tensor(labels, dtype=torch.uint8),
        "game_idx": torch.tensor(game_idx, dtype=torch.int32),
        "target_idx": torch.tensor(target_idx, dtype=torch.uint8),
        "n_cards": torch.tensor(n_cards_list, dtype=torch.uint8),
        "quest_boundary": torch.tensor(quest_boundary, dtype=torch.uint8),
        "v2_vector": torch.tensor(v2_vectors, dtype=torch.int16),
    }
