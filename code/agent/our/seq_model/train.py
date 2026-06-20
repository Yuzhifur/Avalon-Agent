# seq_model/train.py
#
# Track B training (Workstream 3): cross-entropy on the true evil pair over
# all vote-level prefixes, every ego seat per snapshot (the rolled encoding
# makes that the rotation augmentation for free). Reuses the card tensor
# cache built by train_track_a.ensure_cache — each cached row is already one
# (snapshot, ego) example; this trainer only adds the pair label from the
# corpus and the score/role step features derived from the encoded bits.
#
# Usage (dev data):
#   python -m seq_model.train --corpus ../../../../data/corpora/avalonlogs.jsonl \
#       --splits ../../../evaluation/offline/splits/avalonlogs.json
#
# Models -> our/models/seq_v1/.

import argparse
import copy
import json
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_OUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_DIR = os.path.join(_OUR_DIR, "training")
_DATASET_DIR = os.path.join(_TRAIN_DIR, "dataset")
for p in (_OUR_DIR, _TRAIN_DIR, _DATASET_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

REPO_ROOT = os.path.abspath(os.path.join(_OUR_DIR, "..", "..", ".."))

from seq_model.model import (  # noqa: E402
    CardSeqGRU, EVIL_PAIRS, rolled_pair_label, save_seq_model,
    steps_from_encoded,
)
from seq_model.calibrate import fit_temperature  # noqa: E402
from train_track_a import ensure_cache  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(_OUR_DIR, "models", "seq_v1")
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "data", "corpora", "avalonlogs.jsonl")
DEFAULT_SPLITS = os.path.join(
    REPO_ROOT, "code", "evaluation", "offline", "splits", "avalonlogs.json")
DEFAULT_CACHE = os.path.join(REPO_ROOT, "data", "datasets", "avalonlogs_cards")


def load_split_for_seq(cache_dir, corpus_path, splits_path, split):
    """TensorDataset of (cards u8, mask u8, ego_evil u8, pair_label i64)."""
    from event_schema import read_jsonl

    data = torch.load(os.path.join(cache_dir, f"{split}.pt"), weights_only=True)
    with open(splits_path, encoding="utf-8") as fh:
        split_ids = json.load(fh)[split]
    by_id = {r.game_id: r for r in read_jsonl(corpus_path)}
    records = [by_id[g] for g in split_ids if g in by_id]

    game_idx = data["game_idx"].long()
    target_idx = data["target_idx"].long()
    pair_labels = torch.empty(len(game_idx), dtype=torch.long)
    evil_pairs = []
    for record in records:
        evil_pairs.append([s for s, e in enumerate(record.evil) if e])
    for i in range(len(game_idx)):
        pair_labels[i] = rolled_pair_label(
            evil_pairs[game_idx[i]], int(target_idx[i]))

    return TensorDataset(data["cards"], data["mask"], data["label"],
                         pair_labels)


def batch_to_inputs(cards_u8, mask_u8, ego_evil_u8):
    cards = cards_u8.float()
    mask = mask_u8.float()
    return steps_from_encoded(cards, mask, ego_evil_u8.float())


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total = 0
    all_logits, all_labels = [], []
    for cards, mask, ego_evil, pair_label in loader:
        steps, lengths = batch_to_inputs(cards, mask, ego_evil)
        logits = model(steps, lengths)
        loss = criterion(logits, pair_label)
        total_loss += loss.item() * pair_label.shape[0]
        total_correct += (logits.argmax(dim=1) == pair_label).sum().item()
        total += pair_label.shape[0]
        all_logits.append(logits)
        all_labels.append(pair_label)
    return (total_loss / total, total_correct / total,
            torch.cat(all_logits), torch.cat(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-from", default=None,
                        help="warm-start checkpoint (.pth) for fine-tuning")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ensure_cache(args.corpus, args.splits, args.cache_dir)

    train_set = load_split_for_seq(
        args.cache_dir, args.corpus, args.splits, "train")
    val_set = load_split_for_seq(
        args.cache_dir, args.corpus, args.splits, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"train examples: {len(train_set)}, val examples: {len(val_set)}")

    config = {"hidden": args.hidden, "layers": args.layers,
              "dropout": args.dropout}
    model = CardSeqGRU(**config)
    if args.init_from:
        from seq_model.model import load_seq_model
        model, _ = load_seq_model(args.init_from)
        print(f"warm-started from {args.init_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CardSeqGRU parameters: {n_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_correct = total = 0
        for cards, mask, ego_evil, pair_label in train_loader:
            steps, lengths = batch_to_inputs(cards, mask, ego_evil)
            logits = model(steps, lengths)
            loss = criterion(logits, pair_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pair_label.shape[0]
            total_correct += (logits.argmax(dim=1) == pair_label).sum().item()
            total += pair_label.shape[0]
        train_loss = total_loss / total
        train_acc = total_correct / total

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        print(f"epoch {epoch + 1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"pair_top1 train={train_acc:.4f} val={val_acc:.4f}", flush=True)
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch + 1} "
                  f"(best epoch {best_epoch + 1}, val_loss {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)

    val_loss, val_acc, val_logits, val_labels = evaluate(
        model, val_loader, criterion)
    temperature = fit_temperature(val_logits, val_labels)

    os.makedirs(args.out_dir, exist_ok=True)
    save_seq_model(model, os.path.join(args.out_dir, "seq_model.pth"),
                   temperature=1.0, config=config)
    save_seq_model(model,
                   os.path.join(args.out_dir, "seq_model_calibrated.pth"),
                   temperature=temperature, config=config)

    with open(os.path.join(args.out_dir, "training_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_params": n_params,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "val_pair_top1": val_acc,
            "chance_pair_top1": 1.0 / len(EVIL_PAIRS),
            "temperature": temperature,
            "history": history,
        }, f, indent=1)

    print(f"saved Track B weights to {args.out_dir} "
          f"(best val_loss {best_val_loss:.4f}, pair_top1 {val_acc:.4f}, "
          f"T={temperature:.3f})")


if __name__ == "__main__":
    main()
