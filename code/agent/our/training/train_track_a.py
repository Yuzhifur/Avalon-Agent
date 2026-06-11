# train_track_a.py
#
# Track A training (Workstream 2): ProposalSetDistribution on proposal-card
# snapshots from a GameRecord corpus, with early stopping on validation loss
# and post-hoc temperature scaling. Saves weights + metrics to
# our/models/v4_trackA/ (or v4_trackA_ft/ for the self-play fine-tune).
#
# Deviation from the v2 recipe, on purpose: no pos_weight in the loss by
# default. The class ratio is a fixed 2 evil : 4 good, and the constraint
# layer multiplies the output probabilities, so native calibration matters
# more than class balance (temperature-only scaling has no shift term and
# cannot undo a pos_weight bias; the prior phase-3 run fitted T = 1.0 exactly
# without it).
#
# Usage (dev data):
#   python train_track_a.py --corpus ../../../../data/corpora/avalonlogs.jsonl \
#       --splits ../../../evaluation/offline/splits/avalonlogs.json
#
# The encoded tensor cache (--cache-dir) is rebuilt automatically when absent.

import argparse
import copy
import json
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_OUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
for p in (_OUR_DIR, _DATASET_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

REPO_ROOT = os.path.abspath(os.path.join(_OUR_DIR, "..", "..", ".."))

from card_encoder import (  # noqa: E402
    ProposalSetDistribution,
    TemperatureScaledModel,
    save_card_model,
)
from proposal_cards import FEATURE_ABLATIONS, apply_feature_ablation  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(_OUR_DIR, "models", "v4_trackA")
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "data", "corpora", "avalonlogs.jsonl")
DEFAULT_SPLITS = os.path.join(
    REPO_ROOT, "code", "evaluation", "offline", "splits", "avalonlogs.json")
DEFAULT_CACHE = os.path.join(REPO_ROOT, "data", "datasets", "avalonlogs_cards")


def ensure_cache(corpus_path, splits_path, cache_dir):
    """Materialize per-split card tensors from the corpus once."""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if (manifest.get("corpus") == os.path.abspath(corpus_path)
                and manifest.get("splits") == os.path.abspath(splits_path)):
            return manifest
    from event_schema import read_jsonl
    from augment import build_card_dataset

    with open(splits_path, encoding="utf-8") as fh:
        splits = json.load(fh)
    records = list(read_jsonl(corpus_path))
    by_id = {r.game_id: r for r in records}
    os.makedirs(cache_dir, exist_ok=True)
    manifest = {
        "corpus": os.path.abspath(corpus_path),
        "splits": os.path.abspath(splits_path),
        "seed": splits.get("seed"),
        "splits_info": {},
    }
    for split in ("train", "val", "test"):
        split_records = [by_id[g] for g in splits[split] if g in by_id]
        tensors = build_card_dataset(split_records, desc=f"vectorizing {split}")
        torch.save(tensors, os.path.join(cache_dir, f"{split}.pt"))
        manifest["splits_info"][split] = {
            "n_games": len(split_records),
            "n_examples": int(tensors["label"].shape[0]),
            "n_evil_examples": int(tensors["label"].sum().item()),
        }
        print(f"{split}: {len(split_records)} games, "
              f"{tensors['label'].shape[0]} examples")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def load_split(cache_dir, split, feature_ablation="none"):
    data = torch.load(os.path.join(cache_dir, f"{split}.pt"), weights_only=True)
    cards = data["cards"].float()
    mask = data["mask"].float()
    cards, mask = apply_feature_ablation(cards, mask, feature_ablation)
    labels = data["label"].float()
    return TensorDataset(cards, mask, labels)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total = 0
    for cards, mask, labels in loader:
        logits = model(cards, mask).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--card-hidden", type=int, default=64)
    parser.add_argument("--card-embed", type=int, default=64)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--feature-ablation", choices=FEATURE_ABLATIONS,
                        default="none")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="optional BCE pos_weight; OFF by default — see "
                             "the calibration note in the header")
    parser.add_argument("--init-from", default=None,
                        help="warm-start checkpoint (.pth) for fine-tuning")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cache_manifest = ensure_cache(args.corpus, args.splits, args.cache_dir)

    train_set = load_split(args.cache_dir, "train", args.feature_ablation)
    val_set = load_split(args.cache_dir, "val", args.feature_ablation)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"train examples: {len(train_set)}, val examples: {len(val_set)}")

    encoder_config = {
        "card_hidden": args.card_hidden,
        "card_embed": args.card_embed,
        "head_hidden": args.head_hidden,
    }
    model = ProposalSetDistribution(**encoder_config)
    if args.init_from:
        from card_encoder import load_card_model
        model, _ = load_card_model(args.init_from)
        print(f"warm-started from {args.init_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProposalSetDistribution parameters: {n_params}")

    pos_weight = (torch.tensor([args.pos_weight])
                  if args.pos_weight else None)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        for cards, mask, labels in train_loader:
            logits = model(cards, mask).squeeze(-1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total += labels.shape[0]
        train_loss = total_loss / total
        train_acc = total_correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)
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

        if (epoch + 1) % 5 == 0 or stale >= args.patience:
            print(f"epoch {epoch + 1}/{args.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch + 1} "
                  f"(best epoch {best_epoch + 1}, val_loss {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)

    # temperature scaling on the validation split
    scaled = TemperatureScaledModel(model)
    scaled.set_temperature(val_loader)
    temperature = scaled.temperature.item()

    os.makedirs(args.out_dir, exist_ok=True)
    save_card_model(model, os.path.join(args.out_dir, "card_model.pth"),
                    temperature=1.0, config=encoder_config)
    save_card_model(model,
                    os.path.join(args.out_dir, "card_model_calibrated.pth"),
                    temperature=temperature, config=encoder_config)

    final_val_loss, final_val_acc = evaluate(model, val_loader,
                                             nn.BCEWithLogitsLoss())
    with open(os.path.join(args.out_dir, "training_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_params": n_params,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "temperature": temperature,
            "cache_manifest": cache_manifest,
            "history": history,
        }, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(history["train_loss"], label="train")
        axs[0].plot(history["val_loss"], label="val")
        axs[0].set_title("BCE loss")
        axs[0].legend()
        axs[1].plot(history["train_acc"], label="train")
        axs[1].plot(history["val_acc"], label="val")
        axs[1].set_title("accuracy")
        axs[1].legend()
        fig.savefig(os.path.join(args.out_dir, "training_curves.png"))
    except Exception as e:  # plotting is best-effort
        print(f"skipping curve plot: {e}")

    print(f"saved Track A weights to {args.out_dir} "
          f"(best val_loss {best_val_loss:.4f}, T={temperature:.3f})")


if __name__ == "__main__":
    main()
