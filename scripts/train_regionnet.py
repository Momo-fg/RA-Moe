import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramoe.datasets.region_dataset import RegionDataset
from ramoe.models.region_classifier import RegionClassifier


def train_single_subject(subject_name, args, transform, device):
    subject_path = Path(args.data_root) / subject_name
    train_full = RegionDataset(str(subject_path), split="train", transform=transform)
    test_set = RegionDataset(str(subject_path), split="test", transform=transform)

    if len(train_full) == 0:
        print(f"Skip {subject_name}: no training data")
        return

    n_total = len(train_full)
    n_extra_val = int(n_total * args.extra_val_ratio)
    n_train = n_total - n_extra_val

    train_dataset, extra_val_dataset = random_split(
        train_full,
        [n_train, n_extra_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_dataset = torch.utils.data.ConcatDataset([test_set, extra_val_dataset])
    print(
        f"Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} "
        f"(test {len(test_set)} + extra {len(extra_val_dataset)})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = RegionClassifier(
        input_size=args.input_size,
        num_regions=args.num_regions,
    ).to(device)

    pos_weight = torch.ones(args.num_regions, device=device) * args.pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for imgs, gts in train_loader:
            imgs = imgs.to(device)
            gts = gts.to(device)

            logits = model(imgs)
            loss = criterion(logits, gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_probs = []
        all_gts = []

        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs = imgs.to(device)
                gts = gts.to(device)

                logits = model(imgs)
                loss = criterion(logits, gts)
                val_loss += loss.item()

                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_gts.append(gts.cpu().numpy())

        val_loss /= len(val_loader)
        all_probs = np.vstack(all_probs)
        all_gts = np.vstack(all_gts)

        bin_preds = (all_probs > 0.5).astype(np.float32)
        nonzero_mask = all_gts.sum(axis=1) > 0
        if nonzero_mask.any():
            f1s = [
                f1_score(gt, pred, zero_division=0)
                for gt, pred in zip(all_gts[nonzero_mask], bin_preds[nonzero_mask])
            ]
            nonzero_f1 = float(np.mean(f1s))
        else:
            nonzero_f1 = 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"region_cls_{subject_name}.pth")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"NonZeroF1 {nonzero_f1:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    print(f"{subject_name} FINAL | Best ValLoss {best_val_loss:.4f} (epoch {best_epoch + 1})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train per-subject RegionNet checkpoints.")
    parser.add_argument("--data-root", required=True, help="CASME II LOSO directory")
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "regionnet"))
    parser.add_argument("--input-size", type=int, default=48)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--extra-val-ratio", type=float, default=0.15)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])

    subjects = [
        d.name for d in Path(args.data_root).iterdir()
        if d.is_dir()
    ]

    for subject in sorted(subjects):
        print("\n" + "=" * 60)
        print(f"Training Region Classifier for {subject}")
        print("=" * 60)
        train_single_subject(subject, args, transform, device)


if __name__ == "__main__":
    main()
