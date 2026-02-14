"""
Train the body-part classifier head on top of frozen DINOv2 using the MURA dataset.
Supports MURA v1.1 layout:
  MURA_ROOT/train/<XR_BODYPART>/<patient_id>/<study_id>/image.png
  MURA_ROOT/valid/<XR_BODYPART>/<patient_id>/<study_id>/image.png
Body part folders: XR_ELBOW, XR_FINGER, XR_FOREARM, XR_HAND, XR_HUMERUS, XR_SHOULDER, XR_WRIST.
Also supports 3-level layout: MURA_ROOT/<split>/<BODYPART>/<study>/image.png.

Usage (from backend root):
  python scripts/train_body_part.py --data_dir path/to/MURA-v1.1 [--epochs 5] [--batch_size 32]
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# MURA labels (order = class index); must match models/body_part.MURA_LABELS
MURA_LABELS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]
LABEL_TO_IDX = {l: i for i, l in enumerate(MURA_LABELS)}
WEIGHTS_DIR = "weights"
HEAD_PATH = os.path.join(WEIGHTS_DIR, "body_part_dinov2_head.pt")


def discover_mura_images(root, split):
    """Collect (image_path, body_part_index). split in ('train', 'valid').
    Supports MURA v1.1: XR_ELBOW, XR_FINGER, etc. (strips XR_ prefix).
    Recursively finds images under each body-part folder (handles 4-level layout)."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []
    pairs = []
    for part_dir in os.listdir(split_dir):
        part_path = os.path.join(split_dir, part_dir)
        if not os.path.isdir(part_path):
            continue
        part_lower = part_dir.lower()
        if part_lower.startswith("xr_"):
            part_lower = part_lower[3:]
        if part_lower not in LABEL_TO_IDX:
            continue
        idx = LABEL_TO_IDX[part_lower]
        for dirpath, _, filenames in os.walk(part_path):
            for fname in filenames:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    pairs.append((os.path.join(dirpath, fname), idx))
    return pairs


class MURABodyPartDataset(Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, label = self.pairs[i]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label


def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 body-part head on MURA")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MURA root (contains train/ and valid/)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import AutoImageProcessor, AutoModel
    backbone_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(backbone_name)
    backbone = AutoModel.from_pretrained(backbone_name)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    hidden_size = 768
    num_classes = len(MURA_LABELS)
    head = nn.Linear(hidden_size, num_classes).to(device)

    train_pairs = discover_mura_images(args.data_dir, "train")
    valid_pairs = discover_mura_images(args.data_dir, "valid")
    if not train_pairs:
        raise SystemExit("No training images found. Check --data_dir and MURA layout: <data_dir>/train/<BODYPART>/<study>/image.png")
    print(f"Train images: {len(train_pairs)}, Valid images: {len(valid_pairs)}")

    train_ds = MURABodyPartDataset(train_pairs, processor)
    valid_ds = MURABodyPartDataset(valid_pairs, processor) if valid_pairs else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            with torch.no_grad():
                out = backbone(pixel_values)
                features = out.last_hidden_state[:, 0]
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} train loss: {total_loss / n_batches:.4f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    checkpoint = {
        "state_dict": head.state_dict(),
        "config": {"hidden_size": hidden_size, "num_classes": num_classes, "backbone": backbone_name},
    }
    torch.save(checkpoint, HEAD_PATH)
    print(f"Saved head and config to {HEAD_PATH}")


if __name__ == "__main__":
    main()
