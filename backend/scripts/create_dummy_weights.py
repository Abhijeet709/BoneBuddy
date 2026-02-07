"""
One-off script to create dummy TorchScript weights so the app can start.
Run from project root: python scripts/create_dummy_weights.py
"""
import torch
import torch.nn as nn

WEIGHTS_DIR = "weights"


class DummyBodyPart(nn.Module):
    """Outputs logits of shape (batch, 5) for body part classes."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 7, stride=32)
        self.fc = nn.Linear(4 * 7 * 7, 5)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class DummyFracture(nn.Module):
    """Outputs a single value (batch, 1) for fracture probability."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 7, stride=32)
        self.fc = nn.Linear(4 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class DummyBoneAge(nn.Module):
    """Outputs a single value (batch, 1) for bone age in months."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 7, stride=32)
        self.fc = nn.Linear(4 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def main():
    # Input shape matches: ToTensor()(image).unsqueeze(0) -> (1, 1, 224, 224)
    example = torch.rand(1, 1, 224, 224)

    for name, model_class, out_path in [
        ("body_part", DummyBodyPart, f"{WEIGHTS_DIR}/body_part.pt"),
        ("fracture", DummyFracture, f"{WEIGHTS_DIR}/fracture.pt"),
        ("bone_age", DummyBoneAge, f"{WEIGHTS_DIR}/bone_age.pt"),
    ]:
        model = model_class()
        model.eval()
        traced = torch.jit.trace(model, example)
        traced.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
