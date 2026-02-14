"""
Body part recognition using DINOv2 backbone + trained classifier head.
Labels follow MURA dataset: elbow, finger, forearm, hand, humerus, shoulder, wrist.
"""
import torch
import numpy as np
from PIL import Image

# Lazy imports so inference path only loads transformers when needed
def _load_backbone_and_processor():
    from transformers import AutoImageProcessor, AutoModel
    backbone_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(backbone_name)
    model = AutoModel.from_pretrained(backbone_name)
    return model, processor


# MURA anatomical regions (order defines class index)
MURA_LABELS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


class BodyPartModel:
    def __init__(self, head_path="weights/body_part_dinov2_head.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone, self.processor = _load_backbone_and_processor()
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        checkpoint = torch.load(head_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        hidden_size = config.get("hidden_size", 768)
        num_classes = config.get("num_classes", len(MURA_LABELS))
        self.head = torch.nn.Linear(hidden_size, num_classes)
        self.head.load_state_dict(state_dict, strict=True)
        self.head = self.head.to(self.device)
        self.head.eval()

        self.labels = MURA_LABELS

    def predict(self, image):
        """
        image: numpy array (H, W) float in [0,1] or (H, W, 3), or PIL Image.
        Returns: (label: str, confidence: float)
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = Image.fromarray((np.clip(image, 0, 255).astype(np.uint8)))

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.backbone(pixel_values)
            # [CLS] token: (batch, seq_len, hidden) -> (batch, hidden)
            cls_features = outputs.last_hidden_state[:, 0]
            logits = self.head(cls_features)

        prob = torch.softmax(logits, dim=1)
        idx = prob.argmax(dim=1).item()
        return self.labels[idx], prob[0][idx].item()
