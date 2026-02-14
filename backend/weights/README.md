# Model weights

Place model weight files here. The app expects:

| File | Used by | Description |
|------|---------|-------------|
| `body_part_dinov2_head.pt` | BodyPartModel | DINOv2 classifier head for body part (MURA: elbow, finger, forearm, hand, humerus, shoulder, wrist). Create dummy with `python scripts/create_dummy_weights.py`; train on MURA with `python scripts/train_body_part.py --data_dir path/to/MURA`. |
| `body_part.pt` | (legacy) | Old TorchScript body-part model; no longer used. |
| `fracture.pt` | FractureModel | Fracture detection (TorchScript). |
| `bone_age.pt` | BoneAgeModel | Bone age estimation for hand X-rays (TorchScript). |

**Body part (DINOv2):** `body_part_dinov2_head.pt` is a checkpoint dict with `state_dict` (Linear head) and `config` (hidden_size, num_classes, backbone). The backbone is loaded from Hugging Face at runtime (`facebook/dinov2-base`).

**Other models:** Export with `torch.jit.trace()` / `torch.jit.save()` so they load with `torch.jit.load()`.
