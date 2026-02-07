# Model weights

Place TorchScript (`.pt`) model files here. The app expects:

| File            | Used by        | Description                    |
|-----------------|----------------|--------------------------------|
| `body_part.pt`  | BodyPartModel  | Body part classifier (hand, elbow, shoulder, knee, ankle) |
| `fracture.pt`   | FractureModel  | Fracture detection             |
| `bone_age.pt`   | BoneAgeModel   | Bone age estimation (hand X-rays) |

Export your trained PyTorch models with `torch.jit.save()` or `torch.jit.trace()` + `model.save()` so they can be loaded with `torch.jit.load()`.
