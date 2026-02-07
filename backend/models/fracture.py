import torch
import torchvision.transforms as T

class FractureModel:
    def __init__(self, path="weights/fracture.pt"):
        self.model = torch.jit.load(path)
        self.model.eval()

    def predict(self, image):
        tensor = T.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
        return bool(output.item() > 0.5), output.item()
