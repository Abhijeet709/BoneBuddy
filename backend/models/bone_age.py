import torch
import torchvision.transforms as T

class BoneAgeModel:
    def __init__(self, path="weights/bone_age.pt"):
        self.model = torch.jit.load(path)
        self.model.eval()

    def predict(self, image):
        tensor = T.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            age = self.model(tensor)
        return round(age.item(), 2)
