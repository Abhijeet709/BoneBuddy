import torch
import torchvision.transforms as T

class BodyPartModel:
    def __init__(self, path="weights/body_part.pt"):
        self.model = torch.jit.load(path)
        self.model.eval()
        self.labels = ["hand", "elbow", "shoulder", "knee", "ankle"]

    def predict(self, image):
        tensor = T.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
        prob = torch.softmax(logits, dim=1)
        idx = prob.argmax().item()
        return self.labels[idx], prob[0][idx].item()
