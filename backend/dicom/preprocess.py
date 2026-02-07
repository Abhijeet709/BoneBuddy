import cv2
import numpy as np

def preprocess_xray(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    image = cv2.resize(image, (224, 224))
    image = (image / 255.0).astype(np.float32)
    return image
