import albumentations as albu
from albumentations.pytorch import ToTensor
import numpy as np
BORDER_CONSTANT = 0
image_size = 300

def _corner_padding(x, **kwargs):
    h, w = x.shape[:2]
    outp = np.zeros((image_size, image_size, 3))
    outp[:h, :w] = x
    return outp

def Corner_Pad(image_size=300):
    'Padding to left-upped corner for convenience of working with bboxes'
    _transform = [
        albu.Lambda(image=_corner_padding),
    ]
    return albu.Compose(_transform)

def pre_transforms(image_size=300):
    return albu.Compose([
        albu.LongestMaxSize(max_size=image_size, always_apply=True),
#         albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT, value=0),
        Corner_Pad(image_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ])