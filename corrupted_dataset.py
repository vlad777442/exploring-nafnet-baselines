import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from PIL import Image, ImageFilter
import random
import torch
from torchvision.utils import save_image


class CorruptedCIFAR(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.base = CIFAR10(root='./data', download=True, train=train)
        self.transform = T.ToTensor()
        self.blur = lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        img = img.convert("RGB")
        corrupted = self.blur(img)
        corrupted = T.functional.adjust_brightness(corrupted, 1 + random.uniform(-0.1, 0.1))
        clean = self.transform(img)
        noisy = self.transform(corrupted)
        return noisy, clean

    def __len__(self):
        return len(self.base)