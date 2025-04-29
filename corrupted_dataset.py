import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from PIL import Image, ImageFilter
import random
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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
    
    def visualize_corruptions(dataset, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
        
        for i in range(num_samples):
            idx = random.randint(0, len(dataset)-1)
            noisy, clean = dataset[idx]
            
            noisy_img = T.ToPILImage()(noisy)
            clean_img = T.ToPILImage()(clean)
            
            axes[i, 0].imshow(clean_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(noisy_img)
            axes[i, 1].set_title("Corrupted")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('corruption_examples.png')
        plt.show()