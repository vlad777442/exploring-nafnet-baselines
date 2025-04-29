import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
from corrupted_dataset import CorruptedCIFAR
from blocks import NAFBlock_Baseline, NAFBlock_A1, NAFBlock_A2, NAFBlock_A3, NAFBlock_A4
from tqdm import tqdm
import lpips
import torchvision.transforms.functional as TF
import torchmetrics
import csv
import time


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

ssim_metric = torchmetrics.image.ssim.StructuralSimilarityIndexMeasure(data_range=1.0)

def compute_ssim(pred, target):
    return ssim_metric(pred, target).item()

lpips_fn = lpips.LPIPS(net='alex')


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    lpips_fn.to(device)
    ssim_metric.to(device)

    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)

            
            psnr_val = psnr(out, y)
            psnr_scores.append(psnr_val.item())

           
            ssim_val = compute_ssim(out, y)
            ssim_scores.append(ssim_val)

            # LPIPS (normalize to [-1, 1])
            out_norm = (out * 2) - 1
            y_norm = (y * 2) - 1
            lpips_val = lpips_fn(out_norm, y_norm).mean()
            lpips_scores.append(lpips_val.item())

    return (
        sum(psnr_scores) / len(psnr_scores),
        sum(ssim_scores) / len(ssim_scores),
        sum(lpips_scores) / len(lpips_scores)
    )



class SimpleNAFNet(nn.Module):
    def __init__(self, block, num_blocks=4, channels=32):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[block(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return torch.clamp(x, 0, 1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = CorruptedCIFAR(train=True)
    test_set = CorruptedCIFAR(train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    block_map = {
        "baseline": NAFBlock_Baseline,
        "a1": NAFBlock_A1,
        "a2": NAFBlock_A2,
        "a3": NAFBlock_A3,
        "a4": NAFBlock_A4
    }
    block = block_map[args.variant.lower()]
    model = SimpleNAFNet(block).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    os.makedirs(args.output, exist_ok=True)
    start_time = time.time()

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {loss:.4f}")

    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    psnr_score, ssim_score, lpips_score = evaluate(model, test_loader, device)
    print(f"PSNR: {psnr_score:.2f} dB | SSIM: {ssim_score:.4f} | LPIPS: {lpips_score:.4f}")

    csv_path = os.path.join(args.output, "results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Variant", "PSNR (dB)", "SSIM", "LPIPS"])  
        writer.writerow([args.variant, f"{psnr_score:.2f}", f"{ssim_score:.4f}", f"{lpips_score:.4f}"])

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            out = model(x)
            save_image(out, os.path.join(args.output, f"{args.variant}_denoised_{i}.png"))
            save_image(x, os.path.join(args.output, f"{args.variant}_noisy_{i}.png"))
            if i >= 4:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="baseline", help="baseline | a1 | a2 | a3 | a4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="./results")
    args = parser.parse_args()
    main(args)
