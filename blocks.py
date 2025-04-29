import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCA(nn.Module): 
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class NAFBlock_Baseline(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm([channels, 32, 32])
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.attn = SCA(channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.pw2(x)
        x = self.attn(x)
        return x + residual

class NAFBlock_A1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm([channels, 32, 32])
        self.pw1 = nn.Conv2d(channels, channels, 1) 
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.attn = SCA(channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.attn(x)
        return x + residual


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class NAFBlock_A2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm([channels, 32, 32])
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.attn = ECA(channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.pw2(x)
        x = self.attn(x)
        return x + residual



class NAFBlock_A3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.attn = SCA(channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.pw2(x)
        x = self.attn(x)
        return x + residual

class NAFBlock_A4(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm([channels, 32, 32])
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)
        # No attention module

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.pw2(x)
        return x + residual

