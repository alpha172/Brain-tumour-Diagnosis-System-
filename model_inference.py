"""Inference utilities for the trained NeuroVision 3D U-Net checkpoint."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CompactUNet3D(nn.Module):
    """Matches layer names and tensor shapes in the supplied epoch-006 checkpoint."""
    def __init__(self, in_channels=4, num_classes=4, base=8):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base * 4, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def load_trained_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model = CompactUNet3D(in_channels=4, num_classes=4, base=8)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def normalize_nonzero(volume):
    volume = np.asarray(volume, dtype=np.float32)
    mask = volume != 0
    result = np.zeros_like(volume, dtype=np.float32)
    if mask.any():
        values = volume[mask]
        std = float(values.std())
        result[mask] = (values - float(values.mean())) / std if std > 1e-6 else values - float(values.mean())
    return result


def _window_starts(length, patch, stride):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def predict_volume(model, volumes, device, patch_size=96, overlap=0.5):
    """Predict aligned volumes ordered as FLAIR, T1, T1ce, T2.

    Encoded outputs: 0 background, 1 BraTS label 1, 2 BraTS label 2,
    3 BraTS label 4. Confirm label encoding against the training script.
    """
    order = ("FLAIR", "T1", "T1ce", "T2")
    arrays = [np.asarray(volumes[k], dtype=np.float32) for k in order]
    shapes = [a.shape for a in arrays]
    if any(a.ndim != 3 for a in arrays) or len(set(shapes)) != 1:
        raise ValueError(f"All modalities must be aligned 3D arrays; got shapes {shapes}.")
    original_shape = shapes[0]
    data = np.stack([normalize_nonzero(a) for a in arrays], axis=0)
    padded_shape = tuple(max(patch_size, int(np.ceil(n / 8) * 8)) for n in original_shape)
    pad = [(0, padded_shape[i] - original_shape[i]) for i in range(3)]
    data = np.pad(data, [(0, 0), *pad], mode="constant")
    stride = max(1, int(patch_size * (1.0 - overlap)))
    starts = [_window_starts(n, patch_size, stride) for n in padded_shape]
    logits_sum = np.zeros((4, *padded_shape), dtype=np.float32)
    counts = np.zeros(padded_shape, dtype=np.float32)
    with torch.inference_mode():
        for i in starts[0]:
            for j in starts[1]:
                for k in starts[2]:
                    patch = np.ascontiguousarray(data[:, i:i+patch_size, j:j+patch_size, k:k+patch_size])
                    x = torch.from_numpy(patch).unsqueeze(0).to(device)
                    logits = model(x)
                    logits_sum[:, i:i+patch_size, j:j+patch_size, k:k+patch_size] += logits[0].float().cpu().numpy()
                    counts[i:i+patch_size, j:j+patch_size, k:k+patch_size] += 1.0
    logits_sum /= np.maximum(counts[None, ...], 1.0)
    return logits_sum.argmax(axis=0).astype(np.uint8)[:original_shape[0], :original_shape[1], :original_shape[2]]
