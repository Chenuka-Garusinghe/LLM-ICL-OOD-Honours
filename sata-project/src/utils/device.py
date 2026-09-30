"""Compute device and dtype selection: NVIDIA CUDA, then Apple Metal (MPS), then CPU."""

from __future__ import annotations

import os
import platform
import subprocess

if platform.system() == "Darwin":
    # Must be set before torch is imported: unsupported MPS ops fall back to CPU instead of raising.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the requested device; "auto" picks CUDA, then MPS, then CPU."""
    pref = preference.lower()
    if pref not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {preference!r}; expected one of {DEVICE_CHOICES}")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' was requested but no CUDA GPU is available")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' was requested but Apple Metal (MPS) is not available")
    return torch.device(pref)


def resolve_dtype(device: torch.device, preference: str = "bfloat16") -> torch.dtype:
    """Return the requested dtype, downgraded where the device can't run it."""
    if device.type == "cpu":
        return torch.float32
    if preference == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        # Pre-Ampere NVIDIA GPUs (e.g. T4, V100) have no bf16 support.
        return torch.float16
    return getattr(torch, preference)


def device_name(device: torch.device) -> str:
    """Human-readable hardware name, recorded with results for provenance."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            chip = "Apple Silicon"
        return f"{chip} (MPS)"
    return platform.processor() or "CPU"
