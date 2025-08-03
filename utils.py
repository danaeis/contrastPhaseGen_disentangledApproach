import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# Sinusoidal Positional Encoding for Phases
def get_phase_embedding(phase_index, dim=32):
    """Generate sinusoidal positional encoding for phase index (0 to 3)."""
    pe = torch.zeros(dim)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe[0::2] = torch.sin(phase_index * div_term)
    pe[1::2] = torch.cos(phase_index * div_term)
    return pe

# Save volume as NIfTI file
def save_volume(volume, path):
    """Save a PyTorch tensor volume as a NIfTI file."""
    volume_np = volume.cpu().numpy().squeeze()
    nii_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
    nib.save(nii_img, path)