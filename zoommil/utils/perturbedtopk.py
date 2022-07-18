"""
Based on the differentiable Top-K operator from:

Cordonnier, J., Mahendran, A., Dosovitskiy, A.: Differentiable patch selection for
image recognition. In: IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR 2021)

https://openaccess.thecvf.com/content/CVPR2021/supplemental/Cordonnier_Differentiable_Patch_Selection_CVPR_2021_supplemental.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
  def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
    super(PerturbedTopK, self).__init__()
    self.num_samples = num_samples
    self.sigma = sigma
    self.k = k

  def __call__(self, x):
    return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
    # b = batch size
    b, num_patches = x.shape
    # for Gaussian: noise and gradient are the same.
    noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, num_patches)).to(x.device)

    perturbed_x = x[:, None, :] + noise * sigma # [b, num_s, num_p]
    topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # [b, num_s, k]
    indices = torch.sort(indices, dim=-1).values # [b, num_s, k]

    # b, num_s, k, num_p
    perturbed_output = F.one_hot(indices, num_classes=num_patches).float()
    indicators = perturbed_output.mean(dim=1) # [b, k, num_p]

    # constants for backward
    ctx.k = k
    ctx.num_samples = num_samples
    ctx.sigma = sigma

    # tensors for backward
    ctx.perturbed_output = perturbed_output
    ctx.noise = noise

    return indicators

  @staticmethod
  def backward(ctx, grad_output):
    if grad_output is None:
      return tuple([None] * 5)

    noise_gradient = ctx.noise
    expected_gradient = (
        torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
        / ctx.num_samples
        / ctx.sigma
    )
    grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
    return (grad_input,) + tuple([None] * 5)
