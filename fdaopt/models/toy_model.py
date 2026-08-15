import torch
import torch.nn as nn
import torch.nn.functional as F
class ToyModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits
class ToyModel(nn.Module):
    """
    Two-layer toy model matching the paper's formulation:
        f_i(x_i) = ReLU(x_i @ A @ B) @ W_out
    where A in R^{d x r}, B in R^{r x d}, W_out in R^{d x c} is fixed.
    Fixes vs. the original implementation:
      1. `lora_B` is no longer initialized to exact zeros. With ReLU active
         (linear=False), an all-zero pre-activation makes ReLU'(0) = 0
         everywhere, which zeroes out the gradient for both lora_A and
         lora_B on every step -- the model can never leave initialization.
         A small-scale random init preserves LoRA's "near-identity at
         init" spirit without killing the gradient.
      2. `lora_B` now maps back up to `input_dim` (not `output_dim`), and
         `W_out` is a genuine fixed (d x c) projection rather than a
         (c x c) identity, matching the paper's two-stage
         bottleneck -> full representation -> class logits pipeline.
    """
    def __init__(self, input_dim=784, rank=16, output_dim=10, linear=False):
        super().__init__()
        self.linear = linear
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, input_dim) * 1e-4)
        self.W_out = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)
    def forward(self, image=None, input_ids=None, labels=None, **kwargs):
        x = image if image is not None else input_ids
        x = x.view(x.size(0), -1).float() / 255.0
        hidden = x @ self.lora_A @ self.lora_B
        if not self.linear:
            hidden = F.relu(hidden)
        logits = hidden @ self.W_out
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        return ToyModelOutput(loss, logits)

