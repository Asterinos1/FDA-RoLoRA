"""
Lightweight matrix-factorization / toy model architecture for rapid experimentation.

Provides a 2-layer low-rank parameterization matching the theoretical formulation:
    f_i(x_i) = sigma(x_i @ lora_A @ lora_B) @ W_out
enabling fast CPU/GPU debugging, sanity testing, and mathematical validation
without the heavy resource overhead of full transformer backbones.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModelOutput:
    """
    Standardized container for ToyModel forward evaluation results.

    Attributes:
        loss (Optional[torch.Tensor]): Scalar cross-entropy loss if labels are provided.
        logits (torch.Tensor): Unnormalized class prediction scores.
    """

    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor):
        """
        Initialize ToyModelOutput container.

        Args:
            loss (Optional[torch.Tensor]): Scalar loss tensor or None.
            logits (torch.Tensor): Output class prediction logits.
        """
        self.loss = loss
        self.logits = logits


class ToyModel(nn.Module):
    """
    Two-layer toy model matching the paper's theoretical formulation:
        f_i(x_i) = ReLU(x_i @ A @ B) @ W_out
    where A in R^{d x r}, B in R^{r x d}, W_out in R^{d x c} is fixed.

    Architectural Details:
      1. `lora_B` is initialized with small-scale random values (not exact zeros).
         With non-linear ReLU active, an all-zero pre-activation causes ReLU'(0) = 0,
         zeroing gradients for both factors. Small random init avoids dead neurons
         while preserving low initial drift.
      2. `lora_B` projects back up to `input_dim`, and `W_out` provides a fixed
         (d x c) projection to class logits, faithfully reproducing the
         bottleneck -> representation -> classification pipeline.
    """

    def __init__(
        self,
        input_dim: int = 784,
        rank: int = 16,
        output_dim: int = 10,
        linear: bool = False
    ):
        """
        Initialize low-rank factors and fixed output projection matrix.

        Args:
            input_dim (int): Flattened feature input dimension (e.g. 784 for MNIST).
            rank (int): LoRA bottleneck rank (r).
            output_dim (int): Number of target prediction classes.
            linear (bool): If True, omit the intermediate ReLU non-linearity.
        """
        super().__init__()
        self.linear = linear
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, input_dim) * 1e-4)
        self.W_out = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ToyModelOutput:
        """
        Execute the low-rank forward pass and optional loss computation.

        Args:
            image (Optional[torch.Tensor]): Input image tensor of shape (B, ...).
            input_ids (Optional[torch.Tensor]): Alternative input tensor alias.
            labels (Optional[torch.Tensor]): Ground-truth target class labels.
            **kwargs: Unused arguments for compatibility with Hugging Face interfaces.

        Returns:
            ToyModelOutput: Output container containing optional loss and logits.
        """
        # Accept either image or input_ids for compatibility across dataloaders
        x = image if image is not None else input_ids
        x = x.view(x.size(0), -1).float() / 255.0

        # Low-rank factorized transformation
        hidden = x @ self.lora_A @ self.lora_B
        if not self.linear:
            hidden = F.relu(hidden)

        # Fixed projection to output classes
        logits = hidden @ self.W_out

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return ToyModelOutput(loss=loss, logits=logits)
