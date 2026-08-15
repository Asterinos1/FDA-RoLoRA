"""
Alon-Matias-Szegedy (AMS) sketching for memory-efficient second moment estimation.

Implements randomized 4-wise independent hashing in PyTorch to compute compact
streaming sketches of client drift vectors, enabling O(depth * width) estimation of
squared Euclidean norms (||v||^2) and variance in communication-constrained FDA loops.
"""

from math import sqrt
import gc
from typing import Tuple, Dict
import torch

DEVICE = 'cpu'


class AmsSketch:
    """
    AMS Sketch operator for approximate L2-norm squared and variance estimation.

    Attributes:
        depth (int): Number of independent hash functions / sketch rows.
        width (int): Number of buckets per hash table / sketch columns.
        epsilon (float): Theoretical error bound (1 / sqrt(width)).
        F (torch.Tensor): Random integer matrix defining universal hash parameters.
        precomputed_dict (dict): Cache for device-allocated index and sign tensors.
    """

    def __init__(self, depth: int = 3, width: int = 500):
        """
        Initialize the sketch parameters and random universal hash seeds.

        Args:
            depth (int): Number of hash rows (median-of-means trials).
            width (int): Number of bucket columns (controls estimation variance).
        """
        self.depth = depth
        self.width = width
        self.epsilon = 1.0 / sqrt(width)
        self.F = torch.randint(0, (1 << 31) - 1, (6, depth), dtype=torch.int32)
        self.precomputed_dict: Dict[Tuple[str, int], torch.Tensor] = {}

    def precompute(self, d: int):
        """
        Precompute bucket positions and 4-wise independent signs for vectors of dimension d.

        Args:
            d (int): Flattened parameter vector dimensionality.
        """
        indices = torch.arange(d)
        pos_tensor = self.tensor_hash31(indices, self.F[0], self.F[1]) % self.width
        four = self.tensor_fourwise(indices).float()
        self.precomputed_dict[('pos_tensor', d)] = pos_tensor.to(DEVICE)
        self.precomputed_dict[('four', d)] = four.to(DEVICE)

    @staticmethod
    def hash31(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Evaluate universal 2-independent hash: h(x) = ((a * x + b) ^ (r >> 31)) & (2^31 - 1).

        Args:
            x (torch.Tensor): Integer input scalar or tensor.
            a (torch.Tensor): Multiplicative hash parameter.
            b (torch.Tensor): Additive hash parameter.

        Returns:
            torch.Tensor: Hash value in range [0, 2^31 - 1].
        """
        r = a * x + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647

    @staticmethod
    def tensor_hash31(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Vectorized universal hash computation across a 1D tensor of indices.

        Args:
            x (torch.Tensor): Index vector of shape (d,).
            a (torch.Tensor): Hash parameter tensor.
            b (torch.Tensor): Hash parameter tensor.

        Returns:
            torch.Tensor: Hashed indices of shape (d, depth).
        """
        x_reshaped = x.unsqueeze(-1)
        r = a * x_reshaped + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647

    def tensor_fourwise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct 4-wise independent Rademacher random variables {-1, +1} via chained hashing.

        Args:
            x (torch.Tensor): Index vector of shape (d,).

        Returns:
            torch.Tensor: Sign tensor in {-1.0, +1.0} of shape (d, depth).
        """
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])
        in2 = self.tensor_hash31(x, in1, self.F[4])
        in3 = self.tensor_hash31(x, in2, self.F[5])
        in4 = in3 & 32768
        return 2 * (in4 >> 15) - 1

    def sketch_for_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project a high-dimensional vector v into a compact (depth x width) AMS sketch matrix.

        Args:
            v (torch.Tensor): Input flattened vector of shape (d,).

        Returns:
            torch.Tensor: Computed AMS sketch matrix of shape (depth, width).
        """
        d = v.shape[0]
        if ('four', d) not in self.precomputed_dict:
            self.precompute(d)
        four = self.precomputed_dict[('four', d)]
        pos_tensor = self.precomputed_dict[('pos_tensor', d)]
        sketch = self._sketch_for_vector(v, four, pos_tensor)
        gc.collect()
        return sketch

    def _sketch_for_vector(
        self,
        v: torch.Tensor,
        four: torch.Tensor,
        pos_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Scatter-add vector elements into sketch buckets using precomputed sign and index tables.

        Args:
            v (torch.Tensor): Vector to sketch, shape (d,).
            four (torch.Tensor): Sign table, shape (d, depth).
            pos_tensor (torch.Tensor): Bucket index table, shape (d, depth).

        Returns:
            torch.Tensor: Accumulated sketch tensor of shape (depth, width).
        """
        v_expand = v.unsqueeze(-1).to(DEVICE)
        deltas_tensor = four * v_expand
        sketch = torch.zeros((self.depth, self.width), dtype=torch.float32, device=DEVICE)
        for i in range(self.depth):
            width_indices = pos_tensor[:, i]
            deltas = deltas_tensor[:, i]
            sketch[i].index_add_(0, width_indices, deltas)
        return sketch

    @staticmethod
    def estimate_euc_norm_squared(sketch: torch.Tensor) -> float:
        """
        Estimate the squared Euclidean norm ||v||^2 by computing the median of row L2 norms.

        Args:
            sketch (torch.Tensor): AMS sketch tensor of shape (depth, width).

        Returns:
            float: Median estimator of the squared Euclidean norm.
        """
        norm_sq_rows = torch.sum(sketch ** 2, dim=1)
        return torch.median(norm_sq_rows).item()
