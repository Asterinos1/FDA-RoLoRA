from fdaopt.utils import torch
from math import sqrt
import gc
DEVICE = 'cpu'
class AmsSketch:
    """
    AMS Sketch class for approximate second moment estimation in PyTorch.
    """
    def __init__(self, depth=3, width=500):
        self.depth = depth
        self.width = width
        self.epsilon = 1. / sqrt(width)
        self.F = torch.randint(0, (1 << 31) - 1, (6, depth), dtype=torch.int32)
        self.precomputed_dict = {}
    def precompute(self, d):
        pos_tensor = self.tensor_hash31(torch.arange(d), self.F[0], self.F[1]) % self.width
        four = self.tensor_fourwise(torch.arange(d)).float()
        self.precomputed_dict[('pos_tensor', d)] = pos_tensor.to(DEVICE)
        self.precomputed_dict[('four', d)] = four.to(DEVICE)
    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647
    @staticmethod
    def tensor_hash31(x, a, b):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., torch.arange(d)) """
        x_reshaped = x.unsqueeze(-1)
        r = a * x_reshaped + b
        fold = torch.bitwise_xor(r >> 31, r)
        return fold & 2147483647
    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., torch.arange(d)) """
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])
        in2 = self.tensor_hash31(x, in1, self.F[4])
        in3 = self.tensor_hash31(x, in2, self.F[5])
        in4 = in3 & 32768
        return 2 * (in4 >> 15) - 1
    def sketch_for_vector(self, v):
        """ Efficient computation of sketch using PyTorch tensors.
        Args:
        - v (torch.Tensor): Vector to sketch. Shape=(d,).
        Returns:
        - torch.Tensor: An AMS Sketch. Shape=(`depth`, `width`).
        """
        d = v.shape[0]
        if ('four', d) not in self.precomputed_dict:
            self.precompute(d)
        four, pos_tensor = self.precomputed_dict[('four', d)], self.precomputed_dict[('pos_tensor', d)]
        sketch = self._sketch_for_vector(v, four, pos_tensor)
        gc.collect()
        return sketch
    def _sketch_for_vector(self, v, four, pos_tensor):
        """
        PyTorch translation of the TensorFlow function using a simple for loop.
        Args:
        - v (torch.Tensor): Vector to sketch. Shape=(d,).
        - four (torch.Tensor): Precomputed fourwise tensor. Shape=(d, depth).
        - indices (torch.Tensor): Precomputed indices for scattering. Shape=(d, depth, 2).
        Returns:
        - sketch (torch.Tensor): The AMS sketch tensor. Shape=(depth, width).
        """
        v_expand = v.unsqueeze(-1).to(DEVICE)
        deltas_tensor = four * v_expand
        sketch = torch.zeros((self.depth, self.width), dtype=torch.float32).to(DEVICE)
        for i in range(self.depth):
            width_indices = pos_tensor[:, i]
            deltas = deltas_tensor[:, i]
            sketch[i].index_add_(0, width_indices, deltas)
        return sketch
    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.
        Args:
        - sketch (torch.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).
        Returns:
        - float: Estimated squared Euclidean norm.
        """
        norm_sq_rows = torch.sum(sketch ** 2, dim=1)
        return torch.median(norm_sq_rows).item()

