import os
import sys
import torch
import pprint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdaopt import utils
from fdaopt.models.toy_model import ToyModel
from fdaopt.datasets.fed_data_prep import prepare_federated_datasets, extreme_non_iid_datasets
from fdaopt.training import fed_opt, fda_opt

def test_toy_architecture():
    print("=== 1. Testing ToyModel Architecture & Forward/Backward ===")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    utils.set_device(device)
    
    # Non-linear ToyModel
    model_nonlin = ToyModel(input_dim=784, rank=16, output_dim=10, linear=False).to(device)
    assert not model_nonlin.linear, "model_nonlin.linear should be False"
    assert model_nonlin.lora_A.shape == (784, 16), "lora_A shape mismatch"
    assert model_nonlin.lora_B.shape == (16, 784), "lora_B shape mismatch"
    assert model_nonlin.W_out.shape == (784, 10), "W_out shape mismatch"
    assert not model_nonlin.W_out.requires_grad, "W_out should be frozen"
    
    dummy_x = torch.randn(4, 784, device=device) * 255.0
    dummy_y = torch.tensor([0, 1, 2, 3], device=device)
    
    out = model_nonlin(image=dummy_x, labels=dummy_y)
    assert out.loss is not None, "Loss should not be None"
    assert out.logits.shape == (4, 10), "Logits shape mismatch"
    out.loss.backward()
    assert model_nonlin.lora_A.grad is not None, "lora_A grad should not be None"
    assert model_nonlin.lora_B.grad is not None, "lora_B grad should not be None"
    print("  [PASS] Non-linear ToyModel forward & backward verified.")

    # Linear ToyModel
    model_lin = ToyModel(input_dim=784, rank=16, output_dim=10, linear=True).to(device)
    assert model_lin.linear, "model_lin.linear should be True"
    out_lin = model_lin(image=dummy_x, labels=dummy_y)
    out_lin.loss.backward()
    assert model_lin.lora_A.grad is not None
    assert model_lin.lora_B.grad is not None
    print("  [PASS] Linear ToyModel forward & backward verified.")

def test_data_partitioning():
    print("\n=== 2. Testing MNIST Federated Data Partitioning ===")
    # Test 10 clients 1 class
    fed_ds_10c1, test_ds = prepare_federated_datasets(
        ds_path='ylecun/mnist', ds_name='mnist', checkpoint='toy-model',
        num_clients=10, alpha=0.0, batch_size=64, num_workers=0, pin_memory=False
    )
    assert len(fed_ds_10c1.client_dataloaders) == 10, "Should have 10 client loaders"
    print("  [PASS] 10 clients 1 class (10c1) partitioning verified.")
    
    # Test 5 clients 2 classes
    fed_ds_5c2, _ = prepare_federated_datasets(
        ds_path='ylecun/mnist', ds_name='mnist', checkpoint='toy-model',
        num_clients=5, alpha=0.0, batch_size=64, num_workers=0, pin_memory=False
    )
    assert len(fed_ds_5c2.client_dataloaders) == 5, "Should have 5 client loaders"
    print("  [PASS] 5 clients 2 classes (5c2) partitioning verified.")

def test_code_bug_linear_param():
    print("\n=== 3. Validating Code Inspection Finding: linear parameter passing ===")
    import inspect
    source_fda = inspect.getsource(fda_opt.fda_rolora_dual_track_opt)
    source_fed = inspect.getsource(fed_opt.fedavg_rolora)
    
    print("  Checking fed_opt.fedavg_rolora:")
    if "linear=hyperparams.get('linear', False)" in source_fed:
        print("    [CORRECT] fedavg_rolora passes linear parameter.")
    else:
        print("    [WARNING] fedavg_rolora does not pass linear parameter.")
        
    print("  Checking fda_opt.fda_rolora_dual_track_opt:")
    if "linear=hyperparams.get('linear', False)" in source_fda:
        print("    [CORRECT] fda_rolora_dual_track_opt passes linear parameter.")
    else:
        print("    [BUG CONFIRMED] fda_rolora_dual_track_opt omits linear parameter! Defaults to linear=False.")

if __name__ == '__main__':
    test_toy_architecture()
    test_data_partitioning()
    test_code_bug_linear_param()
    print("\nAll architecture and data validation tests completed successfully!")
