import torch
from io import StringIO

from HydroInverseML.src.HydroInverseML import HIL_model
from HydroInverseML.src.HydroInverseML.data_loader import open_file, access_hydrodata

# to execute and show print statements run pytest -s

def test_math():
    assert 1 + 1 == 2

def test_HIL_model():
    test_in_channel = 4
    test_out_channels = [16, 32, 32, 16, 4]
    test_filter_sizes = [5, 33, 9, 33, 5]
    test_max_pool_kernel_sizes = [2]
    model = HIL_model(in_channel=test_in_channel, filter_sizes=test_filter_sizes, max_pool_kernel_sizes=test_max_pool_kernel_sizes, dropout_rate=0.0, out_channels=test_out_channels, stride=1)
    assert len(test_out_channels) == model.num_hidden_layers
    assert len(test_filter_sizes) == model.num_hidden_layers
    print(f"Model created with {model.num_hidden_layers} hidden layers.")
    print(model)

    x = torch.randn((1, test_in_channel, 150, 150))
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    assert y.shape == (1, test_out_channels[-1], 150, 150)

def test_data_loader():
    content = open_file("test.txt")
    assert content == "test file"

"""
def test_access_hydrodata():
    # Test access_hydrodata function
    assert access_hydrodata() is True
"""