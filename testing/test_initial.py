import torch
from HydroInverseML.code import HIL_model

def test_math():
    assert 1 + 1 == 2

def test_HIL_model():
    test_out_channels = [4, 16, 32]
    test_filter_sizes = [5, 33, 9]
    model = HIL_model.HIL_model(in_channel=4, out_channels=test_out_channels, filter_sizes=test_filter_sizes, stride=1, dropout_rate=0.0)
    assert len(test_out_channels) == model.num_hidden_layers
    # x = torch.randn((1, 4, 150, 150))
    # y = model.conv_1(x)
    # assert y.shape == (1, 16, 150, 150)

    