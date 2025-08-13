
import sys
import torch
import torch.nn as nn

class HIL_model(torch.nn.Module):
    def __init__(self, in_channel, filter_sizes, max_pool_kernel_sizes, dropout_rate, out_channels = None, stride = 1, batch_norm_momentum = 0.1, paddings=None, seed=0):
        super().__init__()
        self.num_hidden_layers = len(filter_sizes)
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        self.conv_layers = nn.ModuleList()

        assert self.num_hidden_layers == len(out_channels) if out_channels is not None else True, "Number of out_channels must match number of filter sizes."

        # Encoder layers
        current_channel = in_channel
        pooling_counter = 0
        for current_layer in range((self.num_hidden_layers + 1) // 2):
            out_channel = current_channel * 2 if out_channels is None else out_channels[current_layer]
            current_padding = ((filter_sizes[current_layer] - 1) // 2) if paddings is None else paddings[current_layer]
            self.conv_layers.append(nn.Conv2d(current_channel, out_channel, kernel_size=(filter_sizes[current_layer], filter_sizes[current_layer]), padding=current_padding, stride=stride, padding_mode='zeros'))
            self.conv_layers.append(nn.ELU())
            self.conv_layers.append(nn.BatchNorm2d(out_channel, momentum=batch_norm_momentum))
            if (current_layer + 1) % 2 == 0:
                print(f"Adding MaxPool layer with kernel size {max_pool_kernel_sizes[pooling_counter]} at layer {current_layer}")
                #self.conv_layers.append(nn.Dropout(dropout_rate))
                self.conv_layers.append(nn.MaxPool2d(kernel_size=(max_pool_kernel_sizes[pooling_counter],max_pool_kernel_sizes[pooling_counter]), stride=max_pool_kernel_sizes[pooling_counter], padding=max_pool_kernel_sizes[pooling_counter] // 2, return_indices=True))
                pooling_counter += 1
            current_channel = out_channel
 
        print(f"Total pooling layers added: {pooling_counter}, expected: {len(max_pool_kernel_sizes)}")
        assert pooling_counter == len(max_pool_kernel_sizes), "Number of max pool kernel sizes must match number of executed pooling layers."
        pooling_counter -= 1  # Adjust for the last pooling layer

        # Decoder layers
        for current_layer in range(((self.num_hidden_layers+1) // 2), self.num_hidden_layers):
            if (current_layer + 1) % 2 == 0:
                #self.conv_layers.append(nn.Dropout(dropout_rate))
                self.conv_layers.append(nn.MaxUnpool2d(kernel_size=(max_pool_kernel_sizes[pooling_counter],max_pool_kernel_sizes[pooling_counter]), stride=max_pool_kernel_sizes[pooling_counter], padding=max_pool_kernel_sizes[pooling_counter] // 2))
                pooling_counter -= 1

            out_channel = current_channel // 2 if out_channels is None else out_channels[current_layer]
            current_padding = ((filter_sizes[current_layer] - 1) // 2) if paddings is None else paddings[current_layer]
            self.conv_layers.append(nn.Conv2d(current_channel, out_channel, kernel_size=(filter_sizes[current_layer], filter_sizes[current_layer]), padding=current_padding,  stride=stride, padding_mode='zeros'))
            self.conv_layers.append(nn.ELU())
            self.conv_layers.append(nn.BatchNorm2d(out_channel, momentum=batch_norm_momentum))
            current_channel = out_channel

        # Final layer to ensure that output has no negative values
        self.conv_layers.append(nn.ReLU())

    def forward(self, x):
        pooling_counter = -1 # Start from -1 to handle the first pooling layer correctly
        before_pooling_shapes = []
        pooling_indices = []
        clones_for_skip_connections = []
        for layer in self.conv_layers:
            if isinstance(layer, nn.MaxPool2d):
                before_pooling_shapes.append(x.shape)
                clones_for_skip_connections.append(x.clone())
                x, indices = layer(x)
                pooling_indices.append(indices)
                pooling_counter += 1
            elif isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, pooling_indices[pooling_counter], output_size=before_pooling_shapes[pooling_counter])
                x = clones_for_skip_connections[pooling_counter] + x
                pooling_counter -= 1
            else:
                x = layer(x)
        return x
    
    def loss(self, pred, label):
        return torch.nn.MSELoss()(pred, label)
    