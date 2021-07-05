# Copyright (c) 2021, Sangchun Ha. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
from typing import Any, Optional
import torch.nn as nn


class PreNet(nn.Module):
    r"""
    Tacotron2 PreNet containing 2 fully connected layers.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        dropout (float): Probability of dropout.

    Shapes:
        - input: (B, T, D_in)
        - output: (B, T, D_out)
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super(PreNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.fc(inputs)


class PostNet(nn.Module):
    r"""
    Tacotron2 Postnet

    Args:
        n_mel (int, optional): The number of mel filters. (default: 80)
        postnet_dim (int, optional): Dimension of postnet. (default: 512)
        num_layers (int, optional): The number of convolution layers. (default: 5)
        kernel_size (int, optional): Value of convolution kernel size. (default: 5)
        dropout (float, optional): Probability of dropout. (default: 0.5)

    Shapes:
        - input: (B, T, D)
        - output: (B, T, D)
    """
    def __init__(
            self,
            n_mel: int = 80,
            postnet_dim: int = 512,
            num_layers: int = 5,
            kernel_size: int = 5,
            dropout: float = 0.5,
    ) -> None:
        super(PostNet, self).__init__()

        conv = nn.ModuleList()
        conv.append(ConvBNBlock(n_mel, postnet_dim, kernel_size, dropout, "tanh"))

        for _ in range(num_layers - 2):
            conv.append(ConvBNBlock(postnet_dim, postnet_dim, kernel_size, dropout, "tanh"))

        conv.append(ConvBNBlock(postnet_dim, n_mel, kernel_size, dropout, None))

        self.conv = nn.Sequential(*conv)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs.transpose(1, 2)).transpose(1, 2)


class ConvBNBlock(nn.Module):
    r"""
    Convolutions with Batch Normalization and non-linear activation.

    Args:
        in_channels (int): Input channel in convolutional layer.
        out_channels (int): Output channel in convolutional layer.
        kernel_size (int): Value of convolution kernel size.
        dropout (float, optional): Probability of dropout.
        activation (str, optional): 'relu', 'tanh', None. (default : 'relu')

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_out, T)
    """
    supported_activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'None': nn.Identity(),
    }

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dropout: float = 0.5,
            activation: Optional[Any] = 'relu',
    ) -> None:
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Keep the length unchanged."
        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    ConvBNBlock.supported_activations[str(activation)],
                    nn.Dropout(p=dropout),
                )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)