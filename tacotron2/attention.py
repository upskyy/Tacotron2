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
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationSensitiveAttention(nn.Module):
    r"""
    Location-sensitive attention uses cumulative attention weights
    from previous decoder time steps as an additional feature.

    Args:
        hidden_dim (int): Dimension of rnn hidden state vector. (default: 1024)
        embedding_dim (int): Dimension of embedding. (default: 512)
        attention_dim (int): Dimension of attention. (default: 128)
        attention_kernel_size (int): Value of attention convolution kernel size. (default: 31)
        attention_filter_size (int): Value of attention convolution filter size. (default: 32)
        smoothing (bool): flag indication smoothing or not. (default: False)

    Inputs: query, value, last_attention_weights
        - **query** (batch, q_len, hidden_dim): Tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): Tensor containing features of the encoded input sequence.
        - **last_attention_weights** (batch, v_len): Tensor containing previous time step`s attention weight.

    Returns: context, attention_weights
        - **context** (batch, hidden_dim): Tensor containing the feature from encoder outputs.
        - **attention_weights** (batch, v_len): Tensor containing the attention weight from the encoder outputs.
    """
    def __init__(
            self,
            hidden_dim: int = 1024,
            embedding_dim: int = 512,
            attention_dim: int = 128,
            attention_kernel_size: int = 31,
            attention_filter_size: int = 32,
            smoothing: bool = False,
    ) -> None:
        super(LocationSensitiveAttention, self).__init__()
        self.attention_dim = attention_dim
        self.smoothing = smoothing
        self.query_linear = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.value_linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.alignment_linear = nn.Linear(attention_filter_size, attention_dim, bias=False)
        self.fc = nn.Linear(attention_dim, 1, bias=False)
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=attention_filter_size,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False,
        )

    def forward(
            self,
            query: Tensor,
            value: Tensor,
            last_attention_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        last_attention_weights = self.conv(last_attention_weights).transpose(1, 2)
        last_attention_weights = self.alignment_linear(last_attention_weights)

        attention_weights = self.fc(torch.tanh(
            self.query_linear(query).unsqueeze(1)
            + self.value_linear(value)
            + last_attention_weights)
        ).squeeze(-1)

        if self.smoothing:
            attention_weights = torch.sigmoid(attention_weights)
            attention_weights = torch.div(attention_weights, attention_weights.sum(dim=1).unsqueeze(1))

        else:
            attention_weights = F.softmax(attention_weights, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), value)
        context = context.squeeze(1)  # (B, D)

        return context, attention_weights