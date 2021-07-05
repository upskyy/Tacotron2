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

from tacotron2.net import ConvBNBlock
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    r"""
    The encoder converts a character sequence into a hidden feature
    representation which the decoder consumes to predict a spectrogram.

    Args:
        num_vocabs (int): The number of vocabulary.
        embedding_dim (int, optional): Dimension of embedding. (default: 512)
        hidden_size (int, optional): The number of features in the encoder hidden state. (default: 256)
        num_rnn_layers (int, optional): The number of rnn layers. (default: 1)
        num_conv_layers (int, optional): The number of convolution layers. (default: 3)
        dropout (float, optional): Dropout probability of convolution layer. (default: 0.5)
        kernel_size (int, optional): Value of convolution kernel size. (default: 5)

    Inputs: inputs, input_lengths
        - **inputs**: Tensor representing the character sequence. `LongTensor` of size ``(batch, seq_length)``
        - **input_lengths**: Tensor representing the sequence length. `LongTensor` of size ``(batch)``

    Returns: output
        - **output**: Tensor containing the encoded features of the input character sequences.
    """
    def __init__(
            self,
            num_vocabs: int,
            embedding_dim: int = 512,
            hidden_size: int = 256,
            num_rnn_layers: int = 1,
            num_conv_layers: int = 3,
            dropout: float = 0.5,
            kernel_size: int = 5,
    ) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        self.conv = nn.Sequential(*[
            ConvBNBlock(embedding_dim, embedding_dim, kernel_size, dropout, "relu") for _ in range(num_conv_layers)
        ])
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_rnn_layers, bias=True, batch_first=True, bidirectional=True)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        embedded = self.embedding(inputs).transpose(1, 2)

        conv_output = self.conv(embedded)
        conv_output = conv_output.transpose(1, 2)

        if self.training:
            self.rnn.flatten_parameters()

        output = pack_padded_sequence(conv_output, input_lengths.cpu(), batch_first=True)
        output, _ = self.rnn(output)
        output, _ = pad_packed_sequence(output, batch_first=True)

        return output