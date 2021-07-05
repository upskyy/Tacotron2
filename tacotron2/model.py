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

from tacotron2.encoder import Encoder
from tacotron2.decoder import Decoder
from tacotron2.net import PostNet
from torch import Tensor
from typing import Optional, Dict
import torch.nn as nn


class Tacotron2(nn.Module):
    r"""
    Tacotron 2, a fully neural TTS system that combines a sequence-to-sequence
    recurrent network with attention predicts mel-spectrograms.

    Args:
        num_vocabs (int): The number of vocabulary.
        embedding_dim (int, optional): Dimension of embedding. (default: 512)
        hidden_size (int, optional): The number of features in the encoder hidden state. (default: 256)
        num_rnn_layers (int, optional): The number of rnn layers. (default: 1)
        num_conv_layers (int, optional): The number of convolution layers. (default: 3)
        encoder_dropout (float, optional): Dropout probability of encoder convolution layer. (default: 0.5)
        encoder_kernel_size (int, optional): Value of encoder convolution kernel size. (default: 5)
        n_mel (int, optional): The number of mel filters. (default: 80)
        attention_dim (int, optional): Dimension of attention. (default: 128)
        hidden_dim (int, optional): Dimension of hidden. (default: 1024)
        prenet_dim (int, optional): Dimension of prenet. (default: 256)
        max_decoding_step (int, optional): Max decoding step (default: 1000)
        attention_kernel_size (int, optional): Value of attention convolution kernel size (default: 31)
        attention_filter_size (int, optional): Value of attention convolution filter size (default: 32)
        prenet_dropout (float, optional): Dropout probability of prenet (default: 0.5)
        attention_dropout (float, optional): Dropout probability of attention (default: 0.1)
        decoder_dropout (float, optional): Dropout probability of decoder (default: 0.1)
        stop_threshold (float, optional): Stop threshold (default: 0.5)
        smoothing (bool, optional): Flag indication smoothing or not (default: False)
        postnet_dim (int, optional): Dimension of postnet. (default: 512)
        postnet_num_layers (int, optional): The number of convolution layers. (default: 5)
        postnet_kernel_size (int, optional): Value of convolution kernel size. (default: 5)
        postnet_dropout (float, optional): Probability of dropout. (default: 0.5)

    Inputs: inputs, input_lengths, target, teacher_forcing_ratio
        - **inputs**: Tensor representing the character sequence. `LongTensor` of size ``(batch, seq_length)``
        - **input_lengths**: Tensor representing the sequence length. `LongTensor` of size ``(batch)``
        - **target**: Target mel-spectrogram for training. `FloatTensor` of size ``(batch, seq_length, dimension)``
        - **teacher_forcing_ratio**: The probability that teacher forcing will be used.

    Returns: outputs
        - **outputs**: Dictionary contains feature_outputs, stop_tokens, attention_weights.
    """
    def __init__(
            self,
            num_vocabs: int,
            embedding_dim: int = 512,
            hidden_size: int = 256,
            num_rnn_layers: int = 1,
            num_conv_layers: int = 3,
            encoder_dropout: float = 0.5,
            encoder_kernel_size: int = 5,
            n_mel: int = 80,
            attention_dim: int = 128,
            hidden_dim: int = 1024,
            prenet_dim: int = 256,
            max_decoding_step: int = 1000,
            attention_kernel_size: int = 31,
            attention_filter_size: int = 32,
            prenet_dropout: float = 0.5,
            attention_dropout: float = 0.1,
            decoder_dropout: float = 0.1,
            stop_threshold: float = 0.5,
            smoothing: bool = False,
            postnet_dim: int = 512,
            postnet_num_layers: int = 5,
            postnet_kernel_size: int = 5,
            postnet_dropout: float = 0.5,
    ) -> None:
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(
            num_vocabs=num_vocabs,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            num_conv_layers=num_conv_layers,
            dropout=encoder_dropout,
            kernel_size=encoder_kernel_size,
        )
        self.decoder = Decoder(
            n_mel=n_mel,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            prenet_dim=prenet_dim,
            max_decoding_step=max_decoding_step,
            attention_kernel_size=attention_kernel_size,
            attention_filter_size=attention_filter_size,
            prenet_dropout=prenet_dropout,
            attention_dropout=attention_dropout,
            decoder_dropout=decoder_dropout,
            stop_threshold=stop_threshold,
            smoothing=smoothing,
        )
        self.postnet = PostNet(
            n_mel=n_mel,
            postnet_dim=postnet_dim,
            num_layers=postnet_num_layers,
            kernel_size=postnet_kernel_size,
            dropout=postnet_dropout,
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, Tensor]:
        encoder_outputs = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(encoder_outputs, targets, teacher_forcing_ratio)

        postnet_outputs = self.postnet(decoder_outputs["feature_outputs"])
        decoder_outputs["feature_outputs"] += postnet_outputs

        return decoder_outputs