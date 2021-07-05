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
from typing import Any, Dict, Optional, Tuple
from tacotron2.net import PreNet
from tacotron2.attention import LocationSensitiveAttention
import torch
import torch.nn as nn
import random


class Decoder(nn.Module):
    r"""
    The decoder is an autoregressive recurrent neural network which predicts
    a mel spectrogram from the encoded input sequence one frame at a time.

    Args:
        n_mel (int, optional): The number of mel filters. (default: 80)
        embedding_dim (int, optional): Dimension of embedding. (default: 512)
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

    Inputs: encoder_outputs, inputs, teacher_forcing_ratio
        - **encoder_outputs**: Tensor containing the encoded features of the input character sequences.
        - **inputs**: Target mel-spectrogram for training.
        - **teacher_forcing_ratio**: The probability that teacher forcing will be used.

    Returns: outputs
        - **outputs**: Dictionary contains feature_outputs, stop_tokens, attention_weights.
    """
    def __init__(
            self,
            n_mel: int = 80,
            embedding_dim: int = 512,
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
    ) -> None:
        super(Decoder, self).__init__()
        self.n_mel = n_mel
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_decoding_step = max_decoding_step
        self.stop_threshold = stop_threshold
        self.prenet = PreNet(n_mel, prenet_dim, prenet_dropout)
        self.attention_rnn = nn.LSTMCell(embedding_dim + prenet_dim, hidden_dim)
        self.decoder_rnn = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.attention = LocationSensitiveAttention(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            attention_kernel_size=attention_kernel_size,
            attention_filter_size=attention_filter_size,
            smoothing=smoothing,
        )
        self.attention_dropout = nn.Dropout(p=attention_dropout)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)
        self.feature_generator = nn.Linear(embedding_dim + hidden_dim, n_mel)
        self.stop_generator = nn.Linear(embedding_dim + hidden_dim, 1)

    def _forward_step(
            self,
            step_input: Tensor,
            encoder_outputs: Tensor,
            hidden_states: list,
            cell_states: list,
            attention_weights: Tensor,
            cumulative_attention_weights: Tensor,
            context: Tensor,
    ) -> Dict[str, Any]:
        step_input = torch.cat([step_input, context], dim=1)

        hidden_states[0], cell_states[0] = self.attention_rnn(step_input, (hidden_states[0], cell_states[0]))
        hidden_states[0] = self.attention_dropout(hidden_states[0])

        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), cumulative_attention_weights.unsqueeze(1)), dim=1)
        context, attention_weights = self.attention(hidden_states[0], encoder_outputs, attention_weights_cat)
        cumulative_attention_weights += attention_weights

        step_input = torch.cat([hidden_states[0], context], dim=1)

        hidden_states[1], cell_states[1] = self.decoder_rnn(step_input, (hidden_states[1], cell_states[1]))
        hidden_states[1] = self.decoder_dropout(hidden_states[1])

        output = torch.cat([hidden_states[1], context], dim=1)

        feature_output = self.feature_generator(output)
        stop_token = self.stop_generator(output).squeeze(1)

        return {
            "hidden_states": hidden_states,
            "cell_states": cell_states,
            "attention_weights": attention_weights,
            "cumulative_attention_weights": cumulative_attention_weights,
            "context": context,
            "feature_output": feature_output,
            "stop_token": stop_token,
        }

    def forward(
            self,
            encoder_outputs: Tensor,
            inputs: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, Tensor]:
        feature_outputs, stop_tokens, attention_weights = list(), list(), list()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        inputs, max_decoding_step = self._validate_args(encoder_outputs, inputs, use_teacher_forcing)

        decoder_states = self._init_states(encoder_outputs)

        if use_teacher_forcing:
            inputs = self.prenet(inputs)

            for di in range(max_decoding_step):
                step_input = inputs[:, di, :]
                decoder_states = self._forward_step(
                    step_input=step_input,
                    encoder_outputs=encoder_outputs,
                    hidden_states=decoder_states["hidden_states"],
                    cell_states=decoder_states["cell_states"],
                    attention_weights=decoder_states["attention_weights"],
                    cumulative_attention_weights=decoder_states["cumulative_attention_weights"],
                    context=decoder_states["context"]
                )
                feature_outputs.append(decoder_states["feature_output"])
                stop_tokens.append(decoder_states["stop_token"])
                attention_weights.append(decoder_states["attention_weights"])

        else:
            step_input = inputs

            for _ in range(max_decoding_step):
                step_input = self.prenet(step_input)
                decoder_states = self._forward_step(
                    step_input=step_input,
                    encoder_outputs=encoder_outputs,
                    hidden_states=decoder_states["hidden_states"],
                    cell_states=decoder_states["cell_states"],
                    attention_weights=decoder_states["attention_weights"],
                    cumulative_attention_weights=decoder_states["cumulative_attention_weights"],
                    context=decoder_states["context"]
                )
                feature_outputs.append(decoder_states["feature_output"])
                stop_tokens.append(decoder_states["stop_token"])
                attention_weights.append(decoder_states["attention_weights"])

                if torch.sigmoid(decoder_states["stop_token"]).mean() > self.stop_threshold:
                    break

                step_input = decoder_states["feature_output"]

        return self._parse_outputs(feature_outputs, stop_tokens, attention_weights)

    def _parse_outputs(self, feature_outputs: list, stop_tokens: list, attention_weights: list) -> Dict[str, Tensor]:
        batch = attention_weights[0].size(0)
        seq_length = attention_weights[0].size(1)

        stop_tokens = torch.stack(stop_tokens, dim=1)  # (B, T)
        attention_weights = torch.stack(attention_weights, dim=0)
        attention_weights = attention_weights.contiguous().view(batch, -1, seq_length)

        feature_outputs = torch.stack(feature_outputs, dim=0)
        feature_outputs = feature_outputs.contiguous().view(batch, -1, self.n_mel)

        return {
            "feature_outputs": feature_outputs,
            "stop_tokens": stop_tokens,
            "attention_weights": attention_weights,
        }

    def _init_states(self, encoder_outputs: Tensor) -> Dict[str, Any]:
        hidden_states, cell_states = list(), list()

        batch = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)

        hidden_states.append(encoder_outputs.new_zeros(batch, self.hidden_dim))
        hidden_states.append(encoder_outputs.new_zeros(batch, self.hidden_dim))

        cell_states.append(encoder_outputs.new_zeros(batch, self.hidden_dim))
        cell_states.append(encoder_outputs.new_zeros(batch, self.hidden_dim))

        attention_weights = encoder_outputs.new_zeros(batch, seq_length)
        cumulative_attention_weights = encoder_outputs.new_zeros(batch, seq_length)
        context = encoder_outputs.new_zeros(batch, self.embedding_dim)

        return {
            "hidden_states": hidden_states,
            "cell_states": cell_states,
            "attention_weights": attention_weights,
            "cumulative_attention_weights": cumulative_attention_weights,
            "context": context,
        }

    def _validate_args(
            self,
            encoder_outputs: Tensor,
            inputs: Optional[Any] = None,
            use_teacher_forcing: bool = True,
    ) -> Tuple[Tensor, int]:
        assert encoder_outputs is not None
        batch = encoder_outputs.size(0)

        if not use_teacher_forcing:
            inputs = encoder_outputs.new_zeros(batch, self.n_mel)
            max_decoding_step = self.max_decoding_step

        else:
            go_frame = encoder_outputs.new_zeros(batch, 1, self.n_mel)
            inputs = torch.cat([go_frame, inputs], dim=1)

            max_decoding_step = inputs.size(1) - 1

        return inputs, max_decoding_step