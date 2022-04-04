# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch

from espnet_local.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet_local.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet_local.nets.pytorch_backend.nets_utils import get_activation
from espnet_local.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet_local.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet_local.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet_local.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet_local.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet_local.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet_local.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet_local.nets.pytorch_backend.transformer.repeat import repeat
from espnet_local.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
        activation_type="swish",
        use_cnn_module=False,
        cnn_module_kernel=31,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)

        pos_enc_class = LegacyRelPositionalEncoding

        self.conv_subsampling_factor = 1

        self.embed = Conv2dSubsampling(
            idim,
            attention_dim,
            dropout_rate,
            pos_enc_class(attention_dim, positional_dropout_rate),
        )
        self.conv_subsampling_factor = 4

        self.normalize_before = normalize_before

        encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
        )

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            dropout_rate,
            activation,
        )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        xs, masks = self.encoders(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
