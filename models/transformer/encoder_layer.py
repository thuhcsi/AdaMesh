# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class Linear(torch.nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
    super(Linear, self).__init__()
    self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

    torch.nn.init.xavier_uniform_(
      self.linear_layer.weight,
      gain=torch.nn.init.calculate_gain(w_init_gain))
    return

  def forward(self, x):
    return self.linear_layer(x)

class Conv1d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
               padding=None, dilation=1, bias=True, w_init_gain='linear'):
    super(Conv1d, self).__init__()
    if padding is None:
      assert (kernel_size % 2 == 1)
      padding = int(dilation * (kernel_size - 1) / 2)

    self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=bias)

    torch.nn.init.xavier_uniform_(
      self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return

  def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal


class Postnet(nn.Module):
  """ Postnet: Five 1-d convolution with 512 channels and kernel size 5 """

  def __init__(self, embed_dim, out_dim, kernel_size, n_conv):
    super(Postnet, self).__init__()

    self.convolutions = nn.ModuleList()

    self.convolutions.append(
      nn.Sequential(
        Conv1d(out_dim, embed_dim,
               kernel_size=kernel_size, stride=1,
               padding=int((kernel_size - 1) / 2),
               dilation=1, w_init_gain='tanh'),
        nn.BatchNorm1d(embed_dim))
    )

    for i in range(1, n_conv - 1):
      self.convolutions.append(
        nn.Sequential(
          Conv1d(embed_dim,
                 embed_dim,
                 kernel_size=kernel_size, stride=1,
                 padding=int((kernel_size - 1) / 2),
                 dilation=1, w_init_gain='tanh'),
          nn.BatchNorm1d(embed_dim))
      )

    self.convolutions.append(
      nn.Sequential(
        Conv1d(embed_dim, out_dim,
               kernel_size=kernel_size, stride=1,
               padding=int((kernel_size - 1) / 2),
               dilation=1, w_init_gain='linear'),
        nn.BatchNorm1d(out_dim))

    )
    self.projection = Linear(
      out_dim, out_dim)

  def forward(self, x):
    """ using residual network """
    init = x
    for i, layer in enumerate(self.convolutions):
      if i < len(self.convolutions) - 1: # self.convolutions)[0:-1]
        x = F.dropout(torch.tanh(layer(x)), 0.5, self.training)
      else: # self.convolutions)[-1]
        x = F.dropout(layer(x), 0.5, self.training)
    x = self.projection(x.transpose(2, 1)).transpose(2, 1)
    x = init + x
    return x

  def inference(self, x):
    init = x
    for i, layer in enumerate(self.convolutions):
      if i < len(self.convolutions) - 1: # self.convolutions)[0:-1]
        x = F.dropout(torch.tanh(layer(x)), 1.0, False)
      else: # self.convolutions)[-1]
        x = F.dropout(layer(x), 1.0, False)
    x = self.projection(x.transpose(2, 1)).transpose(2, 1)
    x = init + x
    return x


# class CausalPostnet(nn.Module):
#     """ConvolutionModule in Conformer model."""
#     def __init__(self,
#                  channels: int = 32,
#                  kernel_size: int = 15,
#                  activation_type: str = "swish",
#                  norm: str = "layer_norm",
#                  causal: bool = True,
#                  bias: bool = True):
#         """Construct an ConvolutionModule object.
#         Args:
#             channels (int): The number of channels of conv layers.
#             kernel_size (int): Kernel size of conv layers.
#             causal (int): Whether use causal convolution or not
#         """
#         super().__init__()

#         activation = get_activation(activation_type)

#         self.pointwise_conv1 = nn.Conv1d(
#             channels,
#             2 * channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#         # self.lorder is used to distinguish if it's a causal convolution,
#         # if self.lorder > 0: it's a causal convolution, the input will be
#         #    padded with self.lorder frames on the left in forward.
#         # else: it's a symmetrical convolution
#         if causal:
#             padding = 0
#             self.lorder = kernel_size - 1
#         else:
#             # kernel_size should be an odd number for none causal convolution
#             assert (kernel_size - 1) % 2 == 0
#             padding = (kernel_size - 1) // 2
#             self.lorder = 0
#         self.depthwise_conv = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size,
#             stride=1,
#             padding=padding,
#             groups=channels,
#             bias=bias,
#         )

#         assert norm in ['batch_norm', 'layer_norm']
#         if norm == "batch_norm":
#             self.use_layer_norm = False
#             self.norm = nn.BatchNorm1d(channels)
#         else:
#             self.use_layer_norm = True
#             self.norm = nn.LayerNorm(channels)

#         self.pointwise_conv2 = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#         self.activation = activation

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
#         cache: torch.Tensor = torch.zeros((0, 0, 0)),
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Compute convolution module.
#         Args:
#             x (torch.Tensor): Input tensor (#batch, time, channels).
#             mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
#                 (0, 0, 0) means fake mask.
#             cache (torch.Tensor): left context cache, it is only
#                 used in causal convolution (#batch, channels, cache_t),
#                 (0, 0, 0) meas fake cache.
#         Returns:
#             torch.Tensor: Output tensor (#batch, time, channels).
#         """
#         # exchange the temporal dimension and the feature dimension
#         x = x.transpose(1, 2)  # (#batch, channels, time)

#         # mask batch padding
#         if mask_pad.size(2) > 0:  # time > 0
#             x.masked_fill_(~mask_pad, 0.0)

#         if self.lorder > 0:
#             if cache.size(2) == 0:  # cache_t == 0
#                 x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
#             else:
#                 assert cache.size(0) == x.size(0)  # equal batch
#                 assert cache.size(1) == x.size(1)  # equal channel
#                 x = torch.cat((cache, x), dim=2)
#             assert (x.size(2) > self.lorder)
#             new_cache = x[:, :, -self.lorder:]
#         else:
#             # It's better we just return None if no cache is requried,
#             # However, for JIT export, here we just fake one tensor instead of
#             # None.
#             new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

#         # GLU mechanism
#         x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
#         x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

#         # 1D Depthwise Conv
#         x = self.depthwise_conv(x)
#         if self.use_layer_norm:
#             x = x.transpose(1, 2)
#         x = self.activation(self.norm(x))
#         if self.use_layer_norm:
#             x = x.transpose(1, 2)
#         x = self.pointwise_conv2(x)
#         # mask batch padding
#         if mask_pad.size(2) > 0:  # time > 0
#             x.masked_fill_(~mask_pad, 0.0)

#         return x.transpose(1, 2), new_cache


class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
