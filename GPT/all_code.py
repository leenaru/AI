#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import json
from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from mobileclip.clip import CLIP
from mobileclip.modules.text.tokenizer import (
    ClipTokenizer,
)
from mobileclip.modules.common.mobileone import reparameterize_model


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    reparameterize: Optional[bool] = True,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Any, Any]:
    """
    Method to instantiate model and pre-processing transforms necessary for inference.

    Args:
        model_name: Model name. Choose from ['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
        pretrained: Location of pretrained checkpoint.
        reparameterize: When set to True, re-parameterizable branches get folded for faster inference.
        device: Device identifier for model placement.

    Returns:
        Tuple of instantiated model, and preprocessing transforms for inference.
    """
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cfg = json.load(open(model_cfg_file, "r"))

    # Build preprocessing transforms for inference
    resolution = model_cfg["image_cfg"]["image_size"]
    resize_size = resolution
    centercrop_size = resolution
    aug_list = [
        Resize(
            resize_size,
            interpolation=InterpolationMode.BILINEAR,
        ),
        CenterCrop(centercrop_size),
        ToTensor(),
    ]
    preprocess = Compose(aug_list)

    # Build model
    model = CLIP(cfg=model_cfg)
    model.to(device)
    model.eval()

    # Load checkpoint
    if pretrained is not None:
        chkpt = torch.load(pretrained)
        model.load_state_dict(chkpt)

    # Reparameterize model for inference (if specified)
    if reparameterize:
        model = reparameterize_model(model)

    return model, None, preprocess


def get_tokenizer(model_name: str) -> nn.Module:
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    model_cfg = json.load(open(model_cfg_file, "r"))

    # Build tokenizer
    text_tokenizer = ClipTokenizer(model_cfg)
    return text_tokenizer

# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
""" Model schema in open_clip format for inference only. """
import math
from typing import Any, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn

from mobileclip.text_encoder import (
    TextTransformer,
)

from .image_encoder import MCi


class CLIP(nn.Module):
    """Base class for multi-modal image-text data"""

    def __init__(self, cfg: Dict, output_dict: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.projection_dim = cfg["embed_dim"]
        if self.projection_dim is None:
            raise ValueError("Please specify `embed_dim` in model config.")

        self.image_encoder = MCi(
            model_name=cfg["image_cfg"]["model_name"],
            projection_dim=self.projection_dim,
        )
        self.text_encoder = TextTransformer(
            cfg=cfg["text_cfg"], projection_dim=self.projection_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def encode_image(self, image: torch.Tensor, normalize: bool = False):
        image_encoder_out = self.image_encoder(image)
        if isinstance(image_encoder_out, dict):
            features = image_encoder_out["logits"]
        else:
            features = image_encoder_out
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False):
        text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Any:

        image_embeddings = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_embeddings = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            return {
                "image_features": image_embeddings,
                "text_features": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
            }
        return image_embeddings, text_embeddings, self._exponentiate_and_clip_logits()

# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Any

import torch.nn as nn
from timm.models import create_model

from mobileclip import models  # Added to register models
from mobileclip.modules.image.image_projection import GlobalPool2D


class MCi(nn.Module):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        # Create model
        self.model = create_model(model_name, projection_dim=self.projection_dim)

        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        x = self.model(x)
        return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            # Classifier that uses nn.Sequential usually has global pooling and
            # multiple linear layers. Find the first linear layer and get its
            # in_features
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier


# --------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from mobileclip.modules.common.transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)
from mobileclip.modules.text.repmixer import RepMixerBlock
from mobileclip import logger


class TextTransformer(nn.Module):
    def __init__(self, cfg: dict, projection_dim: int, *args, **kwargs) -> None:
        super().__init__()

        model_dim = cfg["dim"]
        no_scale_embedding = cfg.get("no_scale_embedding", False)
        no_pos_embedding = cfg.get("no_pos_embedding", False)
        embed_dropout = cfg.get("embed_dropout", 0.0)
        norm_layer = cfg["norm_layer"]
        variant = cfg["model_name"]
        self.vocab_size = cfg["vocab_size"]
        self.projection_dim = projection_dim

        # Token embedding layer
        self.embedding_layer = nn.Embedding(
            embedding_dim=model_dim, num_embeddings=self.vocab_size
        )
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        # Context length
        context_length = cfg["context_length"]
        assert (
            context_length is not None
        ), "Context length can't be None. Please set value accordingly."

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(
                num_embeddings=context_length, embedding_dim=model_dim
            )
        )

        self.embedding_dropout = nn.Dropout(p=embed_dropout)

        # Transformer layer
        n_transformer_layers = cfg["n_transformer_layers"]

        # FFN multipliers for transformer layer
        ffn_multipliers = cfg["ffn_multiplier_per_layer"]
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        if not isinstance(ffn_multipliers, Sequence):
            logger.error(
                "{} expects FFN multipliers as a list, whose length is the same as"
                " number of transformer layers. Got: {}".format(
                    self.__class__.__name__, type(ffn_multipliers)
                )
            )
        elif (
            isinstance(ffn_multipliers, Sequence)
            and len(ffn_multipliers) != n_transformer_layers
        ):
            logger.error(
                "We need FFN multiplier for each transformer layer. Got {} ffn"
                " multipliers while number of transformer layers = {}".format(
                    len(ffn_multipliers), n_transformer_layers
                )
            )
        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0)
            for ffn_mult in ffn_multipliers
        ]

        # Heads for transformer layers
        mha_heads = cfg["n_heads_per_layer"]
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if not isinstance(mha_heads, Sequence):
            logger.error(
                "{} expects MHA heads as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(mha_heads)
                )
            )
        elif isinstance(mha_heads, Sequence) and len(mha_heads) != n_transformer_layers:
            logger.error(
                "{} needs MHA heads for each transformer layer. Got {} mha heads while"
                " number of transformer layers = {}".format(
                    self.__class__.__name__, len(mha_heads), n_transformer_layers
                )
            )

        if variant == "base":
            self.transformer = nn.ModuleList(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
        elif variant == "mct":
            self.transformer = nn.ModuleList([RepMixerBlock(dim=model_dim)])
            self.transformer.extend(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
            self.transformer.extend([RepMixerBlock(dim=model_dim)])
        else:
            raise ValueError("Unrecognized text encoder variant {}".format(variant))

        self.final_layer_norm = get_normalization_layer(
            num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = nn.Parameter(
            torch.empty(model_dim, self.projection_dim)
        )
        self.model_dim = model_dim
        self.causal_masking = cfg["causal_masking"]

    def forward_embedding(self, text_tokens: Tensor) -> Tensor:
        """Return text embedding for all tokens.

        Args:
            text_tokens: a tensor of token indices. Shape: [batch_size, context_length]

        Returns:
            A tensor of [batch_size, context_length, hidden_dim].
        """
        # [batch_size, context_length] --> [batch_size, context_length, hidden_dim]
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(
                token_emb.dtype
            )
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int) -> Tensor:
        """Build causal attention mask [batch_size, context_length, context_length]."""
        # Build mask with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(0)  # add dummy batch dimension
        mask = mask.expand(batch_size, -1, -1)
        return mask

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        """Return text token embeddings.

        Args:
            text_tokens: a tensor of token indices. Shape: [batch_size, context_length]
            key_padding_mask: a tensor of boolean values as the padding mask.
                Shape: [batch_size, context_length]
            return_all_tokens: a boolean flag to return all tokens, defaults to False
                to return only EOT token embedding.
        Returns:
            A tensor of [batch_size, context_length, hidden_dim] if return_all_tokens is
            True, otherwise a tensor of [batch_size, hidden_dim].
        """
        # Discrete tokens to continuous embeddings
        # [batch_size, context_length] --> [batch_size, context_length, hidden_dim]
        token_emb = self.forward_embedding(text_tokens)

        # [1, context_length, context_length]
        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
            key_padding_mask = None

        for layer in self.transformer:
            token_emb = layer(
                token_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        # Apply layer norm
        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        token_emb = token_emb[
            torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
        ]

        token_emb = token_emb @ self.projection_layer
        return token_emb

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        # Image-text pair data with single caption
        # [B, CL] --> [B, d]
        text_tokens = self.encode_text(
            text_tokens=text_tokens,
            key_padding_mask=key_padding_mask,
            return_all_tokens=return_all_tokens,
            *args,
            **kwargs
        )
        return text_tokens


# --------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import copy
from functools import partial
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models import register_model

from mobileclip.modules.common.mobileone import MobileOneBlock
from mobileclip.modules.image.replknet import ReparamLargeKernelConv


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
            use_se: If ``True`` SE block will be used.
        """
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)

    return blocks


class FastViT(nn.Module):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        **kwargs,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes
        if pos_embs is None:
            pos_embs = [None] * len(layers)

        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        use_se=se_downsamples[i + 1],
                    )
                )
        self.network = nn.ModuleList(network)

        # Classifier head
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.head = (
            nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.network):
            x = block(x)
        # output only the features of last layer for image classification
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        # for image classification
        x = self.conv_exp(x)
        cls_out = self.head(x)
        return cls_out


@register_model
def mci0(pretrained=False, **kwargs):
    """Instantiate MCi0 model variant."""
    layers = [2, 6, 10, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        se_downsamples=se_downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def mci1(pretrained=False, **kwargs):
    """Instantiate MCi1 model variant."""
    layers = [4, 12, 20, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        se_downsamples=se_downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def mci2(pretrained=False, **kwargs):
    """Instantiate MCi2 model variant."""
    layers = [4, 12, 24, 4]
    embed_dims = [80, 160, 320, 640]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        se_downsamples=se_downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


# --------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Implementation of the following modules is borrowed from ml-cvnets repo:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/vit.py

Please see ACKNOWLEDGEMENTS for license details.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from timm.models import register_model
from mobileclip.modules.common.transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)
from mobileclip.modules.image.image_projection import SimpleImageProjectionHead
from mobileclip import logger


class ConvNormAct(nn.Module):
    """
    Applies an N-dimensional convolution over an input.

    Args:
        cfg: Model configuration.
        in_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        out_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.
        kernel_size: Kernel size for convolution. An integer, or tuple of length ``N``.
        stride: Stride for convolution. An integer, or tuple of length ``N``. Default: 1.
        dilation: Dilation rate for convolution. An integer, or tuple of length ``N``.
            Default: ``1``.
        padding: Padding for convolution. An integer, or tuple of length ``N``.
            If not specified, padding is automatically computed based on kernel size and
            dilation range. Default : ``None`` (equivalent to ``[
            int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(N)]``).
        groups: Number of groups in convolution. Default: ``1``.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular').
            Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
            Default: ``True``.
        norm_layer: If not None, the provided normalization layer object will be used.
            Otherwise, a normalization object will be created based on config
            ``model.normalization.*`` opts.
        act_layer: If not None, the provided activation function will be used.
            Otherwise, an activation function will be created based on config
            ``model.activation.*`` opts.

    Shape:
        - Input: :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        - Output: :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        cfg: Dict,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ndim = 2

        if norm_layer is None and use_norm:
            norm_type = cfg.get("normalization", "batch_norm")
            if norm_type == "batch_norm":
                norm_layer = nn.BatchNorm2d(
                    num_features=out_channels,
                    momentum=cfg.get("momentum", 0.1),
                )
            else:
                norm_layer = get_normalization_layer(
                    num_features=out_channels, norm_type=norm_type
                )
        elif norm_layer is not None and use_norm:
            logger.error(
                f"When use_norm is False, norm_layer should be None, but norm_layer={norm_layer} is provided."
            )

        if act_layer is None and use_act:
            act_layer = nn.GELU()  # Default to GELU
        elif act_layer is not None and use_act:
            logger.error(
                f"When use_act is False, act_layer should be None, but act_layer={act_layer} is provided."
            )

        if (
            use_norm
            and any(param[0] == "bias" for param in norm_layer.named_parameters())
            and bias
        ):
            assert (
                not bias
            ), "Do not use bias when using normalization layers with bias."

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.ndim

        if isinstance(stride, int):
            stride = (stride,) * self.ndim

        if isinstance(dilation, int):
            dilation = (dilation,) * self.ndim

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(self.ndim)
            )

        if in_channels % groups != 0:
            logger.error(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            logger.error(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            padding=padding,
            dilation=dilation,  # type: ignore
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        if use_act:
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VisionTransformer(nn.Module):
    """
    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_. Our model implementation
    is inspired from `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Our positional encoding implementation allows us to use ViT with any multiple input scales
        3. We do not use StochasticDepth
        4. We do not add positional encoding to class token (if enabled), as suggested in `DeiT-3 paper <https://arxiv.org/abs/2204.07118>`_
    """

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        image_channels = 3
        num_classes = cfg.get("n_classes", 1000)

        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        kernel_sizes_conv_stem = [4, 2, 2]
        strides_conv_stem = [4, 2, 2]

        # Typically, in the ImageNet dataset, we use 224x224 as a resolution.
        # For out ViT implementation, patch size is 16 (16 = 4 * 2 * 2)
        # Therefore, total number of embeddings along width and height are (224 / 16)^2
        num_embeddings = (224 // 16) ** 2

        embed_dim = cfg["embed_dim"]
        ffn_dim = cfg["embed_dim"] * 4
        pos_emb_drop_p = cfg.get("pos_emb_drop_p", 0.0)
        n_transformer_layers = cfg["n_transformer_layers"]
        num_heads = cfg["n_attn_heads"]
        attn_dropout = cfg.get("attn_dropout", 0.0)
        dropout = cfg.get("dropout", 0.0)
        ffn_dropout = cfg.get("ffn_dropout", 0.0)
        norm_layer = cfg.get("norm_layer", "layer_norm")

        conv_stem_proj_dim = max(32, embed_dim // 4)
        patch_emb = [
            ConvNormAct(
                cfg=cfg,
                in_channels=image_channels,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[0],
                stride=strides_conv_stem[0],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvNormAct(
                cfg=cfg,
                in_channels=conv_stem_proj_dim,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[1],
                stride=strides_conv_stem[1],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvNormAct(
                cfg=cfg,
                in_channels=conv_stem_proj_dim,
                out_channels=embed_dim,
                kernel_size=kernel_sizes_conv_stem[2],
                stride=strides_conv_stem[2],
                bias=True,
                use_norm=False,
                use_act=False,
            ),
        ]

        self.patch_emb = nn.Sequential(*patch_emb)

        use_cls_token = not cfg.get("no_cls_token", False)
        stochastic_dropout = cfg.get("stochastic_dropout", 0.0)
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]
        transformer_blocks = [
            TransformerEncoder(
                embed_dim=embed_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=norm_layer,
                stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
            )
            for layer_idx in range(n_transformer_layers)
        ]

        self.post_transformer_norm = get_normalization_layer(
            num_features=embed_dim, norm_type=norm_layer
        )

        self.transformer = nn.Sequential(*transformer_blocks)

        if self.projection_dim is None:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = SimpleImageProjectionHead(embed_dim, self.projection_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = PositionalEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=None,
            interpolation_mode="bilinear",
        )
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # input is of shape [Batch, in_channels, height, width]. in_channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]

        # [Batch, in_channels, height, width] --> [Batch, emb_dim, num_patches_height, num_patches_width]
        patch_emb = self.patch_emb(x)
        n_h, n_w = patch_emb.shape[-2:]

        # [Batch, emb_dim, num_patches_height, num_patches_width] --> [Batch, emb_dim, num_patches]
        patch_emb = patch_emb.flatten(2)
        # [Batch, emb_dim, num_patches] --> [Batch, num_patches, emb_dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        n_patches = patch_emb.shape[1]
        # we resize the positional encodings dynamically.
        pos_emb = self.pos_embed(n_patches).to(patch_emb.dtype)

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, emb_dim] --> [Batch, 1, emb_dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concat([Batch, 1, emb_dim], [Batch, num_patches, emb_dim]) --> [Batch, num_patches + 1, emb_dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        # dropout
        patch_emb = self.emb_dropout(patch_emb)
        return patch_emb, (n_h, n_w)

    def _features_from_transformer(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tuple[int, int]]:
        # this function extract patch embeddings and then apply transformer module to learn
        # inter-patch representations

        # [B, N, C] --> [N, B, embed_dim], where B is batch size, N is number of tokens,
        # and embed_dim is feature dim
        x, (n_h, n_w) = self.extract_patch_embeddings(x)

        for layer in self.transformer:
            x = layer(x)
        x = self.post_transformer_norm(x)

        return x, (n_h, n_w)

    def extract_features(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # The extract_features function for ViT returns two outputs: (1) embedding corresponding to CLS token
        # and (2) image embeddings of the shape [B, C, h//o, w//o], where the value of o is typically 16.
        return_image_embeddings = kwargs.get("return_image_embeddings", False)

        # [B, C, H, W] --> [B, N + 1, embed_dim] or [B, N, embed_dim]
        # here, B is batch size, C is input channels
        # H and W are input height and width
        # N is the number of pixels (or tokens) after processing input with conv stem and reshaping
        # We add +1 for cls token (if applicable)
        # embed_dim --> embedding dimension
        x, (n_h, n_w) = self._features_from_transformer(x, *args, **kwargs)

        if self.cls_token is not None:
            # [B, N + 1, embed_dim] --> [B, embed_dim], [B, N, embed_dim]
            cls_embedding, image_embedding = torch.split(
                x, split_size_or_sections=[1, x.shape[1] - 1], dim=1
            )
            cls_embedding = cls_embedding.squeeze(1)
        else:
            # [B, N, embed_dim] -> [B, embed_dim]
            cls_embedding = torch.mean(x, dim=1)
            # [B, N, embed_dim]
            image_embedding = x

        if return_image_embeddings:
            # reshape image embedding to 4-D tensor
            # [B, N, C] --> [B, C, N]
            image_embedding = image_embedding.transpose(1, 2).contiguous()
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], -1, n_h, n_w
            )

            return cls_embedding, image_embedding
        else:
            return cls_embedding, None

    def forward_classifier(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        cls_embedding, image_embedding = self.extract_features(x, *args, **kwargs)
        # classify based on CLS token
        cls_embedding = self.classifier(cls_embedding)
        return cls_embedding, image_embedding

    def forward(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        # In ViT model, we can return either classifier embeddings (logits) or image embeddings or both.
        # To return the image embeddings, we need to set keyword argument (return_image_embeddings) as True.
        if kwargs.get("return_image_embeddings", False):
            out_dict = dict()
            prediction, image_embedding = self.forward_classifier(x, *args, **kwargs)
            out_dict.update({"logits": prediction})
            if image_embedding is not None:
                out_dict.update({"image_embeddings": image_embedding})
            return out_dict
        else:
            prediction, _ = self.forward_classifier(x, *args, **kwargs)
            return prediction


@register_model
def vit_b16(pretrained=False, **kwargs):
    # Vision transformer config
    cfg = {
        "norm_layer": "layer_norm_fp32",
        "act_layer": "gelu",
        "embed_dim": 768,
        "n_transformer_layers": 12,
        "n_attn_heads": 12,
    }
    model = VisionTransformer(cfg=cfg, **kwargs)
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


# --------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Union, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileOneBlock", "reparameterize_model"]


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            groups: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups

                kernel_size = self.kernel_size
                if isinstance(self.kernel_size, int):
                    kernel_size = (self.kernel_size, self.kernel_size)

                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Implementation of the following modules is borrowed from ml-cvnets repo:
https://github.com/apple/ml-cvnets/blob/main/cvnets/layers/multi_head_attention.py
https://github.com/apple/ml-cvnets/blob/main/cvnets/text_encoders/transformer.py

Please see ACKNOWLEDGEMENTS for license details.
"""

from typing import List, Optional, Union

import torch
from torch import Size, Tensor, nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

from mobileclip import logger


class LayerNormFP32(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Convert input from dtype X to FP32 and perform normalization operation.
        # This may help with underflow/overflow issues that we typically see with normalization layers
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


def get_normalization_layer(norm_type, num_features):
    if norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "layer_norm_fp32":
        return LayerNormFP32(num_features)
    else:
        raise NotImplementedError(f"Option: {norm_type} not supported.")


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        # Add other pos embedding here and logic to choose between them
        module = LearnablePositionalEmbedding

        self.pos_embed = module(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs,
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)

    def __repr__(self):
        return self.pos_embed.__repr__()


class LearnablePositionalEmbedding(nn.Module):
    """Learnable Positional embedding"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.interpolation_mode = interpolation_mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        # scale pos embedding
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        # Input is of the form [Batch, Seq_len, Embedding_dim]
        return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return "{}(num_embeddings={}, embedding_dim={}, padding_idx={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
        )


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def _forward_impl(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # [Batch , Sequence, Hidden_dim]
        return self._forward_impl(
            x_q=x_q,
            x_kv=x_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        # Build attention layer
        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True,
        )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        act_name = nn.GELU()
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

        self.drop_path = nn.Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                logger.error(
                    "Stochastic dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(stochastic_dropout, dropout)
                )
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode="row")

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs,
        )  # mha

        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth
        x = x + res

        # Feed forward network
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x


# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mobileclip import logger


class GlobalPool(nn.Module):
    """
    This layers applies global pooling over a 4D or 5D input tensor

    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            logger.error(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class GlobalPool2D(nn.Module):
    """This class implements global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 4
        ), "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
            x.shape
        )

        # [batch, in_dim, in_height, in_width] --> [batch, in_dim]
        x = self.pool(x)
        # [batch, in_dim]  x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        return x


class SimpleImageProjectionHead(nn.Module):
    """This class implements linear projection head."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 2
        ), "Input should be 2-dimensional (Batch x in_dim). Got: {}".format(x.shape)

        # [batch, in_dim] x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        return x


# ---------------------------------------------------------------------------

#
# For acknowledgement see accompanying ACKNOWLEDGEMENTS file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
from typing import Tuple

import torch
import torch.nn as nn

from timm.models.layers import SqueezeExcite

__all__ = ["ReparamLargeKernelConv"]


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation module. Default: ``nn.GELU``
        """
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        # Check if SE is requested
        if use_se:
            self.se = SqueezeExcite(out_channels, rd_ratio=0.25)
        else:
            self.se = nn.Identity()

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        return self.activation(self.se(out))

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            A nn.Sequential Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Optional

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
from mobileclip.modules.common.mobileone import MobileOneBlock


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        context_size: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            context_size: Context size for 1D signals.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, int(context_size)),
                padding=(0, int(context_size // 2)),
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(1, self.kernel_size),
                stride=1,
                padding=(0, self.kernel_size // 2),
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=(1, self.kernel_size),
            stride=1,
            padding=(0, self.kernel_size // 2),
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        *args,
        **kwargs,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            context_size=kernel_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x, *args, **kwargs):
        if x.dim() == 3:
            # B, C, D --- where C is the context length
            # Convert to B, D, C --- to match RepMixer impl.
            x = x.permute(0, 2, 1)
            x = torch.unsqueeze(x, dim=2)
        else:
            raise ValueError(
                f"Expected tensor of dim=3, obtained tensor of dim={x.dim()}"
            )

        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))

        # Convert tensors back
        x = x.squeeze(dim=2).permute(0, 2, 1)
        return x


# ---------------------------------------------------------------------------

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Dict

import open_clip
from torch import Tensor, nn


class ClipTokenizer(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.context_length = cfg["text_cfg"]["context_length"]
        model_name = getattr(cfg["text_cfg"], "open_clip_tokenizer", "ViT-B-16")
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_vocab_size(self) -> int:
        return len(self.tokenizer.encoder)

    def get_encodings(self) -> Dict[str, int]:
        return self.tokenizer.encoder

    def get_eot_token(self) -> int:
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        return self.tokenizer("")[1]

    def get_sot_token(self) -> int:
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        return self.tokenizer("")[0]

    def forward(self, input_sentence: str, *args, **kwargs) -> Tensor:
        # tokenizer returns indices as a string
        tokenized_sentence = self.tokenizer(input_sentence, self.context_length)
        assert (
            tokenized_sentence.shape[-1] == self.context_length
        ), "Tokenized tensor should be exactly `context_length` long."
        return tokenized_sentence


# ---------------------------------------------------------------------------