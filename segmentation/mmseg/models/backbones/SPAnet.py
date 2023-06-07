import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def peripheral_crop(x, level):
    """
    Crop the central region. Gradually increase the feature map range as the peripheral level increases.
    """
    _, _, h, w = x.shape
    crop_ratio = 2 ** (level + 2)
    crop_long = h - h // crop_ratio

    if (crop_long % 2 == 1) & (h % 2 == 0):  # If the crop is odd and the height and width of x is even
        crop_long -= 1  #
    elif (crop_long % 2 == 0) & (h % 2 == 1):  # If the crop is even and the height and width of x is odd
        crop_long -= 1

    start = (h - crop_long) // 2
    return x[:, :, start:start + crop_long, start:start + crop_long]


class SPA(nn.Module):
    """
    The SPA approach mimics biological peripheral vision by combining the advantages of self-attention
    and peripheral vision to model feature information from the central and peripheral areas of the visual field.
    """

    def __init__(self, dim, kernel,  peripheral_level, dilate_ratio=1, bias=True, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.kernel = kernel
        self.peripheral_level = peripheral_level
        self.linear = nn.Linear(dim, 2 * dim + dim // self.peripheral_level, bias=False)
        self.interact = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.peripheral_layers = nn.ModuleList()
        self.dilate_ratio = dilate_ratio

        for k in range(self.peripheral_level):
            dilate_rate = self.dilate_ratio + k
            self.peripheral_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim // self.peripheral_level, dim // self.peripheral_level, kernel_size=self.kernel,
                              stride=1, dilation=dilate_rate, groups=dim // self.peripheral_level,
                              padding='same', bias=True),  # DepthDilatedConv
                    LayerNorm(dim // self.peripheral_level, data_format="channels_first")
                )
            )

    def forward(self, x):
        """
            x: input features with shape of (B, H, W, C)
        """
        _, h, w, c = x.shape

        # linear projection
        x = self.linear(x).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        q, k, v = torch.split(x, [c // self.peripheral_level, c, c], 1)
        query_all = torch.zeros((q.shape[0], 0, h, w)).to(x.device)

        for level in range(self.peripheral_level):
            q = self.peripheral_layers[level](q)
            q_crop = q
            if level != self.peripheral_level - 1:
                q_crop = peripheral_crop(q_crop, level)
                _, _, croph, cropw = q_crop.shape
                pad_h = (h - croph) // 2
                pad_w = (w - cropw) // 2

                # use the pad function to fill the surrounding area with 0
                q_crop = torch.nn.functional.pad(q_crop, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

            query_all = torch.cat((query_all, q_crop), dim=1)

        # interaction
        query_all = self.interact(query_all)

        # hadmard product for score
        score = query_all * k

        # hadmard product
        x = v * score
        x = x.permute(0, 2, 3, 1).contiguous()

        # linear projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SPABlock(nn.Module):
    r""" SPA Block.
       Args:
           dim (int): Number of input channels.
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
           drop (float, optional): Dropout rate. Default: 0.0
           drop_path (float, optional): Stochastic depth rate. Default: 0.0
           act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
           norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
           peripheral_level (int): Number of peripheral levels.
           kernel (int): Kernel size at peripheral level.
           dilate_ratio (int): Number of first level ratio.
       """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 peripheral_level=2, kernel=9, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.kernel = kernel
        self.peripheral_level = peripheral_level
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim)
        self.modulation = SPA(
            dim, kernel=self.kernel, peripheral_level=self.peripheral_level, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # FM
        x = self.modulation(x).view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic SPA-Net layer for one stage.
      Args:
          dim (int): Number of input channels.
          depth (int): Number of blocks.
          mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
          drop (float, optional): Dropout rate. Default: 0.0
          drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
          norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
          downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
          use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
          peripheral_level (int): Number of peripheral levels.
          kernel (int): Kernel size at peripheral level.
          dilate_ratio (int): Number of first level ratio.
      """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 kernel=9,
                 peripheral_level=2,
                 use_conv_embed=False,
                 use_layerscale=False,
                 use_checkpoint=False
                 ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SPABlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                kernel=kernel,
                peripheral_level=peripheral_level,
                use_layerscale=use_layerscale,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim, embed_dim=2 * dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False
            )

        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_reshaped = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W)
            x_down = self.downsample(x_reshaped)
            x_down = x_down.flatten(2).transpose(1, 2)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, use_conv_embed=False, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7;
                padding = 3;
                stride = 4
            else:
                kernel_size = 3;
                padding = 1;
                stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


@BACKBONES.register_module()
class SPAnet(nn.Module):
    r""" Self Peripheral Attention Network (SPANet)
      Args:
          img_size (int | tuple(int)): Input image size. Default 224
          patch_size (int | tuple(int)): Patch size. Default: 4
          in_chans (int): Number of input image channels. Default: 3
          num_classes (int): Number of classes for classification head. Default: 1000
          embed_dim (int): Patch embedding dimension. Default: 96
          depths (tuple(int)): Depth of each SPA-Net layer.
          mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
          drop_rate (float): Dropout rate. Default: 0
          drop_path_rate (float): Stochastic depth rate. Default: 0.1
          norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
          patch_norm (bool): If True, add normalization after patch embedding. Default: True
          use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
          peripheral_levels (list): How many peripheral levels at all stages. Default: [4, 3, 2, 1]
          kernel (list): The kernel size at all stages. Default: [3, 3, 3, 7]
      """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 peripheral_levels=[4, 3, 2, 1],
                 kernels=[3, 3, 3, 7],
                 use_conv_embed=False,
                 use_layerscale=False,
                 use_checkpoint=False,
                 ):
        super().__init__()

        self.pretrain_img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            use_conv_embed=use_conv_embed, is_stem=True)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                kernel=kernels[i_layer],
                peripheral_level=peripheral_levels[i_layer],
                use_conv_embed=use_conv_embed,
                use_layerscale=use_layerscale,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        tic = time.time()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        toc = time.time()
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SPAnet, self).train(mode)
        self._freeze_stages()