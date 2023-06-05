import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc. networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class Mlp(nn.Module):

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

    def __init__(self, dim, kernel, dilate_ratio, peripheral_level, bias=True, proj_drop=0.):
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

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 peripheral_level=1, kernel=3, dilate_ratio=1):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.kernel = kernel
        self.peripheral_level = peripheral_level
        self.norm1 = norm_layer(dim)
        self.SPA = SPA(dim, proj_drop=drop, kernel=kernel,
                       dilate_ratio=dilate_ratio, peripheral_level=self.peripheral_level)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape

        # SPA: Self Peripheral Attention
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.SPA(x).view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

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

    def __init__(self, dim, out_dim, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 peripheral_level=1, kernel=3, dilate_ratio=1):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SPABlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                peripheral_level=peripheral_level,
                kernel=kernel,
                dilate_ratio=dilate_ratio)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_c=dim,
                embed_dim=out_dim,
                norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # If the H, W of the input image is not an integer multiple of the patch_size, padding is required.
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # Downsampling
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SPANet(nn.Module):
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
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 peripheral_levels=(4, 3, 2, 1),
                 kernels=(3, 3, 3, 7),
                 dilate_ratio=1,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               out_dim=embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               peripheral_level=peripheral_levels[i_layer],
                               kernel=kernels[i_layer],
                               dilate_ratio=dilate_ratio,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def spa_tiny_224(num_classes: int = 1000, **kwargs):
    model = SPANet(depths=[2, 2, 2, 2],
                   peripheral_levels=[4, 3, 2, 1],
                   kernels=[3, 3, 3, 3],
                   dilate_ratio=1,
                   embed_dim=96,
                   num_classes=num_classes,
                   **kwargs)
    return model


def spa_small_224(num_classes: int = 1000, **kwargs):
    model = SPANet(depths=[2, 2, 6, 2],
                   peripheral_levels=[4, 3, 2, 1],
                   kernels=[3, 3, 3, 7],
                   dilate_ratio=1,
                   embed_dim=96,
                   num_classes=num_classes,
                   **kwargs)
    return model


def spa_base_224(num_classes: int = 1000, **kwargs):
    model = SPANet(depths=[2, 2, 18, 2],
                   peripheral_levels=[4, 3, 2, 1],
                   kernels=[3, 3, 3, 7],
                   dilate_ratio=1,
                   embed_dim=128,
                   num_classes=num_classes,
                   **kwargs)
    return model
