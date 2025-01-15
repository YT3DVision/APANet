import numbers

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import init
from basicsr.models.arch_util import LayerNorm2d


class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / (1e9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class EAttention(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(EAttention, self).__init__()
        self.EAttentionBlock = ExternalAttention(dim, hidden_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.EAttentionBlock(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, stride=1, bias=bias,
                                groups=hidden_features)
        self.project_out = nn.Conv2d(hidden_features // 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(AttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = EAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.beta
        x = x + self.ffn(self.norm2(x)) * self.gamma
        return x


class SCAM(nn.Module):
    '''
    APAM
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        w_size = x_l.size(3)
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        mask = torch.triu(torch.ones((w_size, w_size), requires_grad=False, device=attention.device, dtype=torch.bool),
                          diagonal=1)
        attention = attention.masked_fill(mask == True, float('-inf'))
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        x_l = F_r2l.permute(0, 3, 1, 2) * self.beta + x_l
        x_r = F_l2r.permute(0, 3, 1, 2) * self.gamma + x_r

        return x_l, x_r


class DerainBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.spatial_block = AttentionBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.stereo_block = SCAM(dim)

    def forward(self, x_l, x_r):
        x_l = self.spatial_block(x_l)
        x_r = self.spatial_block(x_r)
        x_l_1, x_r_1 = self.stereo_block(x_l, x_r)
        return x_l_1, x_r_1


class DerainNet(nn.Module):
    def __init__(self, dim=32, num_blocks=None, num_heads=None, ffn_expansion_factor=2, bias=True,
                 LayerNorm_type='WithBias'):
        super().__init__()
        if num_heads is None:
            num_heads = [64, 64, 64, 64]
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        self.num_blocks = num_blocks
        self.project_in = nn.Conv2d(3, dim, kernel_size=3, padding=1, stride=1, bias=bias)
        self.encoder_level1 = sequentialMultiInput(*[
            DerainBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 3), num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = sequentialMultiInput(*[
            DerainBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.project_out = nn.Conv2d(dim * 2 ** 1, 3, kernel_size=3, padding=1, stride=1, bias=bias)

    def forward(self, x):
        x_l = x[:, :3, :, :]
        x_r = x[:, 3:, :, :]
        x_l = self.project_in(x_l)
        x_r = self.project_in(x_r)

        out_l_enc_level1, out_r_enc_level1 = self.encoder_level1(x_l, x_r)
        inp_l_enc_level2 = self.down1_2(out_l_enc_level1)
        inp_r_enc_level2 = self.down1_2(out_r_enc_level1)

        out_l_enc_level2, out_r_enc_level2 = self.encoder_level2(inp_l_enc_level2, inp_r_enc_level2)
        inp_l_enc_level3 = self.down2_3(out_l_enc_level2)
        inp_r_enc_level3 = self.down2_3(out_r_enc_level2)

        out_l_enc_level3, out_r_enc_level3 = self.encoder_level3(inp_l_enc_level3, inp_r_enc_level3)
        inp_l_enc_level4 = self.down3_4(out_l_enc_level3)
        inp_r_enc_level4 = self.down3_4(out_r_enc_level3)

        latent_l, latent_r = self.latent(inp_l_enc_level4, inp_r_enc_level4)

        inp_l_dec_level3 = self.up4_3(latent_l)
        inp_r_dec_level3 = self.up4_3(latent_r)
        inp_l_dec_level3 = torch.cat([inp_l_dec_level3, out_l_enc_level3], 1)
        inp_l_dec_level3 = self.reduce_chan_level3(inp_l_dec_level3)
        inp_r_dec_level3 = torch.cat([inp_r_dec_level3, out_r_enc_level3], 1)
        inp_r_dec_level3 = self.reduce_chan_level3(inp_r_dec_level3)
        out_l_dec_level3, out_r_dec_level3 = self.decoder_level3(inp_l_dec_level3, inp_r_dec_level3)

        inp_l_dec_level2 = self.up3_2(out_l_dec_level3)
        inp_r_dec_level2 = self.up3_2(out_r_dec_level3)
        inp_l_dec_level2 = torch.cat([inp_l_dec_level2, out_l_enc_level2], 1)
        inp_l_dec_level2 = self.reduce_chan_level2(inp_l_dec_level2)
        inp_r_dec_level2 = torch.cat([inp_r_dec_level2, out_r_enc_level2], 1)
        inp_r_dec_level2 = self.reduce_chan_level2(inp_r_dec_level2)
        out_l_dec_level2, out_r_dec_level2 = self.decoder_level2(inp_l_dec_level2, inp_r_dec_level2)

        inp_l_dec_level1 = self.up2_1(out_l_dec_level2)
        inp_r_dec_level1 = self.up2_1(out_r_dec_level2)
        inp_l_dec_level1 = torch.cat([inp_l_dec_level1, out_l_enc_level1], 1)
        inp_r_dec_level1 = torch.cat([inp_r_dec_level1, out_r_enc_level1], 1)
        out_l_dec_level1, out_r_dec_level1 = self.decoder_level1(inp_l_dec_level1, inp_r_dec_level1)

        x_l = self.project_out(out_l_dec_level1) + x[:, :3, :, :]
        x_r = self.project_out(out_r_dec_level1) + x[:, 3:, :, :]
        x = torch.cat([x_l, x_r], dim=1)
        return x

    def forward_feature(self, x):
        x_l = x[:, :3, :, :]
        x_r = x[:, 3:, :, :]
        x_l = self.project_in(x_l)
        x_r = self.project_in(x_r)

        out_l_enc_level1, out_r_enc_level1 = self.encoder_level1(x_l, x_r)
        inp_l_enc_level2 = self.down1_2(out_l_enc_level1)
        inp_r_enc_level2 = self.down1_2(out_r_enc_level1)

        out_l_enc_level2, out_r_enc_level2 = self.encoder_level2(inp_l_enc_level2, inp_r_enc_level2)
        inp_l_enc_level3 = self.down2_3(out_l_enc_level2)
        inp_r_enc_level3 = self.down2_3(out_r_enc_level2)

        out_l_enc_level3, out_r_enc_level3 = self.encoder_level3(inp_l_enc_level3, inp_r_enc_level3)
        inp_l_enc_level4 = self.down3_4(out_l_enc_level3)
        inp_r_enc_level4 = self.down3_4(out_r_enc_level3)

        latent_l, latent_r = self.latent(inp_l_enc_level4, inp_r_enc_level4)

        inp_l_dec_level3 = self.up4_3(latent_l)
        inp_r_dec_level3 = self.up4_3(latent_r)
        inp_l_dec_level3 = torch.cat([inp_l_dec_level3, out_l_enc_level3], 1)
        inp_l_dec_level3 = self.reduce_chan_level3(inp_l_dec_level3)
        inp_r_dec_level3 = torch.cat([inp_r_dec_level3, out_r_enc_level3], 1)
        inp_r_dec_level3 = self.reduce_chan_level3(inp_r_dec_level3)
        out_l_dec_level3, out_r_dec_level3 = self.decoder_level3(inp_l_dec_level3, inp_r_dec_level3)

        inp_l_dec_level2 = self.up3_2(out_l_dec_level3)
        inp_r_dec_level2 = self.up3_2(out_r_dec_level3)
        inp_l_dec_level2 = torch.cat([inp_l_dec_level2, out_l_enc_level2], 1)
        inp_l_dec_level2 = self.reduce_chan_level2(inp_l_dec_level2)
        inp_r_dec_level2 = torch.cat([inp_r_dec_level2, out_r_enc_level2], 1)
        inp_r_dec_level2 = self.reduce_chan_level2(inp_r_dec_level2)
        out_l_dec_level2, out_r_dec_level2 = self.decoder_level2(inp_l_dec_level2, inp_r_dec_level2)

        inp_l_dec_level1 = self.up2_1(out_l_dec_level2)
        inp_r_dec_level1 = self.up2_1(out_r_dec_level2)
        inp_l_dec_level1 = torch.cat([inp_l_dec_level1, out_l_enc_level1], 1)
        inp_r_dec_level1 = torch.cat([inp_r_dec_level1, out_r_enc_level1], 1)
        out_l_dec_level1, out_r_dec_level1 = self.decoder_level1(inp_l_dec_level1, inp_r_dec_level1)

        x_l = self.project_out(out_l_dec_level1) + x[:, :3, :, :]
        x_r = self.project_out(out_r_dec_level1) + x[:, 3:, :, :]
        x = torch.cat([x_l, x_r], dim=1)
        return [out_l_enc_level1, out_r_enc_level1, out_l_dec_level1, out_r_dec_level1]


if __name__ == '__main__':
    net = DerainNet()
    inp_shape = (6, 128, 128)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
