import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class TMB(nn.Module):
    def __init__(self, out_channel, thita=1e-4):
        super(TMB, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.query_conv = nn.Sequential(  # 提取事件特征
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        #########################################################
        #########################################################
        self.event_feature_extract = nn.Sequential(  # 提取事件特征
            BasicConv(out_channel * 2, self.out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel * 3, self.out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.INF = INF

        self.conv_neighbor = BasicConv(4, self.out_channel * 2, kernel_size=1, stride=1, relu=True)
        self.conv_pro_key = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_pro_val = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_lat_key = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_lat_val = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_now1 = BasicConv(2, self.out_channel, kernel_size=1, stride=1, relu=True)
        # self.conv_now2 = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)

        self.conv_tmp1 = BasicConv(self.out_channel * 2, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_tmp2 = BasicConv(self.out_channel * 2, self.out_channel, kernel_size=1, stride=1, relu=True)

        self.conv_now_key = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_now_val = BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True)

    def forward(self, event):
        event_pro = event[:, range(0, 2), :, :]
        event_now = event[:, range(2, 4), :, :]
        event_lat = event[:, range(4, 6), :, :]

        event_pro = torch.as_tensor(event_pro)
        event_now = torch.as_tensor(event_now)
        event_lat = torch.as_tensor(event_lat)
        event_neighbor = torch.cat((event_pro, event_lat), 1)
        event_neighbor = self.conv_neighbor(event_neighbor)
        event_pro_key = self.conv_pro_key(event_neighbor[:, range(0, self.out_channel), :, ])
        event_pro_val = self.conv_pro_val(event_neighbor[:, range(self.out_channel, self.out_channel * 2), :, :])
        event_lat_key = self.conv_lat_key(event_neighbor[:, range(0, self.out_channel), :, :])
        event_lat_val = self.conv_lat_val(event_neighbor[:, range(self.out_channel, self.out_channel * 2), :, :])

        event_key_neighbor = torch.cat((event_pro_key, event_lat_key), 1)
        event_val_neighbor = torch.cat((event_pro_val, event_lat_val), 1)
        event_key_neighbor = self.conv_tmp1(event_key_neighbor)
        event_val_neighbor = self.conv_tmp1(event_val_neighbor)

        event_now = self.conv_now1(event_now)

        event_now_key = self.conv_now_key(event_now)
        event_now_val = self.conv_now_val(event_now)

        ################################################################
        #################################################################
        m_batchsize, C, width, height = event_key_neighbor.size()
        proj_query = self.query_conv(event_key_neighbor).view(m_batchsize, -1, width * height).permute(0, 2,
                                                                                                       1)  # B X CX(N)
        proj_key = self.key_conv(event_now_key).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(event_val_neighbor).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.thita * out + event_now_val

        return out


class SFB(nn.Module):
    def __init__(self, in_channel, out_channel, thita=1e-4):
        super(SFB, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.query_conv = nn.Sequential(  # 提取事件特征
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        #########################################################
        #########################################################
        self.event_feature_extract = nn.Sequential(  # 提取事件特征
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel * 1, self.out_channel, kernel_size=1, stride=1, relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.INF = INF

    def forward(self, x, event):

        x = self.x_feature_extract[0](x)
        event_feature = self.event_feature_extract[0](event)
        x_2 = x
        event_feature = F.interpolate(event_feature, scale_factor=0.0625)
        x_2 = self.x_feature_extract[1](x_2)
        event_feature = self.event_feature_extract[1](event_feature)

        ################################################################
        #################################################################

        m_batchsize, C, width, height = x_2.size()
        proj_query = self.query_conv(x_2).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(event_feature).view(m_batchsize, -1, width * height)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x_2).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = F.interpolate(out, scale_factor=4)
        out = self.gamma * out

        out = self.thita * out + x
        # event_feature = F.interpolate(event_feature, scale_factor=4)
        return out


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.tmb = TMB(pre_channel)
        self.sfb = SFB(pre_channel, pre_channel, thita=1e-3)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        event = x[:, range(6, 12), :, :]
        x = x[:, range(0, 6), :, :]
        event = self.tmb(event)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        x = self.mid[0](x, t)
        x = self.sfb(x, event)
        x = self.mid[1](x, t)
        # for layer in self.mid:
        #     if isinstance(layer, ResnetBlocWithAttn):
        #         x = layer(x, t)
        #     else:
        #         x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



if __name__ == '__main__':
    model1 = UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=[1, 2, 4, 8, 8],
        attn_res=[16],
        res_blocks=2,
        dropout=0.2,
        image_size=128
    )


    input_img = torch.zeros(4, 12, 128, 128)
    output_img = model1(input_img, torch.zeros(4))
    print("output_img", output_img.shape)

    model2 = TMB(out_channel=16, thita=0.1)
    input_event = torch.zeros(4, 6, 128, 128)
    output_event = model2(input_event)
    print("output_event", output_event.shape)