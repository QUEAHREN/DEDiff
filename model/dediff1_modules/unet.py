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
        super(BasicConv, self).__init__()
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
        _, _, w1, h1 = event.size()
        _, _, w2, h2 = x.size()
        x = self.x_feature_extract[0](x)
        event_feature = self.event_feature_extract[0](event)
        x_2 = x
        event_feature = F.interpolate(event_feature, scale_factor=h2 / h1)
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


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self.fusion = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, eve):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #b, c, h*w/4
        g_x = g_x.permute(0, 2, 1)

        theta_eve = self.theta(eve).view(batch_size, self.inter_channels, -1)   #b, c, h*w
        theta_eve = theta_eve.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   #b, c, h*w/4
        f = torch.matmul(theta_eve, phi_x)   # (b, h*w, c)  X  (b, c, h*w/4)   =  (b, h*w, h*w/4)
        N = f.size(-1)
        f_div_C = f / N
        f_div_C = self.softmax(f_div_C)

        y = torch.matmul(f_div_C, g_x)    # (b, h*w, h*w/4) X (b, h*w/4, c) = (b, h*w, c)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])   # b,c,h,w
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            use_ef=False,
            use_sfb=False,
            use_nlnn=False,
            **kwargs
    ):
        super().__init__()
        self.use_ef = use_ef
        self.use_sfb = use_sfb
        self.use_nlnn = use_nlnn
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
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                pre_channel = channel_mult
            downs.append(Downsample(pre_channel))
            if not is_last:
                feat_channels.append(channel_mult)

                # feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        if self.use_ef:
            self.tmb = TMB(pre_channel // 8)
            self.event_feature_extract = nn.Sequential(  # 提取事件特征
                BasicConv(pre_channel // 8, pre_channel, kernel_size=1, stride=1, relu=True),
            )
        else:
            self.tmb = TMB(pre_channel)

        if self.use_sfb:
            self.sfb = SFB(pre_channel, pre_channel, thita=1e-3)
        elif self.use_nlnn:
            self.transform1 = NONLocalBlock2D(pre_channel, inner_channel)
        else:
            self.feature_extract = nn.Sequential(  # 提取事件特征
                BasicConv(pre_channel * 2, pre_channel, kernel_size=1, stride=1, relu=True),
            )

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        channel_mults = [x - 1 for x in channel_mults]
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                pre_channel = channel_mult
            ups.append(Upsample(pre_channel))
            if not is_last:
                pre_channel += feat_channels.pop()

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        event = x[:, range(6, 12), :, :]
        x = x[:, range(0, 6), :, :]

        event = self.tmb(event)
        if self.use_ef:
            event = self.event_feature_extract(event)
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
                if not layer == self.downs[len(self.downs) - 1]:
                    feats.append(x)

        x = self.mid[0](x, t)
        if self.use_sfb:
            x = self.sfb(x, event)
        elif self.use_nlnn:
            _, _, _, h = x.size()
            event = F.interpolate(event, [h, h])
            x = self.transform1(x, event)

        else:
            _, _, _, h = x.size()
            event = F.interpolate(event, [h, h])
            x = torch.cat((x, event), dim=1)
            x = self.feature_extract[0](x)
        x = self.mid[1](x, t)
        # for layer in self.mid:
        #     if isinstance(layer, ResnetBlocWithAttn):
        #         x = layer(x, t)
        #     else:
        #         x = layer(x)

        begin = False
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                if begin:
                    x1 = feats.pop()
                    x = layer(torch.cat((x, x1), dim=1), t)
                    begin = False
                else:
                    x = layer(x, t)

            else:
                x = layer(x)
                begin = True

        return self.final_conv(x)


if __name__ == '__main__':
    model1 = UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=[2, 3, 4],
        res_blocks=2,
        dropout=0.2,
        use_ef=True,
        use_nlnn=False
    )
    device = torch.device(
        'cuda' if [0] is not None else 'cpu')
    input_img = torch.zeros(2, 12, 64, 64).to(device)
    model1.to(device)
    output_img = model1(input_img, torch.zeros(2).to(device))
    print("output_img", output_img.shape)

    # model2 = TMB(out_channel=16, thita=0.1)
    # input_event = torch.zeros(4, 6, 128, 128)
    # output_event = model2(input_event)
    # print("output_event", output_event.shape)
