import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.MoReT_3D.mobilevit_v3_block import MobileViTBlock
from core.models.MoReT_3D.vit_block import SimpleViT


def conv_1x1_gn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

def conv_nxn_gn(inp, oup, kernel_size=3, stride=1, padding=(0, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=padding, bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

class MV2Block(nn.Module):
    def __init__(self, inp, oup, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), stride=(1, 1, 1), padding=(0,1,1), groups=hidden_dim, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),

                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, groups=hidden_dim, bias=False),
                nn.GroupNorm(1, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),

                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=(1, stride, stride), padding=padding, groups=hidden_dim, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),

                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(1, oup)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.gn = nn.GroupNorm(1, int(n_channels))
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 4), padding=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        out = self.conv3(F.relu(self.gn(out)))
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_channels, n_output_channels, size=None, scale_factor=None):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode='nearest')
        self.conv1x1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return x


class MoReT_3D(nn.Module):
    def __init__(self, image_size, dims, channels, input_channel, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2, 2), device=torch.device("cuda:4")):
        super().__init__()
        i_s, ih, iw = image_size
        ps, ph, pw = patch_size
        assert i_s % ps == 0 and ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.device = device

        self.conv1 = conv_nxn_gn(input_channel, channels[0], stride=2).to(device)

        self.mv2 = nn.ModuleList([]).to(device)
        self.mv2.append(MV2Block(channels[0], channels[1], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[1], channels[2], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[3], channels[4], stride=1, expansion=expansion).to(device))  # repeat
        self.mv2.append(MV2Block(channels[4], channels[5], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[5], channels[6], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[7], channels[8], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[9], channels[10], stride=2, padding=(0, 0, 0), expansion=expansion).to(device))

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[6], kernel_size, patch_size, int(dims[0] * 2)).to(device))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[8], kernel_size, patch_size, int(dims[1] * 4)).to(device))

        self.convMtoN = conv_1x1_gn(channels[10], 128).to(device)
        self.vit = SimpleViT(dims[2], L[2], 128, (1, 6, 6), heads=4, dim_head=32).to(device)
        self.convNtoM = conv_1x1_gn(128, channels[10]).to(device)

        ''''''
        self.resblock1 = ResidualBlock(channels[8] + 256).to(device)  # 256 + 320
        self.resblock2 = ResidualBlock(channels[6] + 128).to(device)  # 128 + 256
        self.resblock3 = ResidualBlock(32 + channels[4]).to(device)  # 64 + 64
        self.resblock4 = ResidualBlock(16 + channels[2]).to(device)  # 52 + 52
        ''''''

        self.deconv1 = nn.ConvTranspose3d(channels[11], channels[10], kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 2, 2)).to(device)  # 640 + 320
        self.deconv2 = nn.ConvTranspose3d(2 * (channels[8] + channels[10]), channels[8], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2*(256 + 320), 256
        self.deconv3 = nn.ConvTranspose3d(2 * (channels[6] + channels[8]), channels[5], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (128 + 256), 64
        self.deconv4 = nn.ConvTranspose3d(2 * (2 * channels[4]), channels[2], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (64 + 64), 52
        self.deconv5 = nn.ConvTranspose3d(2 * (2 * channels[2]), 48, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (52 + 52), 48

        self.last_conv = nn.Conv3d(48, 1, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)

        conv1 = self.conv1(x)  # (1, 52, 20, 112, 112)

        mv2_0 = self.mv2[0](conv1)  # (1, 52, 20, 112, 112)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 52, 20, 112, 112)

        mv2_2 = self.mv2[2](mv2_1)  # (1, 64, 20, 56, 56)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 64, 20, 56, 56)
        mv2_4 = self.mv2[4](mv2_3)  # (1, 64, 20, 56, 56)
        mv2_5 = self.mv2[5](mv2_4)  # (1, 128, 20, 28, 28)
        mvit_0 = self.mvit[0](mv2_5)  # (1, 128, 20, 28, 28)
        mv2_6 = self.mv2[6](mvit_0)  # (1, 256, 20, 14, 14)
        mvit_1 = self.mvit[1](mv2_6)  # (1, 256, 20, 14, 14)
        mv2_7 = self.mv2[7](mvit_1)  # (1, 320, 20, 6, 6)
        convMtoN = self.convMtoN(mv2_7)
        vit = self.vit(convMtoN)
        convNtoM = self.convNtoM(vit)

        deconv1 = self.deconv1(convNtoM)  # (1, 320, 20, 14, 14)
        x = torch.cat((deconv1, mv2_6), dim=1)  # (1, 576, 20, 14, 14)
        res1 = self.resblock1(x)
        x = torch.cat((x, res1), dim=1)  # (1, 1152, 20, 14, 14)

        deconv2 = self.deconv2(x)  # (1, 256, 20, 28, 28)
        x = torch.cat((deconv2, mv2_5), dim=1)  # (1, 384, 20, 28, 28)
        res2 = self.resblock2(x)
        x = torch.cat((x, res2), dim=1)  # (1, 768, 20, 28, 28)

        deconv3 = self.deconv3(x)  # (1, 64, 20, 56, 56)
        x = torch.cat((deconv3, mv2_3), dim=1)  # (1, 128, 20, 56, 56)  # mv2_2
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)  # (1, 256, 10, 56, 56)
        deconv4 = self.deconv4(x)  # (1, 52, 20, 112, 112)
        x = torch.cat((deconv4, mv2_1), dim=1)  # (1, 104, 20, 112, 112)  #mv2_1
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)  # (1, 208, 20, 112, 112)
        deconv5 = self.deconv5(x) # (1, 48, 20, 224, 24)

        reg_output = self.last_conv(deconv5)

        return reg_output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

