import torch
import torch.nn as nn
import torchvision.models as models

from models.networks.base.base_network import BaseNetwork


class SPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=True)

        # input is concatenation of rgb and (inferred) mask
        self.model.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # estimating (w_r, w_g, w_b, b_r, b_g, b_b)
        self.model.fc = nn.Linear(2048, 6)

    def forward(self, rgb, mask):
        h = torch.cat([rgb, mask], dim=1)
        h = self.model(h)
        h = torch.split(h, 3, dim=1)
        return h


class MNet(nn.Module):
    # See sec 7.2 in the supplementary material
    def __init__(self, opt, in_ch=7, out_ch=3, no_last_sigmoid=False):
        super().__init__()
        self.opt = opt
        if opt.norm_G == 'batch':
            norm = nn.BatchNorm2d
        elif opt.norm_G == 'instance':
            norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError

        self.no_last_sigmoid = no_last_sigmoid
        # input is concatenation of rgb, relit, and (inferred) mask
        self.first_conv = nn.Conv2d(in_ch, 64, 4, 2, 1, bias=False)
        self.first_bn = norm(64)
        self.first_relu = nn.LeakyReLU(negative_slope=0.2)

        def set_down_block(in_channels, out_channels):
            sequence = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4,
                          stride=2, padding=1, bias=False),
                norm(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            ]
            return nn.Sequential(*sequence)

        def set_up_block(in_channels, out_channels):
            sequence = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                norm(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            ]
            return nn.Sequential(*sequence)

        setattr(self, 'down_block1', set_down_block(64, 128))
        setattr(self, 'down_block2', set_down_block(128, 256))
        setattr(self, 'down_block3', set_down_block(256, 512))
        setattr(self, 'down_block4', set_down_block(512, 512))

        setattr(self, 'up_block1', set_up_block(256, 64))
        setattr(self, 'up_block2', set_up_block(512, 128))
        setattr(self, 'up_block3', set_up_block(1024, 256))
        setattr(self, 'up_block4', set_up_block(512, 512))

        self.last_conv = nn.ConvTranspose2d(128, out_ch, 4, 2, 1, bias=True)

    def forward(self, inputs):
        h = torch.cat(inputs, dim=1)

        h = self.first_relu(self.first_bn(self.first_conv(h)))
        h_list = [h]

        for i in range(1, 5):
            h_list.append(getattr(self, f'down_block{i}')(h_list[-1]))

        h = torch.cat([h_list[3], self.up_block4(h_list[4])], dim=1)
        h = torch.cat([h_list[2], self.up_block3(h)], dim=1)
        h = torch.cat([h_list[1], self.up_block2(h)], dim=1)
        h = torch.cat([h_list[0], self.up_block1(h)], dim=1)
        h = self.last_conv(h)
        if not self.no_last_sigmoid:
            h = torch.sigmoid(h)
        return h


class SPMGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.spnet = SPNet()
        self.mnet = MNet(opt)

    # def forward(self, rgb, mask):
    #     B, C, H, W = rgb.size()

    #     w, b = self.spnet(rgb, mask)
    #     lit = w.view(B, C, 1, 1).repeat(1, 1, H, W) * rgb + \
    #         b.view(B, C, 1, 1).repeat(1, 1, H, W)
    #     matte = self.mnet(rgb, lit, mask)
    #     result = (1.0 - matte) * rgb + matte * lit
    #     result = torch.clamp(result, min=0.0, max=1.0)
    #     return {'img': result, 'lit': lit, 'matte': matte, 'w': w, 'b': b}

    @staticmethod
    def relit(rgb, w, b):
        B, C, H, W = rgb.size()
        result = w.view(B, C, 1, 1).repeat(1, 1, H, W) * rgb + \
            b.view(B, C, 1, 1).repeat(1, 1, H, W)
        return result

    @staticmethod
    def compose(rgb, lit, matte):
        assert rgb.size() == lit.size() == matte.size()
        result = (1.0 - matte) * rgb + matte * lit
        result = torch.clamp(result, min=0.0, max=1.0)
        return result

    def forward(self, rgb, mask):
        w, b = self.spnet(rgb, mask)

        lit = self.relit(rgb, w, b)
        matte = self.mnet([rgb, lit, mask])
        result = self.compose(rgb, lit, matte)
        return {'img': result, 'lit': lit, 'matte': matte, 'w': w, 'b': b}

    def init_weights(self, init_type='normal', gain=0.02):
        # for skipping BaseNetwork.init_weights
        pass
