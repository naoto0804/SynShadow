# Distraction-Aware Shadow Detection (CVPR2019)
# Adapted from: https://quanlzheng.github.io/projects/Distraction-aware-Shadow-Detection.html
# Note: Distraction-aware Shadow (DS) loss is not used in this implementation

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.networks.base.resnext import ResNeXt101
from models.networks.base.base_network import BaseNetwork


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(), nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)

        return block2


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(64, 1, 3, bias=False, padding=1), nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.att(x)
        block2 = block1.repeat(1, 32, 1, 1)

        return block2


class DSDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess_mode='resize_and_crop',
                            load_size=320, crop_size=320)
        if is_train:
            parser.set_defaults(batch_size=10, niter=5000)
        return parser

    def init_weights(self, init_type='normal', gain=0.02):
        # for skipping BaseNetwork.init_weights
        pass

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, bias=False, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, bias=False, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, bias=False, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, bias=False, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.shad_att = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        self.dst1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.dst2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.refine4_hl = ConvBlock()
        self.refine3_hl = ConvBlock()
        self.refine2_hl = ConvBlock()
        self.refine1_hl = ConvBlock()

        self.refine0_hl = ConvBlock()

        self.attention4_hl = AttentionModule()
        self.attention3_hl = AttentionModule()
        self.attention2_hl = AttentionModule()
        self.attention1_hl = AttentionModule()
        self.attention0_hl = AttentionModule()
        self.conv1x1_ReLU_down4 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down2 = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down0 = nn.Sequential(
            nn.Conv2d(160, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.fuse_predict = nn.Sequential(
            nn.Conv2d(5, 1, 1, bias=False)
        )

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)
        down0 = self.down0(layer0)

        down4_dst1 = self.dst1(down4)
        # down4_dst1_3 = F.upsample(
        #     down4_dst1, size=down3.size()[2:], mode='bilinear')
        # down4_dst1_2 = F.upsample(
        #     down4_dst1, size=down2.size()[2:], mode='bilinear')
        # down4_dst1_1 = F.upsample(
        #     down4_dst1, size=down1.size()[2:], mode='bilinear')
        # down4_dst1_0 = F.upsample(
        #     down4_dst1, size=down0.size()[2:], mode='bilinear')
        down4_dst2 = self.dst2(down4)
        # down4_dst2_3 = F.upsample(
        #     down4_dst2, size=down3.size()[2:], mode='bilinear')
        # down4_dst2_2 = F.upsample(
        #     down4_dst2, size=down2.size()[2:], mode='bilinear')
        # down4_dst2_1 = F.upsample(
        #     down4_dst2, size=down1.size()[2:], mode='bilinear')
        # down4_dst2_0 = F.upsample(
        #     down4_dst2, size=down0.size()[2:], mode='bilinear')
        down4_shad = down4
        down4_shad = (1 + self.attention4_hl(
            torch.cat((down4_shad, down4_dst2), 1))) * down4_shad
        down4_shad = F.relu(-self.refine4_hl(
            torch.cat((down4_shad, down4_dst1), 1)) + down4_shad, True)

        down4_shad_3 = F.upsample(
            down4_shad, size=down3.size()[2:], mode='bilinear')
        down4_shad_2 = F.upsample(
            down4_shad, size=down2.size()[2:], mode='bilinear')
        down4_shad_1 = F.upsample(
            down4_shad, size=down1.size()[2:], mode='bilinear')
        down4_shad_0 = F.upsample(
            down4_shad, size=down0.size()[2:], mode='bilinear')
        # up_down4_dst1 = self.conv1x1_ReLU_down4(down4_dst1)
        # up_down4_dst2 = self.conv1x1_ReLU_down4(down4_dst2)
        up_down4_shad = self.conv1x1_ReLU_down4(down4_shad)
        # pred_down4_dst1 = F.upsample(
        #     up_down4_dst1, size=x.size()[2:], mode='bilinear')
        # pred_down4_dst2 = F.upsample(
        #     up_down4_dst2, size=x.size()[2:], mode='bilinear')
        pred_down4_shad = F.upsample(
            up_down4_shad, size=x.size()[2:], mode='bilinear')

        down3_dst1 = self.dst1(down3)
        down3_dst2 = self.dst2(down3)
        down3_shad = down3

        down3_shad = (1 + self.attention3_hl(
            torch.cat((down3_shad, down3_dst2), 1))) * down3_shad
        down3_shad = F.relu(-self.refine3_hl(
            torch.cat((down3_shad, down3_dst1), 1)) + down3_shad, True)

        # down3_dst1_2 = F.upsample(
        #     down3_dst1, size=down2.size()[2:], mode='bilinear')
        # down3_dst1_1 = F.upsample(
        #     down3_dst1, size=down1.size()[2:], mode='bilinear')
        # down3_dst1_0 = F.upsample(
        #     down3_dst1, size=down0.size()[2:], mode='bilinear')
        # down3_dst2_2 = F.upsample(
        #     down3_dst2, size=down2.size()[2:], mode='bilinear')
        # down3_dst2_1 = F.upsample(
        #     down3_dst2, size=down1.size()[2:], mode='bilinear')
        # down3_dst2_0 = F.upsample(
        #     down3_dst2, size=down0.size()[2:], mode='bilinear')
        down3_shad_2 = F.upsample(
            down3_shad, size=down2.size()[2:], mode='bilinear')
        down3_shad_1 = F.upsample(
            down3_shad, size=down1.size()[2:], mode='bilinear')
        down3_shad_0 = F.upsample(
            down3_shad, size=down0.size()[2:], mode='bilinear')
        # up_down3_dst1 = self.conv1x1_ReLU_down3(
        #     torch.cat((down3_dst1, down4_dst1_3), 1))
        # up_down3_dst2 = self.conv1x1_ReLU_down3(
        #     torch.cat((down3_dst2, down4_dst2_3), 1))
        up_down3_shad = self.conv1x1_ReLU_down3(
            torch.cat((down3_shad, down4_shad_3), 1))
        # pred_down3_dst1 = F.upsample(
        #     up_down3_dst1, size=x.size()[2:], mode='bilinear')
        # pred_down3_dst2 = F.upsample(
        #     up_down3_dst2, size=x.size()[2:], mode='bilinear')
        pred_down3_shad = F.upsample(
            up_down3_shad, size=x.size()[2:], mode='bilinear')

        down2_dst1 = self.dst1(down2)
        down2_dst2 = self.dst2(down2)
        down2_shad = down2
        down2_shad = (1 + self.attention2_hl(
            torch.cat((down2_shad, down2_dst2), 1))) * down2_shad
        down2_shad = F.relu(-self.refine2_hl(
            torch.cat((down2_shad, down2_dst1), 1)) + down2_shad, True)

        # down2_dst1_1 = F.upsample(
        #     down2_dst1, size=down1.size()[2:], mode='bilinear')
        # down2_dst1_0 = F.upsample(
        #     down2_dst1, size=down0.size()[2:], mode='bilinear')
        # down2_dst2_1 = F.upsample(
        #     down2_dst2, size=down1.size()[2:], mode='bilinear')
        # down2_dst2_0 = F.upsample(
        #     down2_dst2, size=down0.size()[2:], mode='bilinear')
        down2_shad_1 = F.upsample(
            down2_shad, size=down1.size()[2:], mode='bilinear')
        down2_shad_0 = F.upsample(
            down2_shad, size=down0.size()[2:], mode='bilinear')
        # up_down2_dst1 = self.conv1x1_ReLU_down2(
        #     torch.cat((down2_dst1, down3_dst1_2, down4_dst1_2), 1))
        # up_down2_dst2 = self.conv1x1_ReLU_down2(
        #     torch.cat((down2_dst2, down3_dst2_2, down4_dst2_2), 1))
        up_down2_shad = self.conv1x1_ReLU_down2(
            torch.cat((down2_shad, down3_shad_2, down4_shad_2), 1))
        # pred_down2_dst1 = F.upsample(
        #     up_down2_dst1, size=x.size()[2:], mode='bilinear')
        # pred_down2_dst2 = F.upsample(
        #     up_down2_dst2, size=x.size()[2:], mode='bilinear')
        pred_down2_shad = F.upsample(
            up_down2_shad, size=x.size()[2:], mode='bilinear')

        down1_dst1 = self.dst1(down1)
        down1_dst2 = self.dst2(down1)
        down1_shad = down1

        down1_shad = (1 + self.attention1_hl(
            torch.cat((down1_shad, down1_dst2), 1))) * down1_shad
        down1_shad = F.relu(-self.refine1_hl(
            torch.cat((down1_shad, down1_dst1), 1)) + down1_shad, True)

        # down1_dst1_0 = F.upsample(
        #     down1_dst1, size=down0.size()[2:], mode='bilinear')
        # down1_dst2_0 = F.upsample(
        #     down1_dst2, size=down0.size()[2:], mode='bilinear')
        down1_shad_0 = F.upsample(
            down1_shad, size=down0.size()[2:], mode='bilinear')
        # up_down1_dst1 = self.conv1x1_ReLU_down1(
        #     torch.cat((
        #         down1_dst1, down2_dst1_1, down3_dst1_1, down4_dst1_1), 1))
        # up_down1_dst2 = self.conv1x1_ReLU_down1(
        #     torch.cat((
        #         down1_dst2, down2_dst2_1, down3_dst2_1, down4_dst2_1), 1))
        up_down1_shad = self.conv1x1_ReLU_down1(
            torch.cat((
                down1_shad, down2_shad_1, down3_shad_1, down4_shad_1), 1))
        # pred_down1_dst1 = F.upsample(
        #     up_down1_dst1, size=x.size()[2:], mode='bilinear')
        # pred_down1_dst2 = F.upsample(
        #     up_down1_dst2, size=x.size()[2:], mode='bilinear')
        pred_down1_shad = F.upsample(
            up_down1_shad, size=x.size()[2:], mode='bilinear')

        down0_dst1 = self.dst1(down0)
        down0_dst2 = self.dst2(down0)
        down0_shad = down0

        down0_shad = (1 + self.attention0_hl(
            torch.cat((down0_shad, down0_dst2), 1))) * down0_shad
        down0_shad = F.relu(-self.refine0_hl(
            torch.cat((down0_shad, down0_dst1), 1)) + down0_shad, True)

        # up_down0_dst1 = self.conv1x1_ReLU_down0(torch.cat(
        #     (down0_dst1, down1_dst1_0, down2_dst1_0, down3_dst1_0,
        #      down4_dst1_0), 1))
        # up_down0_dst2 = self.conv1x1_ReLU_down0(torch.cat(
        #     (down0_dst2, down1_dst2_0, down2_dst2_0, down3_dst2_0,
        #      down4_dst2_0), 1))
        up_down0_shad = self.conv1x1_ReLU_down0(torch.cat(
            (down0_shad, down1_shad_0, down2_shad_0, down3_shad_0,
             down4_shad_0), 1))
        # pred_down0_dst1 = F.upsample(
        #     up_down0_dst1, size=x.size()[2:], mode='bilinear')
        # pred_down0_dst2 = F.upsample(
        #     up_down0_dst2, size=x.size()[2:], mode='bilinear')
        pred_down0_shad = F.upsample(
            up_down0_shad, size=x.size()[2:], mode='bilinear')

        fuse_pred_shad = self.fuse_predict(torch.cat(
            (pred_down0_shad, pred_down1_shad, pred_down2_shad,
             pred_down3_shad, pred_down4_shad), 1))
        # fuse_pred_dst1 = self.fuse_predict(torch.cat(
        #     (pred_down0_dst1, pred_down1_dst1, pred_down2_dst1,
        #      pred_down3_dst1, pred_down4_dst1), 1))
        # fuse_pred_dst2 = self.fuse_predict(torch.cat(
        #     (pred_down0_dst2, pred_down1_dst2, pred_down2_dst2,
        #      pred_down3_dst2, pred_down4_dst2), 1))

        # if self.training:
        #     return fuse_pred_shad, pred_down1_shad, pred_down2_shad,
        #     pred_down3_shad, pred_down4_shad, fuse_pred_dst1,
        #     pred_down1_dst1, pred_down2_dst1, pred_down3_dst1,
        #     pred_down4_dst1, fuse_pred_dst2, pred_down1_dst2,
        #     pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
        #     pred_down0_dst1, pred_down0_dst2, pred_down0_shad
        # else:
        #     return torch.sigmoid(fuse_pred_shad)

        if self.training:
            return {
                'attn': fuse_pred_shad, 'pred_down0': pred_down0_shad,
                'pred_down1': pred_down1_shad, 'pred_down2': pred_down2_shad,
                'pred_down3': pred_down3_shad, 'pred_down4': pred_down4_shad,
            }
        else:
            if self.opt.det_loss_type in ['l1', 'wl1', 'l2']:
                return {'attn': torch.clamp(fuse_pred_shad, min=0.0, max=1.0)}
            else:
                return {'attn': torch.sigmoid(fuse_pred_shad)}
