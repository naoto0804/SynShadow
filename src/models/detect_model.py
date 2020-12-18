import torch
import torch.nn as nn

from models.base_model import BaseModel


class WeightedL1Loss(nn.Module):
    def __init__(self, alpha=1.0):
        """
            Note that input is between 0.0 (negative) and 1.0 (positive)
            If alpha == 0.0, the loss is equal to L1.
            Larger alpha emphasize the importance of positive labels
        """
        super().__init__()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, output, target):
        # The masks are mostly negative, so put more weight on positive masks
        loss = torch.abs(output - target) * (1.0 + self.alpha * target)
        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
        Losses used in DSDNet
        Distraction-aware Shadow Detection (CVPR2019)
        https://quanlzheng.github.io/projects/Distraction-aware-Shadow-Detection.html
    """
    def __init__(self):
        super().__init__()
        self.bc = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        # B, C, H, W = target.size()
        # loss = self.bc(output, target)
        # target = target.int()

        # sample-wise weight
        # pos_w = (target == 0).sum(dim=[1, 2, 3]).float() / (C * H * W)
        # pos_w = pos_w.view(B, 1, 1, 1).repeat(1, C, H, W)
        # neg_w = (target == 1).sum(dim=[1, 2, 3]).float() / (C * H * W)
        # neg_w = neg_w.view(B, 1, 1, 1).repeat(1, C, H, W)
        # w = torch.zeros_like(output)
        # w[target == 0] = neg_w[target == 0]
        # w[target == 1] = pos_w[target == 1]
        # loss = (w * loss).mean()

        # batch-wise weight
        # pos_w = (target == 0).sum().float() / target.numel()
        # neg_w = (target == 1).sum().float() / target.numel()
        # w = torch.zeros_like(output)
        # w[target == 0] = neg_w
        # w[target == 1] = pos_w
        # loss = (w * loss).mean()

        # Following dsdnet
        epsilon = 1e-10
        # sigmoid_pred = torch.sigmoid(output)
        count_pos = torch.sum(target) * 1.0 + epsilon
        count_neg = torch.sum(1.0 - target) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(output, target)

        return loss


class DetectModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--det_loss_type', default='wbc',
            choices=['l1', 'l2', 'bc', 'wbc'])
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.set_defaults(
            netG='dsd', no_adv=True, lr=0.005, lr_decay_policy='poly')
        return parser

    def create_optimizers(self, opt):
        # optimizer_G = torch.optim.SGD(
        #     list(self.netG.parameters()), lr=opt.lr,
        #     momentum=0.9, weight_decay=opt.weight_decay)
        b_params = \
            [p for n, p in self.netG.named_parameters() if n[-4:] == 'bias']
        w_params = \
            [p for n, p in self.netG.named_parameters() if n[-4:] != 'bias']
        optimizer_G = torch.optim.SGD([
            {'params': b_params, 'lr': 2 * opt.lr},
            {'params': w_params, 'lr': opt.lr,
             'weight_decay': opt.weight_decay}
        ], momentum=0.9)
        return optimizer_G, None

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        assert self.opt.netG in ['fsd', 'bdrar', 'dsd']
        if opt.det_loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif opt.det_loss_type == 'l2':
            self.loss_func = nn.MSELoss()
        elif opt.det_loss_type == 'bc':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif opt.det_loss_type in 'wbc':
            self.loss_func = WeightedCrossEntropyLoss()
        else:
            raise NotImplementedError

    def compute_generator_loss(self, data):
        input = data['input']
        mask = data['mask']
        output_dict = self.netG(input)

        if self.opt.netG == 'fsd':
            loss = self.loss_func(output_dict['attn'], mask)
            G_losses = {self.opt.det_loss_type: loss}
        elif self.opt.netG in ['bdrar', 'dsd']:
            G_losses = {}
            for k, v in output_dict.items():
                loss = self.loss_func(v, mask)
                G_losses[f"{k}_{self.opt.det_loss_type}"] = loss
        else:
            raise NotImplementedError
        return G_losses, output_dict

    def add_visualization(self, data: dict, gen: dict, visual_list: list):
        dic = {'gen_attn': gen['attn']}
        for key, content in dic.items():
            visual_list.append((key, content))
        return visual_list
