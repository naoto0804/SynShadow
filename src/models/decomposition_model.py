import torch
import torch.nn as nn
from models.base_model import BaseModel


class DecompositionModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--spm_alpha', type=float, default=10.0)
        parser.set_defaults(
            netG='spm', norm_G='batch', no_adv=True)
        if is_train:
            parser.set_defaults(
                load_size=320, crop_size=256, nepoch=500, display_freq=100000,
                save_epoch_freq=500, save_latest_freq=50000, nepoch_decay=1500,
                fit_affine='global', batch_size=32, mask_to_G='gt')
        else:
            parser.set_defaults(
                load_size=256, batch_size=1, precomp_mask_thresh=0.95)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.spm_alpha = opt.spm_alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, data, mode):
        data = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, output = self.compute_generator_loss(data)
            return g_loss, output
        elif mode == 'inference':
            with torch.no_grad():
                output = self.netG(data['input'], data['precomp_mask'])
            return output
        else:
            raise ValueError("|mode| is invalid")

    def compute_generator_loss(self, data):
        G_losses = {}
        target = data['target']

        output_dict = self.netG(data['input'], data['precomp_mask'])
        G_losses['w_l2'] = self.l2(output_dict['w'], data['affine_w'])
        G_losses['b_l2'] = self.l2(output_dict['b'], data['affine_b'])
        G_losses['L1'] = self.spm_alpha * self.l1(output_dict['img'], target)
        return G_losses, output_dict

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=0.0002)
        return optimizer_G, None

    def add_visualization(self, data: dict, gen: dict, visual_list: list):
        dic = {
            'gen_img': gen['img'], 'gen_matte': gen['matte'],
            'gen_lit': gen['lit'], 'precomp_mask': data['precomp_mask'],
        }
        for key, content in dic.items():
            visual_list.append((key, content))
        return visual_list
