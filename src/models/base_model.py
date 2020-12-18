import torch
import models.networks as networks
import util.util as util


class BaseModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.

    def forward(self, data, mode):
        data = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, output = self.compute_generator_loss(data)
            return g_loss, output
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.mask_to_G is None:
                    input = data['input']
                else:
                    input = torch.cat(
                        [data['input'], data['precomp_mask']], dim=1)
                output = self.netG(input)
            return output
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(
            list(self.netG.parameters()), lr=G_lr, betas=(beta1, beta2))

        if opt.isTrain and not opt.no_adv:
            optimizer_D = torch.optim.Adam(
                list(self.netD.parameters()), lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if self.netD is not None:
            util.save_network(self.netD, 'D', epoch, self.opt)

    ###########################################################################
    # Private helper methods
    ###########################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)

        resume = False
        if hasattr(opt, 'continue_train'):
            if opt.continue_train and opt.isTrain:
                resume = True

        if not opt.isTrain or resume:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)

        if opt.isTrain and opt.finetune_from is not None:
            print(f"Loading {opt.finetune_from} to netG..")
            netG.load_state_dict(torch.load(opt.finetune_from))

        if opt.isTrain and not opt.no_adv:
            netD = networks.define_D(opt)
        else:
            netD = None
        if resume:
            netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        if self.use_gpu():
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
        return data

    def preprocess_for_G_D(self, data):
        if self.opt.mask_to_G is None:
            input_G = data['input']
            input_D = data['input']
        else:
            precomp_mask = data['precomp_mask']
            input_G = torch.cat([data['input'], precomp_mask], dim=1)
            if self.opt.use_precomp_mask_to_D:
                input_D = input_G
            else:
                input_D = data['input']
        return input_G, input_D

    def compute_generator_loss(self):
        raise NotImplementedError

    def compute_discriminator_loss(self):
        raise NotImplementedError

    def discriminate(self):
        raise NotImplementedError

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
