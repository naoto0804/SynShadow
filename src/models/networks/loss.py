import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.networks.base.architecture import VGG19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        assert gan_mode in ['original', 'w', 'ls', 'hinge', 'dhan']

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'dhan':
            EPS = 1e-12
            prob = torch.sigmoid(input)
            if for_discriminator:
                if target_is_real:
                    loss = -torch.log(prob + EPS).mean()
                else:
                    loss = -torch.log(1 - prob + EPS).mean()
            else:
                loss = -torch.log(prob + EPS).mean()
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, \
                    "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(
                    pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    assert a == 1
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class RefineLoss(nn.Module):
    def __init__(self):
        super(RefineLoss, self).__init__()
        self.alpha = 6.0
        self.lambdas = [0.6, 0.8, 1.0]
        self.hyper_params = {
            'multiscale_regression': 1.0,
            'content': 0.5,
            'texture': 50.0,
            'total_variation': 25.0
        }
        self.l1_loss = nn.L1Loss(reduce=False).cuda()
        # pool1, pool2, pool3
        self.downsampler = nn.AvgPool2d(3, 2, 1).cuda()

        self.vgg_indices = [5, 10, 17]
        self.vgg = VGG16Extractor(
            indices=self.vgg_indices, requires_grad=True).cuda()
        self.vgg_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.vgg_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    def forward(self, input, outputs, target, mask):
        assert len(outputs) == len(self.lambdas)
        losses = {
            'multiscale_regression': 0.0,
            'content': 0.0,
            'texture': 0.0,
            'total_variation': 0.0
        }

        tmp_input = input.clone()
        tmp_target = target.clone()
        tmp_mask = mask.clone()

        # multiscale regresion loss (Eq. 2)
        for i in reversed(range(len(outputs))):
            loss = self.lambdas[i] * self.l1_loss(outputs[i], tmp_target)
            loss = tmp_mask * loss + self.alpha * (1 - tmp_mask) * loss
            losses['multiscale_regression'] += loss.mean()

            tmp_input = self.downsampler(tmp_input)
            tmp_target = self.downsampler(tmp_target)
            tmp_mask = self.downsampler(tmp_mask)

        comp = mask * outputs[-1] + (1.0 - mask) * input

        def normalize_for_vgg(tensor, mean=self.vgg_mean, std=self.vgg_std):
            assert tensor.max().item() <= 1.0 and tensor.min().item() >= -1.0
            a, b, c, d = tensor.size()
            zero_one_tensor = (tensor + 1.0) / 2.0  # [0.0, 1.0]
            return (zero_one_tensor - mean.view(b, 1, 1)) / std.view(b, 1, 1)

        act_comp = self.vgg(normalize_for_vgg(comp))
        act_output = self.vgg(normalize_for_vgg(outputs[-1]))
        with torch.no_grad():
            act_target = self.vgg(normalize_for_vgg(target))

        for i in range(len(self.vgg_indices)):
            # content loss (Eq. 3)
            losses['content'] += torch.mean(
                torch.abs(act_comp[i] - act_target[i]))
            losses['content'] += torch.mean(
                torch.abs(act_output[i] - act_target[i]))

            # texture loss (Eq. 4 and Eq. 5)
            gram_act_comp_i = gram_matrix(act_comp[i])
            gram_act_output_i = gram_matrix(act_output[i])
            gram_act_target_i = gram_matrix(act_target[i])
            losses['texture'] += torch.mean(
                torch.abs(gram_act_output_i - gram_act_target_i))
            losses['texture'] += torch.mean(
                torch.abs(gram_act_comp_i - gram_act_target_i))

        # texture loss (Eq. 6)
        output = outputs[-1]
        losses['total_variation'] += torch.mean(
            torch.abs(output[..., :-1] - output[..., 1:]))
        losses['total_variation'] += torch.mean(
            torch.abs(output[..., :-1, :] - output[..., 1:, :]))

        # multiply hyper-parameters (Eq. 7)
        for k in losses.keys():
            losses[k] = losses[k] * self.hyper_params[k]

        return losses


class VGG16Extractor(torch.nn.Module):
    def __init__(self, indices=[], requires_grad=False):
        super().__init__()
        vgg_pretrained_features = \
            torchvision.models.vgg16(pretrained=True).features

        self.indices = indices
        assert len(indices) > 0

        start_ind = 0
        for i in range(len(indices)):
            setattr(self, f'slice{i+1}', torch.nn.Sequential())
            for x in range(start_ind, indices[i]):
                slice_ = getattr(self, f'slice{i+1}')
                slice_.add_module(str(x), vgg_pretrained_features[x])
            start_ind = indices[i]

        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        outputs = [input]
        for i in range(len(self.indices)):
            slice_ = getattr(self, f'slice{i+1}')
            outputs.append(slice_(outputs[-1]))
        return outputs[1:]
