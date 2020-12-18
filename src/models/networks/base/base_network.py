import torch.nn as nn
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. \
             Total number of parameters: %.1f million. \
             To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and \
                (classname.find('Conv') != -1 or \
                    classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'identity':
                    init.eye_(m.weight.data)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        '[%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, padding=1, dilation=1,
                 conv_type='conv', norm_layer=None, act_type='lrelu'):
        super().__init__()

        if conv_type == 'conv':
            conv_layer = \
                nn.Conv2d(in_ch, out_ch, ksize, stride, padding, dilation)
        elif conv_type == 'deconv':
            conv_layer = nn.ConvTranspose2d(
                in_ch, out_ch, ksize, stride, padding)
        else:
            raise NotImplementedError

        if norm_layer is not None:
            sequence = [norm_layer(conv_layer)]
        else:
            sequence = [conv_layer]

        if act_type == 'lrelu':
            sequence.append(nn.LeakyReLU(0.2))
        elif act_type == 'relu':
            sequence.append(nn.ReLU())
        elif act_type == 'sigmoid':
            sequence.append(nn.Sigmoid())
        elif act_type == 'tanh':
            sequence.append(nn.Tanh())
        elif act_type == 'none':
            pass
        else:
            raise NotImplementedError

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
