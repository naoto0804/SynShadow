import argparse
import pickle
from pathlib import Path

import data
import models
import models.networks as networks
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='debug',
                            help='identifier of the experiment')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='../checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='decomposition',
                            choices=['detect', 'decomposition'])
        parser.add_argument('--norm_G', type=str, default='batch',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='batch',
                            help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str,
                            default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batch_size', type=int,
                            default=4, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='resize_and_crop', help='scaling and cropping of images at load time.', choices=(
            "resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=256,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataset_root', type=str, default='../datasets')
        parser.add_argument('--dataset_mode', type=str, default='paired')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--num_threads', default=8,
                            type=int, help='# threads for loading data')

        # for displays
        parser.add_argument('--display_winsize', type=int,
                            default=256, help='display window size')

        # for generator
        parser.add_argument(
            '--netG', type=str, default='spm', choices=['bdrar', 'dsd', 'spm'])
        parser.add_argument('--ngf', type=int, default=64,
                            help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')

        # for our model
        parser.add_argument('--mask_to_G', default=None,
                            help="Type of pre-computed mask feeded to G")
        parser.add_argument('--mask_to_G_thresh', type=float, default=0.0,
                            help="if <= 0.0, do nothing")
        # parser.add_argument('--use_precomp_mask_to_D', action='store_true')
        parser.add_argument('--fit_affine', type=str, default=None)
        parser.add_argument('--mask_preprocess_mode', type=str,
                            default="scale_width")

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        # modify networks-related parser options
        parser = networks.modify_commandline_options(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        if makedir:
            expr_dir.mkdir(parents=True, exist_ok=True)
        file_name = expr_dir / 'opt.txt'
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with file_name.open(mode='wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(
                    str(k), str(v), comment))

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
