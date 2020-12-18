import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import util.util as util
from data.image_folder import make_dataset
from util.illum_affine_model import fit_brightening_params


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # Don't use `parser.set_defaults` here.
        return parser

    def get_precomp_mask_paths(self, opt):
        dir_ = self.root / self.opt.mask_to_G
        precomp_mask_paths = make_dataset(dir_, read_cache=True)
        return precomp_mask_paths

    def initialize(self, opt):
        self.opt = opt
        name = 'train' if opt.isTrain else 'test'
        self.root = Path(opt.dataset_root) / name
        paths_dict = self.get_paths(opt)
        for k, v in paths_dict.items():
            util.natural_sort(v)
        for k, v in paths_dict.items():
            setattr(self, f"{k}_paths", v)
        self.dataset_size = len(v)  # note that the last element is used.

        # commonly used since the evaluation is done in this size
        self.basic_transform = transforms.Compose(
            [transforms.Resize(size=(480, 640)), transforms.ToTensor()])

    # Give subclasses a chance to modify the final output
    def postprocess(self, input_dict):
        d = input_dict
        d['affine_w'], d['affine_b'] = fit_brightening_params(
                d['input'], d['target'], d['mask'].squeeze())
        return d


def get_random_size(size, v_min=256, v_max=480):
    # TODO: v_max should be 480 but failed due to OOM
    w, h = size
    max_len = random.randint(v_min, v_max)
    if w >= h:
        new_w = max_len
        new_h = int((max_len / w) * h)
    else:
        new_h = max_len
        new_w = int((max_len / h) * w)
    return (new_w, new_h)


def get_params(opt, size):
    w, h = size
    new_w, new_h = w, h
    if opt.preprocess_mode == 'fixed_resize':
        # new_w, new_h = 640, 480
        # new_w, new_h = 480, 360
        new_w, new_h = 256, 256
    elif opt.preprocess_mode == 'random_resize':
        new_w, new_h = get_random_size(size)
    elif opt.preprocess_mode == 'resize_and_crop':
        new_w = new_h = opt.load_size
    elif opt.preprocess_mode == 'scale_width':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip, 'new_load_size': (new_w, new_h)}


def get_transform(opt, params, n_ch=3):
    assert opt.input_nc == opt.output_nc == 3
    if n_ch == 3:
        method = Image.BICUBIC
    elif n_ch == 1:
        method = Image.NEAREST
    else:
        raise NotImplementedError

    transform_list = []
    if 'resize' in opt.preprocess_mode:
        # input of Resize is in (h, w) form.
        transform_list.append(
            transforms.Resize(params['new_load_size'][::-1],
                              interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))

    # if opt.preprocess_mode == 'fixed':
    #     w = opt.crop_size
    #     h = round(opt.crop_size / opt.aspect_ratio)
    #     transform_list.append(transforms.Lambda(
    #         lambda img: __resize(img, w, h, method)))

    # if opt.isTrain and rgb_aug is True:
    #     transform_list.append(transforms.ColorJitter(
    #         brightness=(0.5, 1.5), contrast=(0.5, 1.5),
    #         saturation=(0.5, 1.5), hue=0.05))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
