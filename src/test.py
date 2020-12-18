from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import data
from models import create_model
from options.test_options import TestOptions
from util import webhtml
from util.visualizer import Visualizer

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)
visualizer = Visualizer(opt)

model = create_model(opt)
model.eval()

# create a webpage that summarizes the all results
web_dir = Path(
    opt.results_dir) / opt.name / ('%s_%s' % (opt.phase, opt.which_epoch))
webpage = webhtml.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(tqdm.tqdm(dataloader)):
    generated = model(data_i, mode='inference')
    img_path = data_i['path']

    if 'img' in generated.keys():
        img_outputs = generated['img']
        if isinstance(img_outputs, list):
            img_outputs = img_outputs[-1]

    if 'attn' in generated.keys():
        attn_outputs = generated['attn']
        if isinstance(attn_outputs, list):
            attn_outputs = attn_outputs[-1]
        attn_outputs = torch.clamp(attn_outputs, min=0.0, max=1.0)

    # make sure to save only the last (t=N) outputs for generated tensors
    for b in range(data_i['input'].size(0)):
        visuals_list = []
        keys = generated.keys()
        if 'target' in data_i.keys():
            visuals_list.append(('target', data_i['target'][b]))
        if 'img' in keys:
            visuals_list.append(('img_output', img_outputs[b]))
        visuals_list.append(('input', data_i['input'][b]))
        if 'precomp_mask' in data_i.keys():
            visuals_list.append(('precomp_mask', data_i['precomp_mask'][b]))
        if 'lit' in keys:
            visuals_list.append(('lit_output', generated['lit'][b]))
        if 'matte' in keys:
            visuals_list.append(('matte_output', generated['matte'][b]))
        if 'attn' in keys:
            visuals_list.append(('attn_output', attn_outputs[b]))
        if 'mask' in data_i.keys():
            visuals_list.append(('mask', data_i['mask'][b]))

        if 'img_w640_h480' in data_i.keys():
            # For SPMNet, reconver image in original resolution
            rgb_full = data_i['img_w640_h480'][b:b+1]  # assume batchsize = 1
            B, C, H, W = rgb_full.size()

            if opt.model == 'local_decomposition':
                generated['w'] = F.interpolate(generated['w'], size=(H, W), mode='bilinear')
                generated['b'] = F.interpolate(generated['b'], size=(H, W), mode='bilinear')

            lit_full = \
                model.netG.relit(rgb_full, generated['w'], generated['b'])
            matte = generated['matte'][b:b+1]
            matte_full = F.interpolate(matte, size=(H, W), mode='bilinear')
            result = model.netG.compose(rgb_full, lit_full, matte_full)
            visuals_list.append(('img_output_w640_h480', result.squeeze()))

        visuals = OrderedDict(visuals_list)
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
