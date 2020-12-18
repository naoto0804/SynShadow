# from DSDNet code
# https://drive.google.com/open?id=18hv6NAQsST1UsabtNvfM_G-qR-nAbgaa)

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import tqdm
import pydensecrf.densecrf as dcrf

from image import find_image


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img: np.ndarray, annos: np.ndarray):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / \
        (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


parser = argparse.ArgumentParser()
parser.add_argument('guide_dir', type=str)
parser.add_argument('pred_input_dir', type=str)
parser.add_argument('pred_output_dir', type=str)
parser.add_argument('--output_ext', type=str, default='.png')
parser.add_argument('--H', type=int, default=256)
parser.add_argument('--W', type=int, default=256)

args = parser.parse_args()

guide_dir = Path(args.guide_dir)
pred_input_dir = Path(args.pred_input_dir)
pred_output_dir = Path(args.pred_output_dir)
pred_output_dir.mkdir(exist_ok=True, parents=True)

for guide_f in tqdm.tqdm(list(guide_dir.glob('*'))):
    guide_pil = Image.open(guide_f)
    pred_input_pil = Image.open(find_image(pred_input_dir / guide_f.name))
    W, H = guide_pil.size

    guide_npy = np.array(guide_pil)
    pred_output_npy = np.array(pred_input_pil.convert('L').resize((W, H)))
    pred_output_npy = crf_refine(guide_npy, pred_output_npy)
    pred_output_pil = Image.fromarray(pred_output_npy)
    pred_output_pil = pred_output_pil.resize((args.W, args.H))
    pred_output_pil.save(
        str((pred_output_dir / guide_f.name).with_suffix(args.output_ext)))
