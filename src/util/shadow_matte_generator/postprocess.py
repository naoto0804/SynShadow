# reformat render (originally in a very strange format)

# each channel contains 4 values (1, 1, 1, 8) bits
# e.g. [0, 0, 0, 127]

import argparse
import imageio
import numpy as np
import tqdm

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default="data/640x640_light_0p1_1p5/raw")
parser.add_argument('--output_dir', type=str,
                    default="data/640x640_light_0p1_1p5/raw")
args = parser.parse_args()

input_dir = Path(args.input_dir)
assert input_dir.exists()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

for f in tqdm.tqdm(list(input_dir.glob('*'))):
    matte = imageio.imread(str(f))[...,  -1]
    if matte.max() == 0:
        continue
    else:
        matte = matte * 255 / (matte.max() - matte.min())
        matte = np.clip(matte, a_min=0, a_max=255).astype(np.uint8)
        imageio.imsave(
            str(output_dir / f.with_suffix('.png').name), matte)
