import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import tqdm


def find_image(path: Path):
    """
    Find image with the specified name, but can change suffix
    """
    if path.exists():
        return path
    else:
        for ext in ['.png', '.jpeg', '.jpg', '.exr', '.tga', '.gif', '.pfm']:
            new_path = path.with_suffix(ext)
            if new_path.exists():
                return new_path
        raise FileNotFoundError


def load(path: Path, size):
    img_pil = Image.open(find_image(path))
    img_pil = img_pil.resize(size, Image.BILINEAR)
    img_npy = np.array(img_pil) / 255.0
    return img_npy
