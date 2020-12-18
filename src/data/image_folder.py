import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir_, images):
    assert dir_.is_dir(), '%s is not a valid directory' % dir_

    for root, dnames, fnames in sorted(os.walk(dir_, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = '%s/%s' % (root, fname)
                images.append(path)


def make_dataset(dir_, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = dir_ / 'files.list'
        if possible_filelist.is_file():
            with possible_filelist.open(mode='r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir_, images)
    else:
        assert dir_.is_dir(), '%s is not a valid directory' % dir_
        images = sorted(str(f) for f in dir_.glob('*') if f.is_file())

    if write_cache:
        filelist_cache = dir_ / 'files.list'
        with filelist_cache.open(mode='w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
