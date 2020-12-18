from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from data.augmentation import MaskAugmentor


class PairedDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def initialize(self, opt):
        super().initialize(opt)
        if self.opt.mask_to_G == 'augmented_gt':
            self.augmentor = MaskAugmentor()

    def get_paths(self, opt):
        """
            Make sure the organization is as follows;
            - train
                - input
                - target
                - mask
                (- precomp_mask_X)
            (- test)
        """

        input_paths = make_dataset(self.root / 'input', read_cache=True)
        target_paths = make_dataset(
            self.root / 'target', read_cache=True)
        mask_paths = make_dataset(self.root / 'mask', read_cache=True)
        paths_dict = \
            {'input': input_paths, 'target': target_paths, 'mask': mask_paths}
        assert len(target_paths) == len(input_paths) == len(mask_paths)

        # Use ground truth mask in training, use predicted mask in testing
        if self.opt.mask_to_G is not None and not opt.isTrain:
            paths_dict["precomp_mask"] = self.get_precomp_mask_paths(opt)

        return paths_dict

    def __getitem__(self, index):
        input_pil = Image.open(self.input_paths[index]).convert('RGB')
        target_pil = Image.open(self.target_paths[index]).convert('RGB')
        mask_pil = Image.open(self.mask_paths[index]).convert('L')

        params = get_params(self.opt, input_pil.size)
        transform_image = get_transform(self.opt, params)
        transform_mask = get_transform(self.opt, params, n_ch=1)

        input = transform_image(input_pil)
        target = transform_image(target_pil)
        mask = transform_mask(mask_pil)

        input_dict = {
            'input': input, 'target': target,
            'mask': (mask > 0.5).float(),  # 0.0 nor 1.0 due to aug.
            'path': self.input_paths[index],
        }

        if self.opt.mask_to_G is not None:
            if self.opt.mask_to_G == 'gt':
                precomp_mask = mask
            elif self.opt.mask_to_G == 'augmented_gt':
                precomp_mask = self.augmentor((mask > 0.5))
            else:
                precomp_mask = Image.open(
                    self.precomp_mask_paths[index]).convert('L')
                precomp_mask = transform_mask(precomp_mask)

            if self.opt.mask_to_G_thresh > 0.0:
                input_dict['precomp_mask'] = \
                    (precomp_mask > self.opt.mask_to_G_thresh).float()
            else:
                input_dict['precomp_mask'] = precomp_mask.float()

        # For evaluation
        if not self.opt.isTrain and self.opt.netG in ['spm', 'local_spm']:
            input_dict['img_w640_h480'] = \
                self.basic_transform(input_pil)

        return self.postprocess(input_dict)

    def __len__(self):
        return self.dataset_size
