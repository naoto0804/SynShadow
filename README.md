# Learning from Synthetic Shadows for Shadow Detection and Removal (IEEE TCSVT 2021)

## Overview
This repo is for the paper "[Learning from Synthetic Shadows for Shadow Detection and Removal](https://arxiv.org/abs/2101.01713)". We present SynShadow, a novel large-scale synthetic shadow/shadow-free/matte image triplets dataset and pipeline to synthesize it. We further show how to use SynShadow for robust and efficient shadow detection and removal.

![](teaser.png)

In this repo, we provide
- SynShadow dataset: `./datasets`
- [SP+M](https://arxiv.org/abs/1908.08628) implementation: `./src`
- Trained models and results: below

If you find this code or dataset useful for your research, please cite our paper:

```
@article{inoue2021learning,
  title={{Learning from Synthetic Shadows for Shadow Detection and Removal}},
  author={Inoue, Naoto and Yamasaki, Toshihiko},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2021},
  volume={31},
  number={11},
  pages={4187-4197},
  doi={10.1109/TCSVT.2020.3047977}
}
```


## Trained Models and Results
We provide the models for shadow detection and removal for convenience. Downloaded models should be placed under `./checkpoints`.

### Shadow Detection
ALl the results are in 480x640. BER is reported for 480x640 images. Below are results evaluated on ISTD test set. DSDNet++ is a modified variant of [DSDNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Distraction-Aware_Shadow_Detection_CVPR_2019_paper.html).

| Model | Train | BER |      |
|  :-:  |   :-:    | :-: | :-:  |
|DSDNet++|SynShadow|2.74|[results](https://drive.google.com/file/d/1jhVXHSHj4teacLTK_I-Ep4WNDxId-_tp/view?usp=drive_link) / [weights](https://drive.google.com/file/d/13IhP8d5Tb7ZM3kgu0HmgAvSxAS0KMtXZ/view?usp=drive_link)|
|DSDNet++|SynShadow->ISTD|1.09|[results](https://drive.google.com/file/d/14Ni8fEoQJNj1xz2e_JD1zUKSWwuuwcxc/view?usp=drive_link) / [weights](https://drive.google.com/file/d/1CQ6KjDlKesdb86tJnVcX79B3br_w5jHt/view?usp=drive_link)|
|BDRAR|SynShadow|2.74|[results](https://drive.google.com/file/d/1IHTf0901Mzm5a2G1sROfAyJYoddFNVbF/view?usp=drive_link) / [weights](https://drive.google.com/file/d/1y-VQBoclNzPsfBZovMwpymlVb4I3ljVy/view?usp=drive_link)|
|BDRAR|SynShadow->ISTD|1.10|[results](https://drive.google.com/file/d/1Pr7iAxPsjdZMwwEv7aZ54eiw7GMVoEsS/view?usp=drive_link) / [weights](https://drive.google.com/file/d/1pLZr8uRfnPVkUzI3f7Yx0smcNN2W0l0N/view?usp=drive_link)|

### Shadow Removal
ALl the results are in 480x640. For the pre-trained weights, we only provide SP+M weights, since this repository has full implementation of it. RMSE is reported for 480x640 images.

Model: [SP+M](https://arxiv.org/abs/1908.08628)
| Train | Test | RMSE |      |
| :-: | :-: | :-: | :-: |
|SynShadow|ISTD+|4.9|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_spm_istd+_train_synshadow.zip) / [weights](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/trained_models/removal/rem_spm_synshadow.zip) / [precomputed_mask](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/precomputed_masks/precomp_mask_test_istd+_train_synshadow.zip)|
|SynShadow->ISTD+|ISTD+|4.0|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_spm_istd+_finetune_from_synshadow.zip) / [weights](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/trained_models/removal/rem_spm_istd+_finetune_from_synshadow.zip) / [precomputed_mask](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/precomputed_masks/precomp_mask_test_istd+_train_istd+_finetune_from_synshadow.zip)|
|SynShadow|SRD+|5.7|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_spm_srd+_train_synshadow.zip) / [weights](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/trained_models/removal/rem_spm_synshadow.zip) / [precomputed_mask](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/precomputed_masks/precomp_mask_test_srd+_train_synshadow.zip)|
|SynShadow->SRD+|SRD+|5.2|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_spm_srd+_finetune_from_synshadow.zip) / [weights](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/trained_models/removal/rem_spm_srd+_finetune_from_synshadow.zip) / [precomputed_mask](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/precomputed_masks/precomp_mask_test_srd+_train_srd+_finetune_from_synshadow.zip)|
|SynShadow|USR|-|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_spm_usr_train_synshadow.zip) / [weights](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/trained_models/removal/rem_spm_synshadow.zip) / [precomputed_mask](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/precomputed_masks/precomp_mask_test_usr_train_synshadow.zip)|

Model: [DHAN](https://arxiv.org/abs/1911.08718)
| Train | Test | RMSE |      |
| :-: | :-: | :-: | :-: |
|SynShadow->ISTD+|ISTD+|4.6|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_dhan_istd+_finetune_from_synshadow.zip)|
|SynShadow->SRD+|SRD+|6.6|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_dhan_srd+_finetune_from_synshadow.zip)|
|SynShadow|USR|-|[results](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/results/removal/result_rem_dhan_usr_train_synshadow.zip)|

Note: we have accidentially removed some files and cannot provide some results.
