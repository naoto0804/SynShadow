# Overview
This subfolder is for putting various datasets.

You can get
- [Matte images](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/SynShadow.zip) for SynShadow

**Note that these datasets are meant for education and research purposes only.**

## File Organization

### ISTD+/SRD+
```
- <dataset_name>
    - train
        - input
        - mask
        ( - target)  # only required for removal
    - test
        - input
        - mask
        ( - target)  # only required for removal
        ( - precomp_mask)  # precomputed shadow detection results, only required for SP+M in removal models
```

### USR
```
- USR
    - shadow_free
    - shadow_train
    - shadow_test
```

### SynShadow
Synthetic generation is generated during training using [this](https://github.com/naoto0804/SynShadow/blob/main/src/util/illum_affine_model.py#L141-L155).
```
- SynShadow
    - matte
    - shadow_free  # please copy or use symbolic link to get USR/shadow_free as shadow-free background images for shadow composition
```

## Downloads

Here are the pointers for some datasets used in this project.
- Shadow removal
    - [USR](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)
        - [Paper(ICCV2019)](https://arxiv.org/abs/1903.10683)
        - [Project](https://github.com/xw-hu/Mask-ShadowGAN)
    - [ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)
        - [Paper(CVPR2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Stacked_Conditional_Generative_CVPR_2018_paper.pdf)
        - [Project](https://github.com/DeepInsight-PCALab/ST-CGAN)
        - [SP+M(ICCV2019)](https://arxiv.org/abs/1908.08628) suggests to fix a color inconsistency issue. Code for fix is provided [here](https://drive.google.com/open?id=1aGS3fisgXASEqyVvMpwAJCHP__U-dknW)
    - SRD
        - [Paper(CVPR2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf)
        - Please contact [the authors](http://vision.sia.cn/our%20team/JiandongTian/JiandongTian.html) for getting the datasets.
        - [DHAN(AAAI2020)](https://arxiv.org/abs/1911.08718) is providing masks extracted from shadow and shadow-free image pairs at [here](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EZ8CiIhNADlAkA4Fhim_QzgBfDeI7qdUrt6wv2EVxZSc2w?e=hZ0ruG)
        - We found that the original train-test split of SRD is inappropriate since images coming from the identical background are both in the training and testing set. We re-split SRD so that there is no overlap of scenes in the two sets and removed near-duplicate images. The new split is called SRD+, and we provide `SRD+_train.txt` and `SRD+_test.txt` for list of the images for SRD+. 
    - [GTAV](https://drive.google.com/file/d/1ktOXJmMQL_6U2J03mks3yWh6EMWKjUmu/view)
        - [Paper(CVPRW2019)](https://arxiv.org/abs/1811.06604)
        - [Project](https://github.com/acecreamu/angularGAN)
- Shadow detection
    - [SBU](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip)
        - [Paper(ECCV2016)](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_49)
        - [Project](https://www3.cs.stonybrook.edu/~minhhoai/projects/shadow.html)
