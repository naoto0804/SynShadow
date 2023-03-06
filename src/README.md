## Preprocess

### Software Requirements
**Please use [Dockerfile](./Dockerfile) (added in Mar. 2021) for easy installation.**

- Python 3.5~3.7
- PyTorch 1.2+
- TorchVision 0.4+
- Tensorflow 1+ (Note: this makes us be unable to use Python 3.8)

For quantitative evaluation of shadow detection and removal, matlab is required. (I used R2016b. I'm not sure whether other versions work.)

### Downloads
Please refer to `../datasets/README.md` and `../README.md` to download datasets and trained models, respectively.

## Detection
For more details about arguments, please refer to -h option or the actual codes.

### Testing
```
$ python test.py --model detect --netG dsd --dataset_root ../datasets/ISTD+ --name <IDENTIFIER>
```

### Quantitative Evaluation
BER will be computed.

0. Install DenseCRF
`densecrf` is additionally required.
```
$ pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

1. Post-processing
Use DenseCRF to refine the detection, and then resize (to 256x256 by default).
```
$ python util/refine_det_by_crf.py <GUIDE_DIR> <PRED_INPUT_DIR> <PRED_OUTPUT_DIR>
```

2. BER computation
```
$ cd evaluation/detection
$ matlab -nodisplay
$ evaluate(<GT_DIR>, <PRED_OUTPUT_DIR>)
```
do not forget to include the `/` after the name of the directory (e.g., `/tmp/foobar/`)


### Training
If you use our proposed SynShadow datasets,
```
$ python train.py --model detect --dataset_root ../datasets/SynShadow --dataset_mode synth --name <IDENTIFIER>
```

If you use existing paired datasets,
```
$ python train.py --model detect --dataset_root ../datasets/<DATASET_NAME> --name <IDENTIFIER>
```

## Removal

### Testing
```
$ python test.py --dataset_root ../datasets/<DATASET_NAME> --name <IDENTIFIER> --mask_to_G <PRECOMP_MASK_NAME> --mask_to_G_thresh 0.95
```

### Quantitative Evaluation
RMSE in LAB color space will be computed.

```
$ cd evaluation/removal
$ matlab -nodisplay
$ evaluation(<DATASET_NAME>, <PRED_OUTPUT_DIR>)
```
DATASET_NAME is `ISTD+` or `SRD+`.
Note that evaluation is done on 480x640 images.

### Training
If you use our proposed SynShadow datasets,
```
$ python train.py --dataset_root ../datasets/SynShadow --dataset_mode synth --name <IDENTIFIER>
```

If you use existing paired datasets (optionally starting from pre-trained checkpoint),
```
$ python train.py --dataset_root ../datasets/ISTD+ --name <IDENTIFIER> (--finetune_from <CHECKPOINT>)
```
