# AverNet (NeurIPS 2024)

### All-in-one Video Restoration for Time-varying Unknown Degradations
Haiyu Zhao, Lei Tian, Xinyan Xiao, Peng Hu, Yuanbiao Gou, Xi Peng

## Installation

1. `git clone https://github.com/Pandint/AverNet.git`
2. `cd AverNet`
3. `pip install -r requirement.txt`
4. `pip install -U openmim` (install mmcv) [deformable convolution dependency]
5. `mim install "mmcv>=2.0.0rc1"`
6. `python setup.py develop`

## Dataset Preparation

### Training Data

1. Download the DAVIS dataset from [official webset](https://davischallenge.org/davis2016/code.html) or [Google Drive]().
2. Synthesize the low-quality (LQ) videos through `scripts/data_preparation/synthesize_datasets.py`.
```
python scripts/data_preparation/synthesize_datasets.py --input_dir 'The root of DAVIS' --output_dir 'LQ roots' --continuous_frames 6
```
3. Generate meta_info files for the training sets.
> This step can be ommited if you use the `DAVIS` dataset for training since the `DAVIS_meta_info.txt` file is already generated. (located in `basicsr/data/meta_info/DAVIS_meta_info.txt`)
```
python scripts/data_preparation/generate_meta_info.py --dataset_path 'The root of training sets'
# The meta infomation file is automatically saved in `basicsr/data/meta_info/training_meta_info.txt`
```

### Testing Data

The test sets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1NIV4YSMhBJfjQu2SNQgvuluTAZDeFXH6?usp=sharing). 

## Testing

1. Download the pretrained weights of SPyNet and AverNet from [Google Drive](https://drive.google.com/drive/folders/1S0h4wKGPm4pugx94gs1r3aLDzS06vGt3?usp=sharing).
2. Put the SPyNet weights to `experiments/pretrained_models/flownet/` and AverNet weights to `experiments/pretrained_models/`.
3. Modify the option yaml file in `options/test/` to begin. Then run the testing.
```
python basicsr/test.py -opt options/test/test_AverNet_DAVIS_T6.yml
```
Note that the `dataroot_lq` and `dataroot_gt` in the yaml file should be modified to LQ and GT folders of test sets, respectively.

## Training

1. Put the SPyNet weights to `experiments/pretrained_models/flownet/`.
2. Modify the option yaml file in `options/train/` to begin. Then run training.
```
python basicsr/train.py -opt options/train/train_AverNet_DAVIS.yml
```
Note that the `dataroot_lq` and `dataroot_gt` in the yaml file should be modified to LQ and GT folders of training datasets, respectively.

## Acknowledgements
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks the authors for their codes!

## Contact
If you have any question, please contact: haiyuzhao.gm@gmail.com