## Install

### Install dependencies

pip install -r requirement.txt

(install mmcv) [deformable convolution dependency]
pip install -U openmim
mim install "mmcv>=2.0.0rc1"

## Testing



## Training

### Dataset Preparation

> This step can be ommited if you use the `DAVIS` dataset for training since the `DAVIS_meta_info.txt` file is already generated.

Generate training_meta_info.txt by the script in `scripts/data_preparation`.

1. Change director to `AverNet/`.

2. Run the script.

~~~python
python scripts/data_preparation/generate_meta_info.py --dataset_path ./datasets/DAVIS
# dataset_path is the path of training dataset
# The meta infomation file is automatically saved in `basicsr/data/meta_info/training_meta_info.txt`
~~~

### 

Training