from os import path as osp
from PIL import Image

from basicsr.utils import scandir


import os
import glob
import shutil


def generate_meta_info_txt(data_path, meta_info_path):
    '''generate meta_info_DAVIS_GT.txt for DAVIS

    :param data_path: dataset path.
    :return: None
    '''
    f= open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, '*')))
    total_frames = 0
    for path in file_list:
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        first_frame = frames[0]

        img = Image.open(first_frame)
        width, height = img.size
        mode = img.mode
        if mode == 'RGB':
            n_channel = 3
        elif mode == 'L':
            n_channel = 1
        else:
            raise ValueError(f'Unsupported mode {mode}.')

        start_frame = os.path.basename(frames[0]).split('.')[0]

        print(name, len(frames), start_frame)
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} ({height},{width},{n_channel}) {start_frame}\r\n")

    # assert total_frames == 6208, f'DAVIS training set should have 6208 images, but got {total_frames} images'

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate meta information for DAVIS.")
    
    parser.add_argument("--dataset_path", help="The path of training datasets.")
    parser.add_argument("--output_path", default='./basicsr/data/meta_info/training_meta_info.txt', 
                        help="The save path of meta info file.")

    args = parser.parse_args()
    
    generate_meta_info_txt(osp.abspath(args.dataset_path), osp.abspath(args.output_path))