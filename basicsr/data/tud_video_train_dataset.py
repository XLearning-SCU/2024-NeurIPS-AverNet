import numpy as np
import random
import cv2
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms

from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.data.transforms import AddGaussianNoise, AddPoissonNoise, AddSpeckleNoise, AddJPEGCompression, AddVideoCompression, AddGaussianBlur, AddResizingBlur

@DATASET_REGISTRY.register()
class TUDVideoDataset(data.Dataset):
    def __init__(self, opt):
        super(TUDVideoDataset, self).__init__()
        self.opt = opt

        self.prob = opt.get('prob', 0.5)
        self.deg_interval = opt.get('deg_interval', 4)
        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root = Path(opt['dataroot_gt'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'jpg')
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])

        val_partition = []

        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True, flag='color')
            img_gts.append(img_gt)

        # randomly crop
        img_gts, _ = paired_random_crop(img_gts, img_gts, self.gt_size, 1, img_gt_path)


        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        img_gts = img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)

        img_lqs = img_gts.clone()

        # degradation pipeline
        t = img_gts.shape[0]
        for i in range(0, t, self.deg_interval):
            all_transforms = [AddGaussianNoise(10, 15), AddPoissonNoise(alpha=2, beta=4),
                              AddSpeckleNoise(10, 15),
                            AddJPEGCompression([20,30,40]), AddVideoCompression(['libx264', 'h264', 'mpeg4']),
                          AddGaussianBlur([3,5,7]),
                          AddResizingBlur(["area", "bilinear", "bicubic"])
                          ]
            random.shuffle(all_transforms)
            selected_transform = [t for t in all_transforms if random.random() > self.prob]
            deg_transform = transforms.Compose(selected_transform)
            for j in range(i, i + self.deg_interval):
                if j == t:
                    break
                img_lqs[j,:,:,:] = deg_transform(img_lqs[j,:,:,:])

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)