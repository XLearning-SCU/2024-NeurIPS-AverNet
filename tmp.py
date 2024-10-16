from pyiqa.utils.img_util import imread2tensor

img = imread2tensor('/data/haiyu/datasets/videoData/DAVIS-test/aerobatics/00000.jpg', rgb=True)
print(img.shape)
print(img[0])
