from pyiqa.utils.img_util import imread2tensor

img = imread2tensor('/xlearning/haiyu/datasets/videoData/DAVIS-GT/aerobatics/00000.jpg', rgb=True)
print(img)