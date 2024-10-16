import av
import os
import cv2
import math
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
# from .utils import generateTransforms


def tensor2img(tensor):
    img = tensor.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# adding gaussian noise
class AddGaussianNoise(object):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        p = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn(img.size()) * (p / 255.0)
        img = torch.clamp(img + noise, 0.0, 1.0)

        return img

# adding poisson noise
class AddPoissonNoise(object):
    def __init__(self, alpha, beta): # 2-4
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        alpha = 2.0
        # alpha = random.uniform(self.alpha, self.beta)
        p = 10 ** alpha
        # img = torch.clamp(img, 0.0, 1.0)
        img = np.random.poisson(img * p) / p
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img).permute(1, 2, 0)

        img = torch.clamp(img, 0.0, 1.0)
        return img

# adding speckle noise
class AddSpeckleNoise(object):
    def __init__(self, sigma_min, sigma_max): # 15.0
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        p = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn(img.size()) * (p / 255.0)

        img = img + img * noise

        img = torch.clamp(img, 0.0, 1.0)
        return img

# adding JPEG compression blocking
class AddJPEGCompression(object):
    def __init__(self, comp_level):
        self.comp_level = comp_level # 10-40

    def __call__(self, img):
        p = random.choice(self.comp_level)
        np_img = img.mul(255).add(0.5).clamp(0, 255).permute(1,2,0).type(torch.uint8).numpy()
        _, encimg = cv2.imencode('.jpg', np_img, [int(cv2.IMWRITE_JPEG_QUALITY), p])

        img = cv2.imdecode(encimg, 3)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img

# adding video compression artifacts
class AddVideoCompression(object):
    def __init__(self, vcodec):
        self.vcodec = vcodec

    def __call__(self, img):
        vcodec = random.choice(self.vcodec)
        if img.shape[-1] % 2 !=0 or img.shape[-2] % 2 !=0:
            while vcodec == 'libx264' or vcodec == 'h264':
                vcodec = random.choice(self.vcodec)

        print(img.shape, vcodec)
        random_seed_suffix = random.randint(0, 100)

        base_root = "./"
        file_name = str(datetime.now()) + "_" + str(random_seed_suffix) + ".mp4"
        file_name = base_root + file_name

        container = av.open(file_name, mode="w")
        stream = container.add_stream(vcodec, rate="1")

        stream.pix_fmt = "yuv420p"
        stream.bit_rate = random.randint(1e4, 1e5)

        img = img.mul(255).add(0.5).clamp(0, 255).permute(1,2,0).type(torch.uint8).numpy()

        stream.width = img.shape[1]
        stream.height = img.shape[0]

        frame = av.VideoFrame.from_ndarray(img)
        packet = stream.encode(frame)
        container.mux(packet)

        packet = stream.encode(None)
        container.mux(packet)
        container.close()

        container = av.open(file_name)

        for t in container.decode(video=0):
            frame = t
        img = frame.to_rgb().to_ndarray()

        container.close()

        os.remove(file_name)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)

        return img

# Gaussian Blur
class AddGaussianBlur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        kernel_size = random.choice(self.kernel_size)

        kernel = random_mixed_kernels(['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                                    [0.405, 0.225, 0.108, 0.027, 0.108, 0.027],
                                    kernel_size,
                                    (0.2, 3),
                                    (0.2, 3),
                                    [-math.pi, math.pi],
                                    noise_range=None)
        img = img.mul(255).add(0.5).clamp(0, 255).permute(1,2,0).type(torch.uint8).numpy()
        img = cv2.filter2D(img, -1, kernel)

        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img


class AddResizingBlur(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        h, w = img.shape[-2:]
        updown_type = random.choices(["up", "down", "keep"], [0.3, 0.4, 0.3])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, 2)
        elif updown_type == "down":
            scale = np.random.uniform(0.5, 1)
        else:
            scale = 1

        mode = random.choice(self.mode)

        recover = False
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
            recover = True

        img = F.interpolate(img, scale_factor=scale, mode=mode)
        img = F.interpolate(img, size=(h, w), mode="bicubic")

        img = torch.clamp(img, 0.0, 1.0)
        if recover:
            img = img.squeeze(0)

        return img


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)

def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         noise_range=None):
    """Randomly generate mixed kernels.

    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    return kernel

def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def random_bivariate_plateau(kernel_size,
                             sigma_x_range,
                             sigma_y_range,
                             rotation_range,
                             beta_range,
                             noise_range=None,
                             isotropic=True):
    """Randomly generate bivariate plateau kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel

def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int):

    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.

    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a plateau-like anisotropic kernel.

    1 / (1+x^(beta))

    Reference: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel

def random_bivariate_generalized_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          beta_range,
                                          noise_range=None,
                                          isotropic=True):
    """Randomly generate bivariate generalized Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a bivariate generalized Gaussian kernel.

    ``Paper: Parameter Estimation For Multivariate Generalized Gaussian Distributions``

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel

def generateTransforms(prob):
    all_transforms = [AddGaussianNoise(10, 15), AddPoissonNoise(alpha=2, beta=4), AddSpeckleNoise(10, 15),
                      AddJPEGCompression([20,30,40]), AddVideoCompression(['libx264', 'h264', 'mpeg4']),
                      AddGaussianBlur([3,5,7]), AddResizingBlur(["area", "bilinear", "bicubic"])]
    random.shuffle(all_transforms)
    selected_transforms = [t for t in all_transforms if random.random() > prob]
    return transforms.Compose(selected_transforms)

def main(input_folder, output_folder, fixed_interval=6, prob=0.55, variation=3):

    # make output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get subfolders in the input folder
    sub_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    if len(sub_folders) == 0:
        sub_folders = [input_folder]

    toTensor = transforms.Compose([transforms.ToTensor()])

    for folder in sub_folders:
        if folder == input_folder:
            input_sub_folder = folder
            output_sub_folder = output_folder
        else:
            input_sub_folder = os.path.join(input_folder, folder)
            output_sub_folder = os.path.join(output_folder, folder)

            # make output subfolder
            if not os.path.exists(output_sub_folder):
                os.makedirs(output_sub_folder)

        cnt = 0
        deg_transform = generateTransforms(prob)

        tmp_interval = fixed_interval
        print(f"Processing {str(input_sub_folder)}")
        for frame_name in sorted(os.listdir(input_sub_folder)):
            if cnt == 0:
                deg_transform = generateTransforms(prob)
                tmp_interval = random.randint(fixed_interval - variation, fixed_interval + variation)
                assert tmp_interval > 0, "The interval of degradation changes must be positive."
                print(f"Temporal Interval: {str(tmp_interval)}.")

            input_frame_path = os.path.join(input_sub_folder, frame_name)

            frame = Image.open(input_frame_path)
            frame = frame.convert('RGB')
            frame = toTensor(frame)

            frame = deg_transform(frame)
            frame = tensor2img(frame)

            output_frame_path = os.path.join(output_sub_folder, frame_name)
            cv2.imwrite(output_frame_path, frame)

            cnt += 1
            cnt %= tmp_interval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to add degradations to video sequences.")
    parser.add_argument('--input_dir', required=True, type=str, help='Input directory containing the image sequences.')
    parser.add_argument('--output_dir', required=True, type=str, help='Output directory to save the degraded sequences.')
    parser.add_argument('--fixed_interval', type=int, default=12, help='Number of continuous frames with the same degradation.')
    parser.add_argument('--prob', type=float, default=0.55, help='Probability to skip a transformation.')
    parser.add_argument('--variation', type=int, default=6, help='Range for the variability of continuous frames.')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.fixed_interval, args.prob, args.variation)
    
    # /data/haiyu/datasets/videoData/AverNetDataset
    # /data/haiyu/datasets/videoData/DAVIS-test