# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlocksWithInputConv, PixelShufflePack, flow_warp
from basicsr.archs.spynet_arch import SpyNet

from mmcv.ops.deform_conv import DeformConv2dFunction
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class PromptGenBlock(nn.Module):
    def __init__(self, emb_dim=64, prompt_dim=96, prompt_len=5, prompt_size=96):
        super(PromptGenBlock,self).__init__()
        self.prompt_dim = prompt_dim

        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_proj = nn.Linear(emb_dim, prompt_len) # project feature embeddings to prompt dims
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        prompt = interpolate (proj(feature_embedding) * param)
        """
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1)) # (b, t, c) -> (b*t, c)
        prompt_weights = F.softmax(self.linear_proj(emb), dim=1) # channel dim softmax (b, prompt_len)
        # (b, prompt_len, 1, 1, 1) * (b, prompt_len, prompt_dim, prompt_size, prompt_size)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (h, w), mode="bilinear")
        prompt = self.conv(prompt) # b, promptdim, h, w
        return prompt


@ARCH_REGISTRY.register()
class AverNet_OLD(nn.Module):
    """
        Key frame & alignment
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 spynet_pretrained=None,
                 keyframe_interval=6,
                 prompt_size=96,
                 prompt_dim=108,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(load_path=spynet_pretrained)

        # feature extraction module
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.key_extract = PromptGenBlock(emb_dim=mid_channels, prompt_size=self.prompt_size, prompt_dim=self.prompt_dim)
        self.key_proj = ResidualBlocksWithInputConv(mid_channels + self.prompt_dim, mid_channels, 3)

        self.key_fusion = nn.Conv2d(2 * self.mid_channels, self.mid_channels, 3, 1, 1, bias=True)

        # key frame
        self.keyframe_interval = keyframe_interval

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = DeformableAlignment(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=24,
                max_residue_magnitude=max_residue_magnitude,
                prompt_dim=self.prompt_dim)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_keyframe_feature(self, x, keyframe_idx):
        feats_keyframe = {}
        for i in keyframe_idx:
            if self.cpu_cache:
                x_i = x[i].cuda()
            else:
                x_i = x[i]
            feats_keyframe[i] = self.key_extract(x_i)
            feats_keyframe[i] = torch.cat([feats_keyframe[i], x_i], dim=1)
            feats_keyframe[i] = self.key_proj(feats_keyframe[i])
            if self.cpu_cache:
                feats_keyframe[i] = feats_keyframe[i].cpu()
                torch.cuda.empty_cache()
        return feats_keyframe

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            
            hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25,
            mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # generate keyframe features
        keyframe_idx = list(range(0, t, self.keyframe_interval))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # last frame is a keyframe
        feats_keyframe = self.get_keyframe_feature(feats['spatial'], keyframe_idx)

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                n, t, _, h, w = flows.size()
                frame_idx = list(range(0, t + 1))
                flow_idx = list(range(-1, t))
                mapping_idx = list(range(0, len(feats['spatial'])))
                mapping_idx += mapping_idx[::-1]

                if direction == 'backward':
                    frame_idx = frame_idx[::-1]
                    flow_idx = frame_idx

                feat_prop = flows.new_zeros(n, self.mid_channels, h, w)

                for i, idx in enumerate(frame_idx):
                    x_i = feats['spatial'][mapping_idx[idx]]
                    if self.cpu_cache:
                        x_i = x_i.cuda()
                        feat_prop = feat_prop.cuda()

                    pre_feat = feat_prop.clone()

                    if i > 0:
                        flow = flows[:, flow_idx[i], :, :, :]
                        if self.cpu_cache:
                            flow = flow.cuda()
                        feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                    # key frame prompt guidance

                    # not key frame / add deformable alignment
                    if i > 0:
                        cond = torch.cat([feat_prop, x_i], dim=1)
                        # alignment / need mod
                        feat_prop = self.deform_align[module](pre_feat, cond, flow)

                    if idx in keyframe_idx:
                        if self.cpu_cache:
                            feats_keyframe_t = feats_keyframe[idx].cuda()
                        else:
                            feats_keyframe_t = feats_keyframe[idx]

                        feat_prop = torch.cat([feat_prop, feats_keyframe_t], dim=1)
                        feat_prop = self.key_fusion(feat_prop)

                    # concatenate the residual infor
                    feat = [x_i] + [
                        feats[k][idx]
                        for k in feats if k not in ['spatial', module]
                    ] + [feat_prop]
                    if self.cpu_cache:
                        feat = [f.cuda() for f in feat]

                    feat = torch.cat(feat, dim=1)
                    feat_prop = feat_prop + self.backbone[module](feat)
                    if self.cpu_cache:
                        feat_prop = feat_prop.cpu()
                        torch.cuda.empty_cache()
                    feats[module].append(feat_prop)

                if direction == 'backward':
                    feats[module] = feats[module][::-1]

                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


# Prompt guided alignment
class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.prompt_dim = kwargs.pop('prompt_dim', 96)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.proj_1 = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)
        self.prompt_extract = PromptGenBlock(emb_dim=self.out_channels, prompt_dim=self.prompt_dim)
        self.proj_2 = ResidualBlocksWithInputConv(self.out_channels + self.prompt_dim, self.out_channels, 1)

        self.fusion = nn.Conv2d(3 * self.out_channels, 2 * self.out_channels, 3, 1, 1, bias=True)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond, flow):
        # cond feat_prop + x_i
        proj_cond = self.proj_1(cond)
        prompt = self.prompt_extract(proj_cond)
        prompt_guide_cond = self.proj_2(torch.cat([proj_cond, prompt], dim=1))
        cond = self.fusion(torch.cat([prompt_guide_cond, cond], dim=1))

        cond = torch.cat([cond, flow], dim=1) # guide
        out = self.conv_offset(cond)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1,
                                                offset.size(1) // 2, 1,
                                                1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)