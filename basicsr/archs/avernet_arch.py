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

class PromptGenerationInteraction(nn.Module):
    """
        Generating an input-conditioned prompt and integrate it into features.
    """
    def __init__(self, embed_dim=64, prompt_dim=96, prompt_len=5, prompt_size=96, num_blocks=3, align=False):
        super(PromptGenerationInteraction, self).__init__()
        self.align = align
        self.prompt_dim = prompt_dim

        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_proj = nn.Linear(embed_dim, prompt_len) # project feature embeddings to prompt dims
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = ResidualBlocksWithInputConv(embed_dim + prompt_dim, embed_dim, num_blocks)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1)) 
        prompt_weights = F.softmax(self.linear_proj(emb), dim=1) 

        input_conditioned_prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        input_conditioned_prompt = torch.sum(input_conditioned_prompt, dim=1)
        input_conditioned_prompt = F.interpolate(input_conditioned_prompt, (h, w), mode="bilinear")
        input_conditioned_prompt = self.conv(input_conditioned_prompt) # b, promptdim, h, w

        output = self.residual(torch.cat([x, input_conditioned_prompt], dim=1))

        return output

@ARCH_REGISTRY.register()
class AverNet(nn.Module):

    def __init__(self,
                 mid_channels=96,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 spynet_pretrained=None,
                 keyframe_interval=6,
                 prompt_size=96,
                 prompt_dim=96,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(load_path=spynet_pretrained)

        # shallow feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))


        self.PGI_prop = PromptGenerationInteraction(embed_dim=mid_channels, prompt_dim=self.prompt_dim,
                                                    prompt_size=self.prompt_size, num_blocks=3)

        self.key_fusion = nn.Conv2d(2 * self.mid_channels, self.mid_channels, 3, 1, 1, bias=True)

        # key frame
        self.keyframe_interval = keyframe_interval

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = PromptGuidedAlignment(
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

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_keyframe_feature(self, x, keyframe_idx):
        feats_keyframe = {}
        for i in keyframe_idx:
            if self.cpu_cache:
                x_i = x[i].cuda()
            else:
                x_i = x[i]

            feats_keyframe[i] = self.PGI_prop(x_i)

            if self.cpu_cache:
                feats_keyframe[i] = feats_keyframe[i].cpu()
                torch.cuda.empty_cache()
        return feats_keyframe

    def compute_flow(self, lqs):

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
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25,
            mode='bicubic').view(n, t, c, h//4, w//4)

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

                    # Prompt-guided Alignment
                    if i > 0:
                        cond = torch.cat([feat_prop, x_i], dim=1)
                        feat_prop = self.deform_align[module](pre_feat, cond, flow)
                        
                    # Prompt-conditioned Enhancement
                    if idx in keyframe_idx:
                        if self.cpu_cache:
                            feats_keyframe_t = feats_keyframe[idx].cuda()
                        else:
                            feats_keyframe_t = feats_keyframe[idx]

                        feat_prop = self.key_fusion(torch.cat([feat_prop, feats_keyframe_t], dim=1))

                    # concatenate the residual info
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
class PromptGuidedAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.prompt_dim = kwargs.pop('prompt_dim', 96)

        super(PromptGuidedAlignment, self).__init__(*args, **kwargs)

        self.proj = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)

        self.PGI_align = PromptGenerationInteraction(embed_dim=self.out_channels, prompt_dim=self.prompt_dim,
                                                     num_blocks=1, align=True)

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
        proj_cond = self.proj(cond)

        integrated_features = self.PGI_align(proj_cond)
        cond = self.fusion(torch.cat([integrated_features, cond], dim=1))

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