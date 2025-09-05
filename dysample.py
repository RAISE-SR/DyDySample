import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DyDySample(nn.Module):
    def __init__(self, in_channels, out_ch, scale=2, groups=4, end_convolution=True):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution

        out_channels = 2 * groups * scale**2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope  = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, 1)

        self.register_buffer(
            "init_pos",
            self._init_pos(scale=self.scale, groups=self.groups),
            persistent=False
        )

    @staticmethod
    def _init_pos(scale: int, groups: int):
        s = scale
        h = torch.arange((-(s - 1)) / 2.0, ((s - 1) / 2.0) + 1.0) / float(s)
        grid = torch.stack(torch.meshgrid(h, h, indexing="ij"))
        grid = grid.transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1)
        return grid

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        s, g = self.scale, self.groups

        init_pos = self.init_pos.to(dtype=x.dtype, device=x.device)
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + init_pos
        offset = offset.reshape(B, 2, g * s * s, H, W)

        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        yy = coords_h.view(1, 1, H, 1).expand(1, 1, H, W)
        xx = coords_w.view(1, 1, 1, W).expand(1, 1, H, W)
        base = torch.cat([xx, yy], dim=0).view(1, 2, 1, H, W)
        base_exp = base.expand(1, 2, g * s * s, H, W)

        norm = torch.tensor([2.0 / W, 2.0 / H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = (base_exp + offset) * norm - 1.0

        coords = coords.reshape(B, 2 * g * s * s, H, W)
        coords = F.pixel_shuffle(coords, s)

        coords = coords.reshape(B, 2, g, s * H, s * W).permute(0, 2, 3, 4, 1).reshape(B * g, s * H, s * W, 2)

        c_pg = C // g
        x_grouped = x.reshape(B, g, c_pg, H, W).reshape(B * g, c_pg, H, W)

        out = F.grid_sample(
            x_grouped,
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border"
        ).reshape(B, g * c_pg, s * H, s * W)

        if self.end_convolution:
            out = self.end_conv(out)
        return out
