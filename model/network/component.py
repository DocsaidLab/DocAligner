from typing import List

import docsaidkit.torch as DT
import torch
import torch.nn as nn
from docsaidkit.torch import (Hsigmoid, build_backbone, build_transformer,
                              list_transformer, replace_module)


class Backbone(nn.Module):

    def __init__(self, name, replace_components: bool = True, **kwargs):
        super().__init__()
        self.backbone = build_transformer(name=name, **kwargs) \
            if name in list_transformer() else build_backbone(name=name, **kwargs)

        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            self.channels = [i.size(1) for i in self.backbone(dummy)]

        # For quantization
        if replace_components:
            replace_module(self.backbone, nn.Hardswish, nn.ReLU())
            replace_module(self.backbone, nn.Hardsigmoid, Hsigmoid())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class DocAlignedHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int = 8,
    ) -> None:
        super().__init__()
        self.point_regression = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_channels, num_points)
        )
        self.edge_regression = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.point_regression(x[-1]), self.edge_regression(x[0])
