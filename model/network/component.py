import math
from typing import List

import docsaidkit.torch as DT
import torch
import torch.nn as nn
from docsaidkit.torch import (Hsigmoid, build_backbone, build_transformer,
                              list_transformer, replace_module)


class Backbone(nn.Module):

    def __init__(self, name, replace_components: bool = False, **kwargs):
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


class PointRegHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.point_feats = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
        )
        self.point_reg = nn.Linear(in_channels, num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.point_feats(x[-1])
        points = self.point_reg(feats)
        return points,


class DocAlignedHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.point_regression = nn.Sequential(
            DT.Flatten(),
            nn.Linear(in_channels * 8 * 8, num_points)
        )
        self.edge_regression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.point_regression(x[-1]), self.edge_regression(x[0])


class BoxPointRegHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.point_feats = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
        )
        self.point_reg = nn.Linear(in_channels * 5, num_points)

        self.edge_regression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        feats = [self.point_feats(x) for x in xs]
        points = self.point_reg(torch.cat(feats, dim=1))
        return points, self.edge_regression(xs[0])


class ViTBoxPointRegHead(nn.Module):

    def __init__(
        self,
        in_c: int,
        d_model: int,
        num_layers: int,
        num_points: int,
        image_size: List[int],
        patch_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = DT.ImageEncoder(
            d_model=d_model,
            num_layers=num_layers,
            image_size=image_size,
            patch_size=patch_size,
            in_c=in_c,
            **kwargs
        )
        self.point_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_points)
        )
        self.edge_reg = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 1, 1),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(d_model, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.cls_proj = nn.Sequential(
            DT.GAP(),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model)
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        edge_mask = self.edge_reg(xs[0])
        x, *_ = self.encoder(xs[0], self.cls_proj(xs[-1]).unsqueeze(1))
        points = self.point_reg(x)
        return points, edge_mask


class ViT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_layers: int,
        num_points: int,
        image_size: List[int] = [256, 256],
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        h, w = image_size
        nh, nw = h // patch_size, w // patch_size

        # 初始化'cls_token'及位置嵌入
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.Tensor(nh*nw, 1, d_model))

        # 使用特定的初始化方法
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.tokenizer = nn.Conv2d(
            in_channels,
            d_model,
            patch_size,
            patch_size,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                **kwargs,
            ),
            num_layers=num_layers
        )

        self.point_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, num_points)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.tokenizer(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = x + self.pos_emb.expand(-1, x.size(1), -1)
        cls_tokens = self.cls_token.expand(-1, x.size(1), -1)
        x = torch.cat((cls_tokens, x), dim=0)
        x = self.encoder(x)
        cls_token, _ = torch.split(x, [1, x.size(0) - 1], dim=0)
        points = self.point_reg(cls_token.squeeze(0))
        return points
