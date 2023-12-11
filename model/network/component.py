import math
from typing import List

import docsaidkit.torch as DT
import torch
import torch.nn as nn
from docsaidkit.torch import (Hsigmoid, build_backbone, build_transformer,
                              list_transformer, replace_module)


class Transpose(nn.Module):

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


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
        **kwargs,
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


class PointRegBNHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.point_feats = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
        )
        self.point_reg = nn.Linear(in_channels, num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.point_feats(x[-1])
        points = self.point_reg(feats)
        return points,


class PointRegFeatMergeHead(nn.Module):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        num_points: int,
        in_channels: int = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Override in_channels_list
        if in_channels is not None:
            in_channels_list = [in_channels] * 5

        self.point_feats = nn.ModuleList([
            nn.Sequential(
                DT.GAP(),
                nn.Linear(in_c, out_channels),
                nn.LayerNorm(out_channels),
            )
            for in_c in in_channels_list
        ])

        self.point_reg = nn.Linear(out_channels * 5, num_points)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        feats = [point_feat(x) for point_feat, x in zip(self.point_feats, xs)]
        feats = torch.cat(feats, dim=1)
        points = self.point_reg(feats)
        return points,


class PointRegFeatMergeEdgeHead(nn.Module):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        num_points: int,
        in_channels: int = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Override in_channels_list
        if in_channels is not None:
            in_channels_list = [in_channels] * 5

        self.point_feats = nn.ModuleList([
            nn.Sequential(
                DT.GAP(),
                nn.Linear(in_c, out_channels),
                nn.LayerNorm(out_channels),
            )
            for in_c in in_channels_list
        ])

        self.point_reg = nn.Linear(out_channels * 5, num_points)

        self.edge_reg = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        feats = [point_feat(x) for point_feat, x in zip(self.point_feats, xs)]
        feats = torch.cat(feats, dim=1)
        points = self.point_reg(feats)
        return points, self.edge_reg(xs[0])


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
        in_channels_list: List[int],
        d_model: int,
        num_layers: int,
        num_points: int,
        image_size: List[int],
        patch_size: int,
        nhead: int,
        **kwargs,
    ) -> None:
        super().__init__()

        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)

        n_tokens = sum([
            (h // 2**i // patch_size) * (w // 2**i // patch_size)
            for i in range(5)
        ])
        self.pos_emb = nn.Parameter(torch.Tensor(1, n_tokens, d_model))

        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.tokenizer = nn.ModuleList([
            nn.Sequential(
                DT.SeparableConvBlock(in_c, d_model, patch_size, patch_size),
                nn.Flatten(2),
                Transpose(1, 2),
                nn.LayerNorm(d_model)
            ) for in_c in in_channels_list
        ])

        self.cls_token = DT.ImageEncoder(
            d_model=d_model,
            num_layers=1,
            image_size=8,
            patch_size=1,
            in_c=in_channels_list[4],
        )
        self.cls_proj = nn.Linear(d_model, d_model * 5)

        self.encoder = nn.ModuleList([
            DT.ImageEncoderLayer(d_model, nhead=nhead, **kwargs)
            for _ in range(num_layers)
        ])

        self.point_reg = nn.Linear(d_model, num_points)
        self.box_reg = nn.Linear(d_model, 4)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        cls_token, *_ = self.cls_token(xs[4])
        cls_token = self.cls_proj(cls_token).reshape(cls_token.size(0), 5, -1)
        xs = [tokenizer(x) for tokenizer, x in zip(self.tokenizer, xs)]
        x = torch.cat(xs, dim=1)
        x = x + self.pos_emb.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        for layer in self.encoder:
            x, _ = layer(x)

        x_poly, x_box, _ = torch.split(x, [4, 1, x.size(1) - 5], dim=1)
        points = self.point_reg(x_poly)
        boxes = self.box_reg(x_box)
        return points, boxes


class ViT(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        in_channels: int = 3,
        image_size: List[int] = [256, 256],
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        h, w = image_size
        nh, nw = h // patch_size, w // patch_size

        # 初始化 4 個 token 及位置嵌入
        self.p1 = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.p2 = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.p3 = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.p4 = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.Tensor(nh*nw, 1, d_model))

        # 使用特定的初始化方法
        nn.init.kaiming_uniform_(self.p1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.p2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.p3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.p4, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.tokenizer = nn.Conv2d(
            in_channels,
            d_model,
            patch_size,
            patch_size,
            bias=False
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.tokenizer(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = x + self.pos_emb.expand(-1, x.size(1), -1)
        p1 = self.p1.expand(-1, x.size(1), -1)
        p2 = self.p2.expand(-1, x.size(1), -1)
        p3 = self.p3.expand(-1, x.size(1), -1)
        p4 = self.p4.expand(-1, x.size(1), -1)
        x = torch.cat((p1, p2, p3, p4, x), dim=0)
        x = self.encoder(x)
        corner_token, img_token = torch.split(x, [4, x.size(0) - 4], dim=0)
        return corner_token.squeeze(0), img_token


class ViTPointRegHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.point_reg = nn.Linear(in_channels, num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        corner_token, _ = x
        corner_token = corner_token.transpose(0, 1)
        points = self.point_reg(corner_token)
        return points,


class ViTBoxPointEdgeRegHead(nn.Module):

    def __init__(
        self,
        in_c: int,
        d_model: int,
        num_layers: int,
        num_points: int,
        image_size: List[int],
        patch_size: int,
        nhead: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = DT.ImageEncoder(
            d_model=d_model,
            num_layers=num_layers,
            image_size=image_size,
            patch_size=patch_size,
            in_c=in_c,
            nhead=nhead,
        )
        self.cls_token = DT.ImageEncoder(
            in_c=in_c,
            d_model=d_model,
            num_layers=num_layers,
            image_size=8,
            patch_size=1,
            nhead=nhead,
        )
        self.point_reg = nn.Linear(d_model, num_points)
        self.box_reg = nn.Linear(d_model, 4)
        self.edge_reg = nn.Sequential(
            nn.Conv2d(d_model, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        cls_token, *_ = self.cls_token(xs[4])
        hid, *_ = self.encoder(xs[0], cls_token=cls_token.unsqueeze(1))
        points = self.point_reg(hid)
        boxes = self.box_reg(hid)
        edges = self.edge_reg(xs[0])
        aux_points = self.point_reg(cls_token)
        aux_boxes = self.box_reg(cls_token)
        return points, edges, boxes, aux_points, aux_boxes
