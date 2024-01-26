import math
from typing import List, Tuple

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


class HeatmapEdgeRegHead(nn.Module):

    def __init__(self, in_c: int, **kwargs) -> None:
        super().__init__()
        self.heatmap_reg = nn.Sequential(
            nn.Conv2d(in_c, 4, 3, padding=1),
            nn.Sigmoid()
        )
        self.edge_reg = nn.Sequential(
            nn.Conv2d(in_c, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.has_obj = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_c, 1)
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        heatmap = self.heatmap_reg(xs[0]).squeeze(1)
        edge = self.edge_reg(xs[0]).squeeze(1)
        has_obj = self.has_obj(xs[4])
        return heatmap, edge, has_obj


class HeatmapEdgeThicknessRegHead(nn.Module):

    def __init__(self, in_c: int, **kwargs) -> None:
        super().__init__()
        self.heatmap_reg = nn.Sequential(
            nn.Conv2d(in_c, 4, 3, padding=1),
            nn.Sigmoid()
        )
        self.edge_reg = nn.Sequential(
            nn.Conv2d(in_c, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.thickness = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_c, in_c),
            nn.LayerNorm(in_c),
            nn.Linear(in_c, in_c),
            nn.LayerNorm(in_c),
            nn.Linear(in_c, 1)
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        heatmap = self.heatmap_reg(xs[0]).squeeze(1)
        edge = self.edge_reg(xs[0]).squeeze(1)
        thickness = self.thickness(xs[4])
        return heatmap, edge, thickness


class HeatmapEdgeRegUp2Head(nn.Module):

    def __init__(self, in_c: int, **kwargs) -> None:
        super().__init__()
        self.heatmap_reg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, 4, 3, padding=1),
            nn.Sigmoid()
        )
        self.edge_reg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        heatmap = self.heatmap_reg(xs[0]).squeeze(1)
        edge = self.edge_reg(xs[0]).squeeze(1)
        return heatmap, edge


class HeatmapEdgePointRegDecoderHead(nn.Module):

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

        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)

        nh, nw = (h // patch_size),  (w // patch_size)
        mh, mw = (h // 16),  (w // 16)

        self.pos_emb_low = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        self.pos_emb_high = nn.Parameter(torch.Tensor(mh * mw, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb_low, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_high, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer_low = nn.Sequential(
            nn.Conv2d(in_c, d_model, patch_size, patch_size),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_low = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.tokenizer_high = nn.Sequential(
            nn.Conv2d(in_c, d_model, 1, 1, 0),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_high = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.point_reg = nn.Linear(d_model, num_points)
        self.box_reg = nn.Linear(d_model, 4)
        self.has_obj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 2, 1),
            nn.BatchNorm2d(d_model),
            DT.GAP(),
            nn.Linear(d_model, 1)
        )
        self.heatmap_rec = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, 4, 3, padding=1),
            nn.Sigmoid()
        )
        self.edge_rec = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:

        has_obj = self.has_obj(xs[4])

        h_lavel_feat = self.tokenizer_high(xs[4])
        pos_emb_high = self.pos_emb_high.expand(-1, h_lavel_feat.size(1), -1)
        h_lavel_feat = h_lavel_feat + pos_emb_high

        query = self.cls_token.expand(-1, h_lavel_feat.size(1), -1)
        query = self.decoder_high(query, h_lavel_feat)
        query = self.norm1(query)  # t, b, d

        aux_points = self.point_reg(query.transpose(0, 1).squeeze(1))
        aux_boxes = self.box_reg(query.transpose(0, 1).squeeze(1))

        l_level_feat = self.tokenizer_low(xs[0])
        pos_emb_low = self.pos_emb_low.expand(-1, l_level_feat.size(1), -1)
        l_level_feat = l_level_feat + pos_emb_low
        query = self.decoder_low(query, l_level_feat)
        query = self.norm2(query)

        points = self.point_reg(query.transpose(0, 1).squeeze(1))
        boxes = self.box_reg(query.transpose(0, 1).squeeze(1))

        heatmap = self.heatmap_rec(xs[0]).squeeze(1)
        edge = self.edge_rec(xs[0]).squeeze(1)

        return points, heatmap, edge, has_obj, aux_points, boxes, aux_boxes


class PointRegHead(nn.Module):

    def __init__(
        self,
        in_c: int,
        n_points: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.point_feats = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_c, in_c),
            nn.LayerNorm(in_c),
        )
        self.point_reg = nn.Linear(in_c, n_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.point_feats(x[-1])
        points = self.point_reg(feats)
        return points,


class ViTBoxPointEdgeRecHead(nn.Module):

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        cls_token, *_ = self.cls_token(xs[4])
        cls_token = self.norm1(cls_token)
        hid, *_ = self.encoder(xs[0], cls_token=cls_token.unsqueeze(1))
        hid = self.norm2(hid)
        points = self.point_reg(hid)
        boxes = self.box_reg(hid)
        edges = self.edge_reg(xs[0])
        aux_points = self.point_reg(cls_token)
        aux_boxes = self.box_reg(cls_token)
        return points, edges, boxes, aux_points, aux_boxes


class BoxPointEdgeDecoderHead(nn.Module):

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

        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)

        nh, nw = (h // patch_size),  (w // patch_size)

        self.pos_emb_low = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        self.pos_emb_high = nn.Parameter(torch.Tensor(8 * 8, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb_low, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_high, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer_low = nn.Sequential(
            nn.Conv2d(in_c, d_model, patch_size, patch_size),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_low = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.tokenizer_high = nn.Sequential(
            nn.Conv2d(in_c, d_model, 1, 1, 0),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_high = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.point_reg1 = nn.Linear(d_model, num_points)
        self.point_reg2 = nn.Sequential(
            nn.Linear(d_model, num_points),
            nn.Tanh()
        )
        self.box_reg1 = nn.Linear(d_model, 4)
        self.box_reg2 = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Tanh()
        )
        self.edge_reg = nn.Sequential(
            nn.Conv2d(d_model, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:

        h_lavel_feat = self.tokenizer_high(xs[4])
        pos_emb_high = self.pos_emb_high.expand(-1, h_lavel_feat.size(1), -1)
        h_lavel_feat = h_lavel_feat + pos_emb_high

        query = self.cls_token.expand(-1, h_lavel_feat.size(1), -1)
        query = self.decoder_high(query, h_lavel_feat)
        query = self.norm1(query)  # t, b, d
        points = self.point_reg1(query.transpose(0, 1).squeeze(1))
        boxes = self.box_reg1(query.transpose(0, 1).squeeze(1))

        l_level_feat = self.tokenizer_low(xs[0])
        pos_emb_low = self.pos_emb_low.expand(-1, l_level_feat.size(1), -1)
        l_level_feat = l_level_feat + pos_emb_low
        query = self.decoder_low(query, l_level_feat)
        query = self.norm2(query)
        points = points + self.point_reg2(query.transpose(0, 1).squeeze(1))
        boxes = boxes + self.box_reg2(query.transpose(0, 1).squeeze(1))
        edges = self.edge_reg(xs[0])

        return points, edges, boxes


class BoxPointEdgeDecoderAuxHead(nn.Module):

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

        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)

        nh, nw = (h // patch_size),  (w // patch_size)
        mh, mw = (h // 16),  (w // 16)

        self.pos_emb_low = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        self.pos_emb_high = nn.Parameter(torch.Tensor(mh * mw, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb_low, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_high, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer_low = nn.Sequential(
            nn.Conv2d(in_c, d_model, patch_size, patch_size),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_low = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.tokenizer_high = nn.Sequential(
            nn.Conv2d(in_c, d_model, 1, 1, 0),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_high = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.point_reg = nn.Linear(d_model, num_points)
        self.box_reg = nn.Linear(d_model, 4)
        self.edge_reg = nn.Sequential(
            nn.Conv2d(d_model, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:

        h_lavel_feat = self.tokenizer_high(xs[4])
        pos_emb_high = self.pos_emb_high.expand(-1, h_lavel_feat.size(1), -1)
        h_lavel_feat = h_lavel_feat + pos_emb_high

        query = self.cls_token.expand(-1, h_lavel_feat.size(1), -1)
        query = self.decoder_high(query, h_lavel_feat)
        query = self.norm1(query)  # t, b, d
        aux_points = self.point_reg(query.transpose(0, 1).squeeze(1))
        aux_boxes = self.box_reg(query.transpose(0, 1).squeeze(1))

        l_level_feat = self.tokenizer_low(xs[0])
        pos_emb_low = self.pos_emb_low.expand(-1, l_level_feat.size(1), -1)
        l_level_feat = l_level_feat + pos_emb_low
        query = self.decoder_low(query, l_level_feat)
        query = self.norm2(query)
        points = self.point_reg(query.transpose(0, 1).squeeze(1))
        boxes = self.box_reg(query.transpose(0, 1).squeeze(1))
        edges = self.edge_reg(xs[0])

        return points, edges, boxes, aux_points, aux_boxes


class BoxPointEdgeDecoderAuxNoFPNHead(nn.Module):

    def __init__(
        self,
        in_channels_list: int,
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

        nh, nw = (h // patch_size),  (w // patch_size)
        mh, mw = (h // 16),  (w // 16)

        self.pos_emb_low = nn.Parameter(torch.Tensor(nh * nw * 4, 1, d_model))
        self.pos_emb_high = nn.Parameter(torch.Tensor(mh * mw, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb_low, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_high, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer_low = nn.Sequential(
            DT.SeparableConvBlock(
                in_channels_list[0], d_model, patch_size, 4, 2),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_low = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.tokenizer_high = nn.Sequential(
            DT.SeparableConvBlock(
                in_channels_list[4], d_model, 1, 1, 0),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )
        self.decoder_high = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
            ),
            num_layers=num_layers,
        )

        self.point_reg = nn.Linear(d_model, num_points)
        self.box_reg = nn.Linear(d_model, 4)
        self.edge_reg = nn.Sequential(
            nn.Conv2d(in_channels_list[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        h_lavel_feat = self.tokenizer_high(xs[4])
        pos_emb_high = self.pos_emb_high.expand(-1, h_lavel_feat.size(1), -1)
        h_lavel_feat = h_lavel_feat + pos_emb_high

        query = self.cls_token.expand(-1, h_lavel_feat.size(1), -1)
        query = self.decoder_high(query, h_lavel_feat)
        query = self.norm1(query)  # t, b, d
        aux_points = self.point_reg(query.transpose(0, 1).squeeze(1))
        aux_boxes = self.box_reg(query.transpose(0, 1).squeeze(1))

        l_level_feat = self.tokenizer_low(xs[0])
        pos_emb_low = self.pos_emb_low.expand(-1, l_level_feat.size(1), -1)
        l_level_feat = l_level_feat + pos_emb_low
        query = self.decoder_low(query, l_level_feat)
        query = self.norm2(query)
        points = self.point_reg(query.transpose(0, 1).squeeze(1))
        boxes = self.box_reg(query.transpose(0, 1).squeeze(1))
        edges = self.edge_reg(xs[0])

        return points, edges, boxes, aux_points, aux_boxes


class ViTDecoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        image_size: Tuple[int, int],
        patch_size: int,
        num_points: int,
        in_c: int = 3,
        **kwargs,
    ):
        super().__init__()
        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)
        nh, nw = (h // patch_size),  (w // patch_size)
        self.pos_emb = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer = nn.Sequential(
            nn.Conv2d(in_c, d_model, patch_size, patch_size),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model),
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                **kwargs,
            ),
            num_layers=num_layers
        )

        self.point_reg = nn.Linear(d_model, num_points)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.tokenizer(x)
        query = self.cls_token.expand(-1, x.size(1), -1)
        x = x + self.pos_emb.expand(-1, x.size(1), -1)
        x = self.decoder(query, x)
        x = self.norm(x).transpose(0, 1).squeeze(1)
        points = self.point_reg(query)
        return points,


class ViT(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        image_size: Tuple[int, int],
        in_c: int,
        patch_size: int,
        dim_feedforward: int,
        **kwargs,
    ):
        super().__init__()
        h, w = image_size
        nh, nw = h // patch_size, w // patch_size
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.Tensor(nh*nw, 1, d_model))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.tokenizer = nn.Sequential(
            DT.SeparableConvBlock(
                in_c,
                d_model,
                patch_size,
                patch_size,
            ),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                norm_first=True,
                dropout=0,
                **kwargs,
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.tokenizer(x)
        x = x + self.pos_emb.expand(-1, x.size(1), -1)
        cls_token = self.cls_token.expand(-1, x.size(1), -1)
        x = torch.cat((cls_token, x), dim=0)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token, img_token = torch.split(x, [1, x.size(0) - 1], dim=0)
        return cls_token, img_token


class ViTPointHeatmapEdgeRegHead(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_points: int,
        up_scale: int,
        patch_size: int,
        **kwargs
    ):
        super().__init__()
        self.up_scale = up_scale
        self.patch_size = patch_size
        self.upscale_img = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.Upsample(scale_factor=up_scale,
                        mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
        )

        self.point_reg = nn.Linear(d_model, n_points)
        self.heatmap_rec = nn.Sequential(
            nn.Conv2d(d_model, 4, 3, padding=1),
            nn.Sigmoid()
        )
        self.edge_rec = nn.Sequential(
            nn.Conv2d(d_model, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        cls_token, img_token = x

        # process img_token from
        _, b, d = img_token.shape
        img_token = img_token.permute(1, 2, 0).reshape(
            b, d, self.patch_size, self.patch_size)
        img_token = self.upscale_img(img_token)

        # img_token = img_token.transpose(0, 1) # t,b,d -> b,t,d
        # img_token = self.upscale_img(img_token) # b,t,d -> b,t,d*scale*scale
        # img_token = img_token.transpose(1, 2) # b,t,d*scale*scale -> b,d*scale*scale,t
        # img_token = img_token.reshape(
        #     b, d, self.up_scale, self.up_scale, self.patch_size, self.patch_size) # b,d*scale*scale,t -> b,d,scale,scale,patch_size,patch_size
        # img_token = img_token.permute(0, 1, 4, 2, 5, 3)
        # img_token = img_token.reshape(
        #     b, d, self.patch_size * self.up_scale, self.patch_size * self.up_scale)
        # img_token = self.norm(img_token)

        # auxillary branch
        heatmap = self.heatmap_rec(img_token)
        edge = self.edge_rec(img_token).squeeze(1)

        # main prediction branch
        cls_token = cls_token.transpose(0, 1).squeeze(1)
        points = self.point_reg(cls_token)

        return points, heatmap, edge


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_c,
        d_model: int,
        num_layers: int,
        image_size: List[int],
        patch_size: int,
        nhead: int,
        **kwargs,
    ) -> None:
        super().__init__()

        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)
        nh, nw = h // patch_size, w // patch_size

        # Tokenizer
        self.tokenizer = nn.Sequential(
            DT.SeparableConvBlock(in_c, d_model, patch_size, patch_size),
            nn.Flatten(2),
            DT.Permute([2, 0, 1]),
            nn.LayerNorm(d_model)
        )

        # Positional Embedding
        self.pos_emb = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        # Decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                dim_feedforward=d_model * 2,
                norm_first=True,
                dropout=0,
                nhead=nhead,
                **kwargs,
            ),
            num_layers=num_layers,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.tensor, feature: torch.tensor):
        x = self.tokenizer(feature)
        x = x + self.pos_emb.expand(-1, x.size(1), -1)
        x = self.decoder(query, x)
        x = self.out_norm(x)
        return x


class PointRegMultiDecoderHead(nn.Module):

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

        # Decoder block
        self.decoder_block = nn.ModuleList([
            DecoderBlock(
                in_c=in_channels_list[i],
                d_model=d_model,
                num_layers=num_layers,
                image_size=(h // (2 ** i), w // (2 ** i)),
                patch_size=patch_size,
                nhead=nhead,
                **kwargs,
            ) for i in range(5)
        ])

        # Query
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

        # Head
        self.point_reg = nn.Linear(d_model, num_points)
        self.has_obj = nn.Sequential(
            DT.GAP(),
            nn.Linear(in_channels_list[-1], 1)
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        has_obj = self.has_obj(xs[4])
        query = self.query.expand(-1, xs[0].size(0), -1)
        for i in range(5):
            query = self.decoder_block[4 - i](query, xs[4 - i])
        points = self.point_reg(query.transpose(0, 1).squeeze(1))
        return points, has_obj
