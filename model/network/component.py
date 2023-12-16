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

        self.pos_emb_low = nn.Parameter(torch.Tensor(nh * nw, 1, d_model))
        self.pos_emb_high = nn.Parameter(torch.Tensor(8 * 8, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.pos_emb_low, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_high, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        self.tokenizer_low = nn.Sequential(
            nn.Conv2d(in_channels_list[0], d_model, patch_size, patch_size),
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
            nn.Conv2d(in_channels_list[4], d_model, 1, 1, 0),
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

        self.point_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_points)
        )

        self.box_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4)
        )

        self.edge_reg = nn.Sequential(
            nn.Conv2d(in_channels_list[0], 1, 3, padding=1),
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
