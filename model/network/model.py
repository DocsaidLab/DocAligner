from pathlib import Path
from typing import Any, Dict, List

import docsaidkit as dsk
import docsaidkit.torch as D
import lightning as L
import numpy as np
import torch
import torch.nn as nn

from .backbone import Backbone
from .base import BaseMixin


class DocAlignedHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int = 8,
    ) -> None:
        super().__init__()
        self.point_regression = nn.Sequential(
            D.GAP(),
            nn.Linear(in_channels, num_points)
        )
        self.edge_regression = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.point_regression(x[-1]), self.edge_regression(x[0])


class DocAlignedModel(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg['common']['preview_batch']
        self.apply_solver_config(cfg['optimizer'], cfg['lr_scheduler'])

        # Setup model
        cfg_model = cfg['model']
        self.backbone = Backbone(**cfg_model['backbone'])
        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            channels = [i.size(1) for i in self.backbone(dummy)]
        cfg_model['neck'].update({'in_channels_list': channels})
        self.neck = D.build_neck(**cfg_model['neck'])
        self.head = globals()[cfg_model['head']['name']](**cfg_model['head']['options'])

        # Setup loss function
        self.loss_fn_edge = D.WeightedAWingLoss()
        self.loss_fn_point = nn.SmoothL1Loss(beta=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, polys, edges, edge_masks = batch
        pred_polys, pred_edges = self.forward(imgs)
        loss_points = self.loss_fn_point(pred_polys.reshape(-1), polys.reshape(-1))
        loss_edges = self.loss_fn_edge(pred_edges.squeeze(1), edges, edge_masks)
        loss = loss_points + loss_edges

        if batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, polys, edges, pred_polys, pred_edges)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss_points': loss_points,
                'loss_edges': loss_edges,
                'loss': loss,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        img, poly, edge, edge_mask = batch
        pred_points, pred_edges = self.forward(img)

    @ property
    def preview_dir(self):
        dir_path = Path().joinpath(self.cfg.name, self.cfg.name_ind)
        img_path = Path(dir_path) / "preview" / f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path

    def preview(self, batch_idx, imgs, polys, edges, pred_polys, pred_edges):
        preview_dir = self.preview_dir / f'batch_{batch_idx}'
        if not preview_dir.exists():
            preview_dir.mkdir(parents=True)
        imgs = imgs.detach().cpu().numpy()
        polys = polys.detach().cpu().numpy()
        edges = edges.detach().cpu().numpy()
        pred_polys = pred_polys.reshape(-1, 4, 2).detach().cpu().numpy()
        pred_edges = pred_edges.squeeze(1).detach().cpu().numpy()

        for idx, (img, poly, edge, pred_poly, pred_edge) in \
                enumerate(zip(imgs, polys, edges, pred_polys, pred_edges)):
            img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)
            edge = np.stack([np.uint8(edge * 255)] * 3, axis=-1)
            pred_edge = np.stack([np.uint8(pred_edge * 255)] * 3, axis=-1)
            poly = dsk.Polygon(poly, normalized=True).denormalize(*img.shape[:2][::-1])
            pred_poly = dsk.Polygon(pred_poly, normalized=True).denormalize(*img.shape[:2][::-1])

            img1 = dsk.draw_polygon(img.copy(), poly, color=(0, 255, 0), thickness=2)
            img2 = dsk.draw_polygon(img.copy(), pred_poly, color=(0, 0, 255), thickness=2)
            img_poly = np.concatenate([img1, img2], axis=1)
            img_edge = np.concatenate([edge, pred_edge], axis=1)
            img_output = np.concatenate([img_poly, img_edge], axis=0)
            img_output_name = str(preview_dir / f'{idx}.jpg')
            dsk.imwrite(img_output, img_output_name)
