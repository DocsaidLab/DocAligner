from pathlib import Path
from typing import Any, Dict, List

import docsaidkit as D
import docsaidkit.torch as DT
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate

from .component import *


class PointModel(DT.BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg['common']['preview_batch']
        self.apply_solver_config(cfg['optimizer'], cfg['lr_scheduler'])

        # Setup model
        cfg_model = cfg['model']
        self.backbone = DT.Identity()
        self.neck = DT.Identity()
        self.head = DT.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model['backbone']['name']](
                **cfg_model['backbone']['options'])

            channels = []
            if cfg_model['backbone']['name'] == 'Backbone':
                with torch.no_grad():
                    dummy = torch.rand(1, 3, 128, 128)
                    channels = [i.size(1) for i in self.backbone(dummy)]

        if hasattr(cfg_model, 'neck'):
            cfg_model['neck'].update({'in_channels_list': channels})
            self.neck = DT.build_neck(**cfg_model['neck'])

        if hasattr(cfg_model, 'head'):
            cfg_model['head']['options'].update({'in_channels_list': channels})
            self.head = globals()[cfg_model['head']['name']](
                **cfg_model['head']['options'])

        # Setup loss function
        self.loss_point = nn.SmoothL1Loss(beta=0.1)
        self.loss_obj = nn.BCEWithLogitsLoss()

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, _, polys, _, _, _, _, has_objs = batch
        pred_points, pred_has_objs = self.forward(imgs)

        loss_obj = self.loss_obj(pred_has_objs, has_objs)
        has_objs_idx = has_objs.to(torch.bool).squeeze(1)
        pred_points = pred_points.reshape(-1, 4, 2)
        loss_points = self.loss_point(
            pred_points[has_objs_idx], polys[has_objs_idx])
        loss = 1000 * loss_points + loss_obj

        if batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, polys, pred_points)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                'l_p': loss_points,
                'l_ob': loss_obj,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, polys, doc_types = batch
        pred_polys, *_ = self.forward(imgs)
        pred_polys = pred_polys.reshape(-1, 4, 2)

        polys = polys.detach().cpu().numpy()
        pred_polys = pred_polys.detach().cpu().numpy()

        ious = []
        for gt, _polys in zip(polys, pred_polys):

            # Fixed on SmartDoc dataset
            h, w = 1080, 1920

            gt = D.Polygon(gt, normalized=True).denormalize(w, h).numpy()

            # pred from polygons
            pred_polygons_from_points = D.Polygon(
                _polys, normalized=True).denormalize(w, h).numpy()
            ious_point = self.mask_iou(pred_polygons_from_points, gt)

            ious.append(ious_point)

        self.validation_step_outputs.append((ious, doc_types))

    def on_validation_epoch_end(self):
        point_ious, doc_types = [], []
        for _ious, _doc_types in self.validation_step_outputs:
            point_ious.extend(_ious)
            doc_types.extend(_doc_types)
        point_ious = np.array(point_ious)

        df = pd.DataFrame({
            'DocType': doc_types,
            'PointIoU': point_ious,
        })

        grp_question = df.groupby(by='DocType', group_keys=False)

        n_ds = grp_question['DocType'].count() \
            .reset_index(name='Number') \
            .set_index('DocType')

        df_point_iou = grp_question['PointIoU'].mean() \
            .reset_index(name='PointIoU') \
            .set_index('DocType')

        overall = pd.DataFrame({
            'Number': [len(point_ious)],
            'PointIoU': [point_ious.mean()],
        }, index=['Overall'])

        df = pd.concat([n_ds, df_point_iou], axis=1)
        df = pd.concat([df, overall], axis=0)

        print('\n')
        print(tabulate(df.T, headers='keys', tablefmt='psql',
              numalign='right', stralign='right', floatfmt='.4f', intfmt='d'))
        print('\n')

        self.log('val_iou', point_ious.mean(), sync_dist=True)
        self.validation_step_outputs.clear()

    def mask_iou(self, pred_poly: np.ndarray, gt_poly: np.ndarray):
        # 設定 SmartDoc 資料集影像的尺寸
        doc_h, doc_w = 2970, 2100
        return D.jaccard_index(pred_poly, gt_poly, (doc_h, doc_w))

    @ property
    def preview_dir(self):
        img_path = Path(self.cfg.root_dir) / "preview" / \
            f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path

    def preview(self, batch_idx, imgs, polys, pred_polys, suffix='train'):

        # setup preview dir
        preview_dir = self.preview_dir / f'{suffix}_batch_{batch_idx}'
        if not preview_dir.exists():
            preview_dir.mkdir(parents=True)

        imgs = imgs.detach().cpu().numpy()
        polys = polys.detach().cpu().numpy()
        pred_polys = pred_polys.reshape(-1, 4, 2).detach().cpu().numpy()

        for idx, (img, poly, pred_poly) in \
                enumerate(zip(imgs, polys, pred_polys)):
            img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)

            poly = D.Polygon(poly, normalized=True).denormalize(
                *img.shape[:2][::-1])
            pred_poly = D.Polygon(pred_poly, normalized=True).denormalize(
                *img.shape[:2][::-1])
            img_poly1 = D.draw_polygon(
                img.copy(), poly, color=(0, 255, 0), thickness=2)
            img_poly2 = D.draw_polygon(
                img.copy(), pred_poly, color=(0, 0, 255), thickness=2)
            img_output = np.concatenate([img, img_poly1, img_poly2], axis=1)

            img_output_name = str(preview_dir / f'{idx}.jpg')
            D.imwrite(img_output, img_output_name)
