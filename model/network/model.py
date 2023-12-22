from pathlib import Path
from typing import Any, Dict, List

import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate

from .component import *


class DocAlignerModel(DT.BaseMixin, L.LightningModule):

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
        self.loss_fn = DT.WeightedAWingLoss()
        self.loss_point = nn.SmoothL1Loss(beta=1)

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, _, polys, edges, edges_masks, hmaps, hmaps_masks = batch
        pred_points, pred_hmaps, pred_edges = self.forward(imgs)
        pred_points = pred_points.reshape(-1, 4, 2)

        loss_hmaps = self.loss_fn(pred_hmaps, hmaps, hmaps_masks)
        loss_edge = self.loss_fn(pred_edges, edges, edges_masks)
        loss_points = self.loss_point(pred_points, polys)
        loss = loss_points * 100 + loss_edge + loss_hmaps

        if batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, polys, pred_points,
                         hmaps, pred_hmaps, edges, pred_edges)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'l_point': loss_points,
                'l_edge': loss_edge,
                'l_hmaps': loss_hmaps,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, polys, doc_types = batch
        preds, *_ = self.forward(imgs)
        pred_polys = preds.reshape(-1, 4, 2)

        mask_ious = []
        for pred, gt in zip(pred_polys, polys):
            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
            pred = D.Polygon(pred, normalized=True) \
                .denormalize(1920, 1080).numpy()
            pred = D.order_points_clockwise(pred)
            gt = D.Polygon(gt, normalized=True) \
                .denormalize(1920, 1080).numpy()
            gt = D.order_points_clockwise(gt)
            mask_ious.append(self.mask_iou(pred, gt))

        self.validation_step_outputs.append((mask_ious, doc_types))

    def on_validation_epoch_end(self):
        mask_ious, doc_types = [], []
        for _mask_ious, _doc_types in self.validation_step_outputs:
            mask_ious.extend(_mask_ious)
            doc_types.extend(_doc_types)
        mask_ious = np.array(mask_ious)

        df = pd.DataFrame({
            'DocType': doc_types,
            'IoU': mask_ious,
        })

        grp_question = df.groupby(by='DocType', group_keys=False)

        n_ds = grp_question['DocType'].count() \
            .reset_index(name='Number') \
            .set_index('DocType')

        iou = grp_question['IoU'].mean() \
            .reset_index(name='IoU') \
            .set_index('DocType')

        overall = pd.DataFrame({
            'Number': [len(mask_ious)],
            'IoU': [mask_ious.mean()],
        }, index=['Overall'])

        df = pd.concat([n_ds, iou], axis=1)
        df = pd.concat([df, overall], axis=0)

        print('\n')
        print(tabulate(df.T, headers='keys', tablefmt='psql',
              numalign='right', stralign='right', floatfmt='.4f', intfmt='d'))
        print('\n')

        self.log('val_iou', mask_ious.mean(), sync_dist=True)
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

    def preview(self, batch_idx, imgs, polys, pred_polys, hmaps, preds_hmaps,
                edges, preds_edges, suffix='train'):

        # setup preview dir
        preview_dir = self.preview_dir / f'{suffix}_batch_{batch_idx}'
        if not preview_dir.exists():
            preview_dir.mkdir(parents=True)

        imgs = imgs.detach().cpu().numpy()
        polys = polys.detach().cpu().numpy()
        pred_polys = pred_polys.reshape(-1, 4, 2).detach().cpu().numpy()
        hmaps = hmaps.detach().cpu().numpy()
        preds_hmaps = preds_hmaps.detach().cpu().numpy()
        edges = edges.detach().cpu().numpy()
        preds_edges = preds_edges.detach().cpu().numpy()

        for idx, (img, poly, pred_poly, hmap, pred_hmap, edge, preds_edge) in \
                enumerate(zip(imgs, polys, pred_polys, hmaps, preds_hmaps, edges, preds_edges)):
            img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)

            heatmap_gt = np.zeros_like(img)
            colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255),
                      (0, 255, 255), (255, 0, 255), (255, 0, 0)]
            for _hmap, color in zip(hmap, colors):
                _hmap = np.uint8(255 * _hmap)
                _hmap = D.imresize(_hmap, size=(img.shape[0], img.shape[1]))
                _hmap = cv2.applyColorMap(_hmap, cv2.COLORMAP_BONE)
                _hmap[..., np.argwhere(np.array(color) == 0)] = 0
                heatmap_gt = cv2.addWeighted(heatmap_gt, 1, _hmap, 1, gamma=0)

            edge = np.uint8(edge * 255)
            edge = D.imresize(edge, size=(img.shape[0], img.shape[1]))
            edge = cv2.applyColorMap(edge, cv2.COLORMAP_JET)
            img_output1 = cv2.addWeighted(heatmap_gt, 1, edge, 0.5, gamma=0)

            heatmap_pred = np.zeros_like(img)
            for _hmap, color in zip(pred_hmap, colors):
                _hmap = np.uint8(255 * _hmap)
                _hmap = D.imresize(_hmap, size=(img.shape[0], img.shape[1]))
                _hmap = cv2.applyColorMap(_hmap, cv2.COLORMAP_BONE)
                _hmap[..., np.argwhere(np.array(color) == 0)] = 0
                heatmap_pred = cv2.addWeighted(
                    heatmap_pred, 1, _hmap, 1, gamma=0)

            preds_edge = np.uint8(preds_edge * 255)
            preds_edge = D.imresize(
                preds_edge, size=(img.shape[0], img.shape[1]))
            preds_edge = cv2.applyColorMap(preds_edge, cv2.COLORMAP_JET)
            img_output2 = cv2.addWeighted(
                heatmap_pred, 1, preds_edge, 0.5, gamma=0)

            poly = D.Polygon(poly, normalized=True).denormalize(
                *img.shape[:2][::-1])
            pred_poly = D.Polygon(pred_poly, normalized=True).denormalize(
                *img.shape[:2][::-1])
            img_poly1 = D.draw_polygon(
                img.copy(), poly, color=(0, 255, 0), thickness=2)
            img_poly2 = D.draw_polygon(
                img.copy(), pred_poly, color=(0, 0, 255), thickness=2)
            img_poly = np.concatenate([img, img_poly1, img_poly2], axis=1)

            img_output = np.concatenate(
                [img, img_output1, img_output2], axis=1)
            img_output = np.concatenate([img_poly, img_output], axis=0)
            img_output_name = str(preview_dir / f'{idx}.jpg')
            D.imwrite(img_output, img_output_name)
