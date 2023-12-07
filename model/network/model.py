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


class DocAlignedModel(DT.BaseMixin, L.LightningModule):

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
        self.loss_fn_edge = DT.WeightedAWingLoss()
        self.loss_fn_point = nn.SmoothL1Loss(beta=0.1)

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, boxes, polys, edges, edge_masks = batch
        preds, *other_branchs = self.forward(imgs)
        pred_polys = preds.reshape(-1, 4, 2)
        loss = self.loss_fn_point(pred_polys, polys)

        # edge loss
        edge_args = {}
        if len(other_branchs) > 0:
            pred_edges = other_branchs[0].squeeze(1)
            loss_edge = self.loss_fn_edge(
                pred_edges, edges, torch.zeros_like(edges))
            loss += loss_edge
            edge_args.update({
                'edges': edges,
                'pred_edges': pred_edges,
            })

        if batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, polys, pred_polys, **edge_args)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
            },
            prog_bar=True,
            on_step=True,
        )

        # checkout nan loss
        if torch.isnan(loss):
            self.preview(batch_idx, imgs, polys, pred_polys, suffix='NaN')
            raise ValueError('Loss is nan.')

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, polys, doc_types = batch
        preds, *_ = self.forward(imgs)
        pred_polys = preds.reshape(-1, 4, 2)

        mask_ious = []
        for pred, gt in zip(pred_polys, polys):
            mask_ious.append(self.mask_iou(pred, gt))

        self.validation_step_outputs.append((mask_ious, doc_types))

        if batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, polys, pred_polys, suffix='val')

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

    def mask_iou(self, pred_poly, gt_poly):
        h, w = self.cfg.common.image_size
        pred_poly = pred_poly.detach().cpu().numpy()
        gt_poly = gt_poly.detach().cpu().numpy()
        pred_poly = D.Polygon(pred_poly, normalized=True).denormalize(h, w)
        gt_poly = D.Polygon(gt_poly, normalized=True).denormalize(h, w)
        pred_mask = D.draw_polygon(
            np.zeros((h, w), dtype=np.uint8), pred_poly, color=1, fillup=True)
        gt_mask = D.draw_polygon(
            np.zeros((h, w), dtype=np.uint8), gt_poly, color=1, fillup=True)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union

    @ property
    def preview_dir(self):
        img_path = Path(self.cfg.root_dir) / "preview" / \
            f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path

    def preview(self, batch_idx, imgs, polys, pred_polys, edges=None,
                pred_edges=None, suffix='train'):
        preview_dir = self.preview_dir / f'{suffix}_batch_{batch_idx}'
        if not preview_dir.exists():
            preview_dir.mkdir(parents=True)
        imgs = imgs.detach().cpu().numpy()
        polys = polys.detach().cpu().numpy()
        pred_polys = pred_polys.reshape(-1, 4, 2).detach().cpu().numpy()
        edges = edges.detach().cpu().numpy() \
            if edges is not None else [None] * len(imgs)
        pred_edges = pred_edges.detach().cpu().numpy() \
            if pred_edges is not None else [None] * len(imgs)

        for idx, (img, poly, pred_poly, edge, pred_edge) in \
                enumerate(zip(imgs, polys, pred_polys, edges, pred_edges)):
            img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)
            poly = D.Polygon(poly, normalized=True).denormalize(
                *img.shape[:2][::-1])
            pred_poly = D.Polygon(pred_poly, normalized=True).denormalize(
                *img.shape[:2][::-1])

            img1 = D.draw_polygon(
                img.copy(), poly, color=(0, 255, 255), thickness=2)
            img2 = D.draw_polygon(
                img.copy(), pred_poly, color=(255, 0, 255), thickness=2)
            img_output = np.concatenate([img1, img2], axis=1)

            if edge is not None and pred_edge is not None:
                edge = np.stack([np.uint8(edge * 255)] * 3, axis=-1)
                pred_edge = np.stack([np.uint8(pred_edge * 255)] * 3, axis=-1)
                img_edge = np.concatenate([edge, pred_edge], axis=1)
                img_edge = D.imresize(img_edge, (None, img_output.shape[1]))
                img_output = np.concatenate([img_output, img_edge], axis=0)

            img_output_name = str(preview_dir / f'{idx}.jpg')
            D.imwrite(img_output, img_output_name)
