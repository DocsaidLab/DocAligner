from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple, Union

import docsaidkit as D
import docsaidkit.torch as DT
import lightning as L
import torch
import torch.nn as nn
from timm.utils.model import reparameterize_model

from . import dataset as ds
from . import network as net

DIR = D.get_curdir(__file__)


class Identity(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class WarpPointReg(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, img: torch.Tensor):
        points, has_obj = self.head(self.neck(self.backbone(img)))
        return points, has_obj.sigmoid()


class WarpHeatmapReg(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.heatmap_reg = model.head.heatmap_reg

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        xs = self.neck(xs)
        heatmap = self.heatmap_reg(xs[0]).squeeze(1)
        return heatmap


class WarpHeatmapThickReg(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.heatmap_reg = model.head.heatmap_reg
        self.thickness = model.head.thickness

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        xs = self.neck(xs)
        heatmap = self.heatmap_reg(xs[0]).squeeze(1)
        thickness = self.thickness(xs[4]).sigmoid()
        return heatmap, thickness


TORCH_TYPE_LOOKUP = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'uint8': torch.uint8,
    'bool': torch.bool,
    'complex32': torch.complex32,
    'complex64': torch.complex64,
    'complex128': torch.complex128,
}


def input_constructor(xs: Tuple[Dict[str, Any]]):
    return {
        k: torch.zeros(*v['shape'], dtype=TORCH_TYPE_LOOKUP[v['dtype']])
        for k, v in xs
    }


def convert_numeric_keys(input_dict):
    def try_int(key):
        # Try to convert key to an integer, return original if not possible
        try:
            return int(key)
        except ValueError:
            return key

    return {
        try_int(k): {
            try_int(inner_k): v for inner_k, v in inner_dict.items()
        } for k, inner_dict in input_dict.items()
    }


def main_docaligner_torch2onnx(cfg_name: Union[str, Path]):
    model, cfg = DT.load_model_from_config(
        cfg_name, root=DIR, stem='config', network=net)
    model = model.eval().cpu()

    warp_model_name = cfg.onnx.pop('name')
    warp_model = globals()[warp_model_name](model)
    warp_model = reparameterize_model(warp_model)  # For fastvit
    dummy_input = input_constructor(tuple(cfg.onnx.input_shape.items()))

    if dynamic_axes := getattr(cfg.onnx, "dynamic_axes", None):
        dynamic_axes = convert_numeric_keys(dynamic_axes)

    export_name = DIR / f"{cfg_name.lower()}_{D.now(fmt='%Y%m%d')}_fp32"

    torch.onnx.export(
        warp_model, tuple(dummy_input.values()), str(export_name) + '.onnx',
        input_names=cfg.onnx.input_names,
        output_names=cfg.onnx.output_names,
        dynamic_axes=dynamic_axes,
        **cfg.onnx.options
    )

    # To torchscript
    scripted_model = torch.jit.trace(
        warp_model, example_kwarg_inputs=dummy_input)
    torch.jit.save(scripted_model, str(export_name) + '.pt')

    macs, params = DT.get_model_complexity_info(
        warp_model,
        tuple(cfg.onnx.input_shape.items()),
        input_constructor=input_constructor,
        print_per_layer_stat=False,
        as_strings=False
    )

    additional_meta_info = getattr(cfg.onnx, 'additional_meta_info', {})
    meta_data = DT.get_meta_info(macs, params)
    meta_data.update({
        'InputInfo': repr({k: v for k, v in cfg.onnx.input_shape.items()}),
        **additional_meta_info
    })

    pprint(meta_data)

    D.write_metadata_into_onnx(
        onnx_path=str(export_name) + '.onnx',
        out_path=str(export_name) + '.onnx',
        drop_old_meta=False,
        **meta_data
    )

    # Quantize
    if cfg.quantize.get('do_quant', False):
        quant_fpath = D.quantize(
            str(export_name) + '.onnx',
            calibration_data_reader=QATDataset(cfg),
            dst_device=cfg.quantize.device,
            **cfg.quantize.options
        )
        Path(quant_fpath).rename(quant_fpath.replace('__', '_'))


class QATDataset:

    def __init__(self, cfg: dict, length_of_dataset=300) -> None:
        ds_train_name, ds_train_opts = cfg.dataset.train_options.values()
        ds_train_opts.update({'length_of_dataset': length_of_dataset})
        ds_train = getattr(ds, ds_train_name)(**ds_train_opts)
        keys = list(cfg.onnx.input_shape.keys())
        self.enum_data_dicts = iter([
            {
                keys[0]: ds_train[i][0][None]
            } for i in range(length_of_dataset)
        ])

    def get_next(self):
        return next(self.enum_data_dicts, None)
