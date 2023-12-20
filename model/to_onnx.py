from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple, Union

import docsaidkit as D
import docsaidkit.torch as DT
import lightning as L
import torch
import torch.nn as nn

from . import dataset as ds
from . import network as net

DIR = D.get_curdir(__file__)


class Identity(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class WarpLC100FPN(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, img: torch.Tensor):
        return self.head(self.neck(self.backbone(img)))


class WarpLC100Heatmap(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head.heatmap_rec

    def forward(self, img: torch.Tensor):
        return self.head(self.neck(self.backbone(img))[0]).squeeze(1)


class WarpLC50FPNEncoder(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head_encoder = model.head.encoder
        self.head_cls = model.head.cls_token
        self.head_point_reg = model.head.point_reg
        self.norm1 = model.head.norm1
        self.norm2 = model.head.norm2

    def forward(self, img: torch.Tensor):
        x = self.backbone(img)
        x = self.neck(x)
        cls_token, *_ = self.head_cls(x[4])
        cls_token = self.norm1(cls_token)
        hid, *_ = self.head_encoder(x[0], cls_token=cls_token.unsqueeze(1))
        hid = self.norm2(hid)
        points = self.head_point_reg(hid)
        return points


class WarpLC50FPNDecoder(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.tokenizer_high = model.head.tokenizer_high
        self.tokenizer_low = model.head.tokenizer_low
        self.pos_emb_high = model.head.pos_emb_high
        self.pos_emb_low = model.head.pos_emb_low
        self.cls_token = model.head.cls_token
        self.decoder_high = model.head.decoder_high
        self.decoder_low = model.head.decoder_low
        self.point_reg = model.head.point_reg
        self.norm1 = model.head.norm1
        self.norm2 = model.head.norm2

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        xs = self.neck(xs)

        h_lavel_feat = self.tokenizer_high(xs[4])
        pos_emb_high = self.pos_emb_high.expand(-1, h_lavel_feat.size(1), -1)
        h_lavel_feat = h_lavel_feat + pos_emb_high
        query = self.cls_token.expand(-1, h_lavel_feat.size(1), -1)
        query = self.decoder_high(query, h_lavel_feat)
        query = self.norm1(query)
        l_level_feat = self.tokenizer_low(xs[0])
        pos_emb_low = self.pos_emb_low.expand(-1, l_level_feat.size(1), -1)
        l_level_feat = l_level_feat + pos_emb_low
        query = self.decoder_low(query, l_level_feat)
        query = self.norm2(query)
        points = self.point_reg(query.transpose(0, 1).squeeze(1))

        return points


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

    meta_data = DT.get_meta_info(macs, params)
    meta_data.update({
        'InputInfo': repr({k: v for k, v in cfg.onnx.input_shape.items()})
    })

    # meta_data.update({
    #     'Name': 'OCR_00',
    #     'Version': 'V3.0',
    # })

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
