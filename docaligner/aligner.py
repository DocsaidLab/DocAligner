from enum import Enum
from typing import Union

import capybara as cb
import numpy as np

from .heatmap_reg import Inference as HeatmapRegInference
from .point_reg import Inference as PointRegInference

__all__ = [
    'DocAligner', 'ModelType', 'HeatmapRegInference', 'PointRegInference',
]


class ModelType(cb.EnumCheckMixin, Enum):
    heatmap = 1
    point = 2


class DocAligner:

    def __init__(
        self,
        model_type: ModelType = ModelType.heatmap,
        model_cfg: str = None,
        backend: cb.Backend = cb.Backend.cpu,
        gpu_id: int = 0,
        **kwargs
    ):
        model_type = ModelType.obj_to_enum(model_type)
        if model_type == ModelType.heatmap:
            model_cfg = 'fastvit_sa24' if model_cfg is None else model_cfg
            valid_model_cfgs = list(HeatmapRegInference.configs.keys())
            if model_cfg not in valid_model_cfgs:
                raise ValueError(
                    f'Invalid model_cfg: {model_cfg}, '
                    f'valid model_cfgs: {valid_model_cfgs}'
                )
            self.detector = HeatmapRegInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                **kwargs
            )
        elif model_type == ModelType.point:
            model_cfg = 'lcnet050' if model_cfg is None else model_cfg
            valid_model_cfgs = list(PointRegInference.configs.keys())
            if model_cfg not in valid_model_cfgs:
                raise ValueError(
                    f'Invalid model_cfg: {model_cfg}, '
                    f'valid model_cfgs: {valid_model_cfgs}'
                )
            self.detector = PointRegInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                **kwargs
            )

    def list_models(self) -> list:
        return list(self.detector.configs.keys())

    def render(self, *args, **kwargs):
        return self.__call__(*args, **kwargs).gen_doc_info_image()

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = False,
    ) -> Union[np.ndarray]:
        return self.detector(img, do_center_crop)

    def __repr__(self) -> str:
        return f'{self.detector.__class__.__name__}({self.detector.model})'
