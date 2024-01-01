from enum import Enum

import docsaidkit as D
import numpy as np

from .heatmap_reg import Inference as HeatmapRegInference
from .point_reg import Inference as PointRecInference

__all__ = ['DocAligner', 'ModelType']


class ModelType(D.EnumCheckMixin, Enum):
    heatmap = 1
    point = 2


class DocAligner:

    def __init__(
        self,
        model_type: ModelType = ModelType.heatmap,
        model_cfg: str = 'mobilenetv2_140',
        backend: D.Backend = D.Backend.cpu,
        gpu_id: int = 0,
        **kwargs
    ):
        model_type = ModelType.obj_to_enum(model_type)
        if model_type == ModelType.heatmap:
            self.detector = HeatmapRegInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                **kwargs
            )
        elif model_type == ModelType.point:
            self.detector = PointRecInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                **kwargs
            )

    def list_models(self) -> list:
        return list(self.detector.configs.keys())

    def __call__(self, img: np.ndarray, do_center_crop: bool = False) -> D.Document:
        polygon = self.detector(img, do_center_crop)
        return D.Document(**{
            'image': img,
            'doc_polygon': polygon if len(polygon) == 4 else None,
        })

    def __repr__(self) -> str:
        return f'{self.detector.__class__.__name__}({self.detector.model})'
