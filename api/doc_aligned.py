from enum import Enum

import docsaidkit as D
import numpy as np

from .document import Document
from .heatmap_rec import Inference as HeatmapRegInference

__all__ = ['DocAligned', 'ModelType']


class ModelType(D.EnumCheckMixin, Enum):
    HeatmapBased = 1
    PointBased = 2


class DocAligned:

    def __init__(
        self,
        gpu_id: int = 0,
        backend: D.Backend = D.Backend.cpu,
        model_type: ModelType = ModelType.HeatmapBased,
        **kwargs
    ):
        model_type = ModelType.obj_to_enum(model_type)
        if model_type == ModelType.HeatmapBased:
            self.detector = HeatmapRegInference(
                gpu_id=gpu_id,
                backend=backend,
                **kwargs
            )

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = True,
    ):
        polygon = self.detector(img, do_center_crop)
        return Document(**{
            'image': img,
            'polygon': polygon,
        })
