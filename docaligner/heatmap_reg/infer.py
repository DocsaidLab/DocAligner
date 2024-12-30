from typing import List, Tuple

import capybara as cb
import numpy as np

DIR = cb.get_curdir(__file__)

__all__ = ['Inference']


def preprocess(
    img: np.ndarray,
    img_size_infer: Tuple[int, int] = None,
    do_center_crop: bool = False,
    return_tensor: bool = True,
):
    if not cb.is_numpy_img(img):
        raise ValueError("Input image must be numpy array.")

    h, w = img.shape[0:2]
    center_crop_align = [0, 0]

    if do_center_crop:
        img = cb.centercrop(img)
        if h > w:
            center_crop_align = [0, (h - w) // 2]
        else:
            center_crop_align = [(w - h) // 2, 0]

    nh, nw = img.shape[0:2]
    if img_size_infer is not None:
        img = cb.imresize(img, size=img_size_infer)

    if return_tensor:
        img = np.transpose(img, axes=(2, 0, 1)).astype('float32')
        img = img[None] / 255.

    return {
        'input': {'img': img},
        'img_size_ori': (nh, nw),
        'img_size_infer': img_size_infer,
        'return_tensor': return_tensor,
        'center_crop_align': center_crop_align
    }


def postprocess(
    preds: np.ndarray,  # (1, 4, H, W)
    imgs_size: Tuple[int, int],
    heatmap_threshold: float = 0.3
) -> List[float]:

    def _get_point_with_max_area(mask):
        polygons = cb.Polygons.from_image(mask).drop_empty()
        if len(polygons) > 0:
            polygons = polygons[polygons.area == polygons.area.max()]
        return polygons.centroid.flatten().tolist()

    polygon = []
    for ii, pred in enumerate(preds[0]):
        pred = cb.imresize(pred, size=imgs_size)
        pred[pred < heatmap_threshold] = 0
        pred = np.uint8(pred * 255)
        pred = cb.imbinarize(pred)
        point = _get_point_with_max_area(pred)
        if len(point) == 2 and ii < 4:
            polygon.append(point)

    return polygon


class Inference:

    configs = {
        'lcnet100': {
            'model_path': 'lcnet100_h_e_bifpn_256_fp32.onnx',
            'file_id': 'EXygK5Qn9dyA5Ck',
            'img_size_infer': (256, 256),
        },
        'fastvit_t8': {
            'model_path': 'fastvit_t8_h_e_bifpn_256_fp32.onnx',
            'file_id': 'YdEZCay4eiadHrY',
            'img_size_infer': (256, 256),
        },
        'fastvit_sa24': {
            'model_path': 'fastvit_sa24_h_e_bifpn_256_fp32.onnx',
            'file_id': 'w2ZD9CoK38CayrH',
            'img_size_infer': (256, 256),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backend.cpu,
        model_cfg: str = 'fastvit_sa24',
        **kwargs
    ):
        self.root = DIR / 'ckpt'
        self.cfg = cfg = self.configs[model_cfg]
        self.img_size_infer = cfg['img_size_infer']
        model_path = self.root / cfg['model_path']
        if not cb.Path(model_path).exists():
            cb.download_from_docsaid(
                cfg['file_id'], model_path.name, str(model_path))

        self.model = cb.ONNXEngine(model_path, gpu_id, backend, **kwargs)

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = False,
    ) -> np.ndarray:
        img_infos = preprocess(
            img=img,
            img_size_infer=self.img_size_infer,
            do_center_crop=do_center_crop
        )
        x = self.model(**img_infos['input'])
        polygon = postprocess(
            preds=x['heatmap'],
            imgs_size=img_infos['img_size_ori'],
        )
        polygon = np.array(polygon)

        if len(polygon):
            polygon = polygon + np.array(img_infos['center_crop_align'])

        return polygon.astype(np.float32)
