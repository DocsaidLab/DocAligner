from typing import Tuple

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
        img = img[None] / 255

    return {
        'input': {'img': img},
        'img_size_ori': (nh, nw),
        'img_size_infer': img_size_infer,
        'return_tensor': return_tensor,
        'center_crop_align': center_crop_align
    }


def postprocess(
    points: np.ndarray,
    has_obj: bool,
    imgs_size: Tuple[int, int]
) -> np.ndarray:
    if has_obj > 0.5:
        points = points.reshape(4, 2)
        polygon = points * np.array(imgs_size[::-1])
    else:
        polygon = np.array([])
    return polygon


class Inference:

    configs = {
        'lcnet050': {
            'model_path': 'lcnet050_p_multi_decoder_l3_d64_256_fp32.onnx',
            'file_id': 'HkK9WKKWbH5zEsM',
            'img_size_infer': (256, 256),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backend.cpu,
        model_cfg: str = 'lcnet050',
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
            points=x['points'],
            has_obj=x['has_obj'],
            imgs_size=img_infos['img_size_ori'],
        )

        if len(polygon):
            polygon = np.array(polygon) + \
                np.array(img_infos['center_crop_align'])

        return polygon
