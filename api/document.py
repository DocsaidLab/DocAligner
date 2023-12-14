import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import docsaidkit as D
import numpy as np

__all__ = ['Document']


def calc_angle(v1, v2):

    angle = np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)))
    angle = np.degrees(angle)

    v1 = np.array([*v1, 0])
    v2 = np.array([*v2, 0])
    if np.cross(v1, v2)[-1] < 0:
        angle = 360 - angle

    return angle


def poly_angle(
    poly1: D.Polygon,
    poly2: D.Polygon = None,
    base_vector: Tuple[int, int] = (0, 1)
) -> float:

    def _get_angle(poly):
        poly = poly.numpy()
        _v1 = poly[2] - poly[0]
        _v2 = poly[3] - poly[1]
        return _v1 + _v2

    v1 = _get_angle(poly1)
    v2 = _get_angle(poly2) if poly2 is not None else np.array(
        base_vector, dtype='float32')

    return calc_angle(v1, v2)


@dataclass
class Document(D.DataclassCopyMixin, D.DataclassToJsonMixin):

    image: Optional[np.ndarray] = field(default=None)
    polygon: Optional[np.ndarray] = field(default=None)

    @property
    def has_polygon(self):
        return False if self.polygon is None else True

    def be_jsonable(self, exclude_image: bool = True) -> dict:
        if exclude_image:
            img = self.pop('image')
        out = super().be_jsonable()
        if exclude_image:
            self.image = img

        if self.doc_info is not None:
            self.doc_info = self.doc_info.be_jsonable(
                exclude_image=exclude_image)

        return out

    @property
    def flat_img(self):
        return self.gen_flat_img()

    @property
    def angle(self):
        return poly_angle(self.polygon)

    def gen_flat_img(self, image_size: Tuple[int, int] = None):
        if self.has_polygon:
            if image_size is None:
                return D.imwarp_quadrangle(self.image, self.polygon)
            else:
                img_h, img_w = image_size

            point1 = self.polygon.astype('float32')
            point2 = np.array([
                [0, 0],
                [img_w, 0],
                [img_w, img_h],
                [0, img_h]
            ], dtype='float32')
            M = cv2.getPerspectiveTransform(point1, point2)
            flat_img = cv2.warpPerspective(
                self.image, M, (int(img_w), int(img_h)))
            return flat_img
        else:
            warnings.warn(
                'No polygon in the image, returns the original image.')
            return self.image.copy()

    def gen_info_image(self, thickness: int = None) -> np.ndarray:
        colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        export_img = self.image.copy()
        if self.has_polygon:
            _polys = self.polygon.astype(int)
            _polys_roll = np.roll(_polys, 1, axis=0)
            for p1, p2, color in zip(_polys, _polys_roll, colors):
                _thickness = max(int(export_img.shape[1] * 0.005), 1) \
                    if thickness is None else thickness
                export_img = cv2.circle(export_img, p2, radius=_thickness*2,
                                        color=color, thickness=-1, lineType=cv2.LINE_AA)
                export_img = cv2.arrowedLine(export_img, p2, p1, color=color,
                                             thickness=_thickness, line_type=cv2.LINE_AA)
        return export_img

    def draw(self, folder: str = None, name: str = None, **kwargs) -> np.ndarray:
        if folder is None:
            folder = '.'
        folder = D.Path(folder)
        if name is None:
            name = f'output_{D.now()}.jpg'
        D.imwrite(self.gen_info_image(**kwargs), folder / name)
