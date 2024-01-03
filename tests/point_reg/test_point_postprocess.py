import numpy as np
import pytest
from docaligner.point_reg.infer import postprocess


def test_postprocess_with_object():
    points = np.random.rand(8)
    imgs_size = (100, 200)

    polygon = postprocess(points, has_obj=True, imgs_size=imgs_size)
    assert polygon.shape == (4, 2)
    for point in polygon:
        assert 0 <= point[0] <= imgs_size[1]
        assert 0 <= point[1] <= imgs_size[0]


def test_postprocess_without_object():
    points = np.random.rand(8)
    polygon = postprocess(points, has_obj=False, imgs_size=(100, 200))
    assert polygon.size == 0
    assert isinstance(polygon, np.ndarray)


def test_postprocess_invalid_input():
    points = np.random.rand(5)
    with pytest.raises(ValueError):
        postprocess(points, has_obj=True, imgs_size=(100, 200))
