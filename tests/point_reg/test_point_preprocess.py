import numpy as np
import pytest
from docaligner.point_reg.infer import preprocess


def test_preprocess_with_invalid_input():
    # 測試非 NumPy 圖像輸入
    with pytest.raises(ValueError):
        preprocess("not_a_numpy_array")


def test_preprocess_with_center_crop():
    # 創建假的輸入圖像
    img = np.random.rand(200, 100, 3)

    # 開啟 center crop
    result = preprocess(img, do_center_crop=True)
    assert result['center_crop_align'] != [0, 0]

    # 關閉 center crop
    result = preprocess(img, do_center_crop=False)
    assert result['center_crop_align'] == [0, 0]


def test_preprocess_with_img_size_infer():
    img = np.random.rand(100, 100, 3)
    img_size_infer = (50, 50)

    # 提供 img_size_infer
    result = preprocess(img, img_size_infer=img_size_infer)
    assert result['img_size_infer'] == img_size_infer

    # 不提供 img_size_infer
    result = preprocess(img)
    assert result['img_size_infer'] is None


def test_preprocess_with_return_tensor():
    img = np.random.rand(100, 100, 3)

    # 開啟 return tensor
    result = preprocess(img, return_tensor=True)
    assert result['input']['img'].ndim == 4

    # 關閉 return tensor
    result = preprocess(img, return_tensor=False)
    assert result['input']['img'].ndim == 3


def test_preprocess_output_structure():
    img = np.random.rand(100, 100, 3)
    result = preprocess(img)

    assert isinstance(result, dict)
    assert 'input' in result
    assert 'img_size_ori' in result
    assert 'img_size_infer' in result
    assert 'return_tensor' in result
    assert 'center_crop_align' in result
