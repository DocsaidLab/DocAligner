import numpy as np
from docaligner.heatmap_reg.infer import postprocess


def create_fake_preds(num_classes=4, h=100, w=100):
    """ 創建假的預測數據 """
    return np.random.rand(1, num_classes, h, w)


def test_postprocess_with_different_thresholds():
    preds = create_fake_preds()
    img_size = (100, 100)

    # 測試不同的 heatmap_threshold 值
    for threshold in [0.1, 0.3, 0.5]:
        result = postprocess(preds, img_size, heatmap_threshold=threshold)
        assert isinstance(result, list)


def test_postprocess_with_different_img_sizes():
    preds = create_fake_preds()
    result = postprocess(preds, (100, 100))
    assert isinstance(result, list)


def test_postprocess_output_structure():
    preds = create_fake_preds()
    img_size = (100, 100)

    result = postprocess(preds, img_size)
    assert isinstance(result, list)
    # 每個項目應該是一個長度為 2 的點列表
    for point in result:
        assert isinstance(point, list)
        assert len(point) == 2

    assert len(result) <= preds.shape[1]
