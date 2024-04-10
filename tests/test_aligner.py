import docsaidkit as D
import numpy as np
import pytest
from docaligner import (DocAligner, HeatmapRegInference, ModelType,
                        PointRegInference)


def create_fake_image(height=256, width=256, channels=3):
    """創建一個假的圖像數組"""
    return np.random.rand(height, width, channels)


@pytest.fixture
def mock_heatmap_reg_inference(mocker):
    """模擬 HeatmapRegInference"""
    mock_detector = mocker.Mock(spec=HeatmapRegInference)
    mock_detector.return_value = {'heatmap': np.random.rand(1, 4, 100, 100)}
    return mock_detector


@pytest.fixture
def mock_point_reg_inference(mocker):
    """模擬 PointRegInference"""
    mock_detector = mocker.Mock(spec=PointRegInference)
    mock_detector.return_value = {
        'points': np.random.rand(1, 8),
        'has_obj': np.random.rand(1, 1)
    }
    return mock_detector


def test_doc_aligner_initialization_with_invalid_model_cfg():
    """測試無效的 model_cfg"""
    with pytest.raises(ValueError):
        DocAligner(model_type=ModelType.heatmap, model_cfg='invalid_cfg')


def test_doc_aligner_initialization_with_heatmap(mocker, mock_heatmap_reg_inference):
    mocker.patch('docaligner.HeatmapRegInference',
                 new=mock_heatmap_reg_inference)

    doc_aligner = DocAligner(model_type=ModelType.heatmap)
    assert isinstance(doc_aligner.detector, HeatmapRegInference)


def test_doc_aligner_initialization_with_point(mocker, mock_point_reg_inference):
    mocker.patch('docaligner.PointRegInference', new=mock_point_reg_inference)

    doc_aligner = DocAligner(model_type=ModelType.point)
    assert isinstance(doc_aligner.detector, PointRegInference)


def test_doc_aligner_list_models(mocker, mock_heatmap_reg_inference):
    mocker.patch('docaligner.HeatmapRegInference',
                 new=mock_heatmap_reg_inference)
    doc_aligner = DocAligner(model_type=ModelType.heatmap)

    models = doc_aligner.list_models()
    assert isinstance(models, list)


def test_doc_aligner_call(mocker, mock_heatmap_reg_inference):
    mocker.patch('docaligner.HeatmapRegInference',
                 new=mock_heatmap_reg_inference)

    doc_aligner = DocAligner(model_type=ModelType.heatmap)
    fake_img = create_fake_image()
    result = doc_aligner(fake_img)

    assert isinstance(result, D.Document)
    assert result.doc_polygon is None
    assert result.image is fake_img


def test_doc_aligner_repr(mocker, mock_heatmap_reg_inference):
    mocker.patch('docaligner.HeatmapRegInference',
                 new=mock_heatmap_reg_inference)

    doc_aligner = DocAligner(model_type=ModelType.heatmap)
    repr_str = repr(doc_aligner)
    assert isinstance(repr_str, str)
    assert 'Inference' in repr_str
