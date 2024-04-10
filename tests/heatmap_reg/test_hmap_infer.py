import numpy as np
import pytest
from docaligner.heatmap_reg.infer import Inference


def create_fake_image(height=256, width=256, channels=3):
    return np.random.rand(height, width, channels)


@pytest.fixture
def mock_onnx_engine(mocker):
    mock_engine = mocker.Mock()
    mock_engine.return_value = {'heatmap': np.random.rand(1, 4, 100, 100)}
    return mock_engine


@pytest.mark.parametrize("model_cfg", ['lcnet050', 'lcnet050_fpn', 'lcnet100', 'lcnet100_fpn', 'mobilenetv2_140', 'fastvit_t8', 'fastvit_sa24'])
def test_inference_initialization(model_cfg):
    inference = Inference(model_cfg=model_cfg)
    assert inference.cfg == Inference.configs[model_cfg]
    assert inference.model is not None


def test_inference_call(mock_onnx_engine):
    inference = Inference()
    inference.model = mock_onnx_engine

    fake_img = create_fake_image()
    result = inference(fake_img)

    assert isinstance(result, np.ndarray)
