import numpy as np
import pytest

from docaligner.point_reg.infer import Inference


def create_fake_image(height=256, width=256, channels=3):
    return np.random.rand(height, width, channels)


@pytest.fixture
def mock_onnx_engine(mocker):
    mock_engine = mocker.Mock()
    mock_engine.return_value = {
        'points': np.random.rand(1, 8),
        'has_obj': np.random.rand(1, 1)
    }
    return mock_engine


@pytest.mark.parametrize("model_cfg", ['lcnet050'])
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
