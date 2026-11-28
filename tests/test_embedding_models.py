"""Weights-free tests for DINO preprocessing and model adaptation."""

from unittest.mock import Mock

import numpy as np
import pytest

from scenedetect.detectors import _embedding_models as models

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def test_preprocess_normalizes_rgb_and_returns_contiguous_nchw():
    frame = np.empty((128, 256, 3), dtype=np.uint8)
    frame[:] = (17, 89, 203)  # BGR: unequal channels expose channel swaps.

    blob = models._preprocess(frame)

    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous

    expected = (np.array([203, 89, 17], dtype=np.float32) / 255.0 - MEAN) / STD
    np.testing.assert_allclose(
        blob[0],
        np.broadcast_to(expected[:, None, None], (3, 224, 224)),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("portrait", [False, True], ids=["landscape", "portrait"])
def test_preprocess_resizes_short_edge_and_center_crops(portrait):
    frame = np.empty((128, 256, 3), dtype=np.uint8)
    frame[:, :96] = (255, 0, 0)
    frame[:, 96:160] = (0, 255, 0)
    frame[:, 160:] = (0, 0, 255)
    if portrait:
        frame = frame.transpose(1, 0, 2).copy()

    blob = models._preprocess(frame)

    # Distinguish short-edge resizing to 256 from 224, using probes
    # inside solid-color regions rather than interpolation boundaries.
    indices = (16, 52, 172, 208)
    samples = np.stack(
        [blob[0, :, index, 112] if portrait else blob[0, :, 112, index] for index in indices]
    )
    expected_rgb = np.array(
        [[0, 0, 255], [0, 255, 0], [0, 255, 0], [255, 0, 0]],
        dtype=np.float32,
    )
    expected = (expected_rgb / 255.0 - MEAN) / STD

    np.testing.assert_allclose(samples, expected, rtol=1e-6, atol=1e-6)


def test_preprocess_uses_exact_center_crop_coordinates():
    # Short edge is already 256, so resizing preserves the input.
    # The 224x224 center crop must span rows 16..239 and columns 48..271.
    frame = np.zeros((256, 320, 3), dtype=np.uint8)
    frame[16, 48] = (255, 0, 0)
    frame[239, 271] = (0, 0, 255)

    blob = models._preprocess(frame)

    corners = np.stack([blob[0, :, 0, 0], blob[0, :, -1, -1]])
    expected_rgb = np.array(
        [[0, 0, 255], [255, 0, 0]],
        dtype=np.float32,
    )
    expected = (expected_rgb / 255.0 - MEAN) / STD

    np.testing.assert_allclose(corners, expected, rtol=1e-6, atol=1e-6)


def test_dino_loads_once_and_returns_only_the_cls_token(monkeypatch):
    output = np.full((1, 257, 384), -1000.0, dtype=np.float32)
    expected = np.arange(384, dtype=np.float32)
    output[0, 0] = expected

    network = Mock()
    network.forward.return_value = output
    load_model = Mock(return_value=network)
    monkeypatch.setattr(models, "readNetFromONNX", load_model)

    embedder = models.DinoEmbedder("unused.onnx")
    load_model.assert_not_called()

    first = np.zeros((256, 256, 3), dtype=np.uint8)
    second = np.full_like(first, 255)

    np.testing.assert_array_equal(embedder(first), expected)
    np.testing.assert_array_equal(embedder.embed(second), expected)

    load_model.assert_called_once_with("unused.onnx")
    assert network.forward.call_count == 2

    # Strict zip also verifies that exactly two inputs were supplied.
    for call, frame in zip(network.setInput.call_args_list, [first, second], strict=True):
        np.testing.assert_array_equal(call.args[0], models._preprocess(frame))


def test_dino_rejects_an_incompatible_output_layout(monkeypatch):
    network = Mock()
    network.forward.return_value = np.zeros((1, 384), dtype=np.float32)
    monkeypatch.setattr(models, "readNetFromONNX", Mock(return_value=network))
    embedder = models.DinoEmbedder("unused.onnx")

    with pytest.raises(ValueError, match="token output"):
        embedder(np.zeros((256, 256, 3), dtype=np.uint8))
