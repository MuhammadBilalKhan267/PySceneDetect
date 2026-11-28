"""Weights-free tests for embedding detector behavior."""

from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest

from scenedetect.common import FramePosition, Timecode
from scenedetect.detectors.embedding_detector import (
    EmbeddingDetector,
    cosine_distance,
)


def _position(frame_number, pts=None):
    return FramePosition(
        timecode=Timecode(
            pts=frame_number if pts is None else pts,
            time_base=Fraction(1, 10),
        ),
        frame_number=frame_number,
    )


def _frame(value=0):
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_first_frame_never_produces_a_cut():
    distance = Mock(return_value=0.0)
    detector = EmbeddingDetector(
        threshold=0.0,
        min_scene_len=0,
        embedder=lambda _: np.ones(1),
        distance=distance,
    )

    assert detector.process_frame(_position(100), _frame()) == []
    distance.assert_not_called()


@pytest.mark.parametrize("start", [0, 100])
def test_suppressed_candidates_update_previous_embedding(start):
    embeddings = np.eye(2, dtype=np.float32)
    embedder = Mock(side_effect=lambda frame: embeddings[int(frame[0, 0, 0])])
    detector = EmbeddingDetector(
        threshold=1.0,
        min_scene_len=3,
        embedder=embedder,
    )
    sequence = [0, 1, 1, 1, 0, 1, 1, 0]  # A B B B A B B A
    positions = [_position(start + index) for index in range(len(sequence))]

    cuts = []
    for position, value in zip(positions, sequence, strict=True):
        cuts.extend(detector.process_frame(position, _frame(value)))

    assert cuts == [positions[4], positions[7]]
    assert embedder.call_count == len(sequence)


@pytest.mark.parametrize(
    "score, expect_cut",
    [(0.99, False), (1.0, True), (1.01, True)],
)
def test_threshold_is_inclusive(score, expect_cut):
    detector = EmbeddingDetector(
        threshold=1.0,
        min_scene_len=1,
        embedder=lambda _: np.ones(1),
        distance=lambda previous, current: score,
    )
    first, second = _position(100), _position(101)
    detector.process_frame(first, _frame())

    assert detector.process_frame(second, _frame()) == ([second] if expect_cut else [])


@pytest.mark.parametrize("form", ["function", "instance", "method"])
def test_accepts_callable_embedders_and_custom_distance(form):
    class FakeEmbedder:
        def embed(self, frame):
            # Deliberately not a valid cosine-helper input.
            return np.array([[frame[0, 0, 0]]], dtype=np.float32)

        def __call__(self, frame):
            return self.embed(frame)

    fake = FakeEmbedder()
    embedder = {
        "function": lambda frame: fake.embed(frame),
        "instance": fake,
        "method": fake.embed,
    }[form]
    detector = EmbeddingDetector(
        threshold=0.5,
        min_scene_len=1,
        embedder=embedder,
        distance=lambda previous, current: float(current.item() - previous.item()),
    )
    first, second = _position(100), _position(101)

    assert detector.process_frame(first, _frame(0)) == []
    assert detector.process_frame(second, _frame(1)) == [second]


@pytest.mark.parametrize(
    "minimum, expected_indices",
    [
        pytest.param(3, [3], id="frames"),
        pytest.param(0.5, [1], id="seconds"),
    ],
)
def test_spacing_preserves_frame_and_time_units(minimum, expected_indices):
    positions = [
        _position(100 + index, pts=pts) for index, pts in enumerate([1000, 1005, 1006, 1007])
    ]
    detector = EmbeddingDetector(
        threshold=1.0,
        min_scene_len=minimum,
        embedder=lambda _: np.ones(1),
        distance=lambda previous, current: 1.0,
    )

    cuts = []
    for position in positions:
        cuts.extend(detector.process_frame(position, _frame()))

    assert cuts == [positions[index] for index in expected_indices]


@pytest.mark.parametrize(
    "other, expected",
    [
        ([3.0, 0.0], 0.0),
        ([0.0, 5.0], 1.0),
        ([-7.0, 0.0], 2.0),
    ],
)
def test_cosine_distance_normalizes_vectors(other, expected):
    first = np.array([2.0, 0.0], dtype=np.float32)
    second = np.array(other, dtype=np.float32)

    assert cosine_distance(first, second) == pytest.approx(expected)


@pytest.mark.parametrize(
    "first, second, message",
    [
        ([1, 0], [1, 0, 0], "1-D"),
        ([[1, 0]], [1, 0], "1-D"),
        ([1, 0], [[1, 0]], "1-D"),
        ([0, 0], [1, 0], "non-zero"),
        ([1, 0], [0, 0], "non-zero"),
    ],
)
def test_cosine_distance_rejects_invalid_vectors(first, second, message):
    with pytest.raises(ValueError, match=message):
        cosine_distance(
            np.array(first, dtype=np.float32),
            np.array(second, dtype=np.float32),
        )
