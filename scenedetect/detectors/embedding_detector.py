from scenedetect.detector import SceneDetector
from scenedetect.common import FramePosition, TimecodeLike, parse_frame_or_time
from scenedetect.detectors._embedding_models import DinoEmbedder
import numpy as np

from typing import Callable

Embedder = Callable[[np.ndarray], np.ndarray]
Distance = Callable[[np.ndarray, np.ndarray], float]

def cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    return 1.0 - cosine_similarity(e1, e2)

def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    if e1.ndim != 1 or e2.ndim != 1 or e1.shape != e2.shape:
        raise ValueError(
            f"Expected two 1-D embeddings of the same shape, got {e1.shape} and {e2.shape}"
        )
    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
    if denom == 0:
        raise ValueError('Cosine similarity requires non-zero vectors')
    return float(np.clip((e1 @ e2) / (denom), -1.0, 1.0))

class EmbeddingDetector(SceneDetector):
    def __init__(self, *, threshold: float, min_scene_len: TimecodeLike, embedder: Embedder, distance: Distance = cosine_distance):
        super().__init__()
        self._threshold = threshold
        self._min_scene_len = parse_frame_or_time(min_scene_len)
        self._embedder: Embedder = embedder
        self._distance: Distance  = distance
        self._last_scene_cut: FramePosition | None = None
        self._previous_embedding: np.ndarray | None = None

    def process_frame(self, position: FramePosition, frame_img: np.ndarray) -> list[FramePosition]:
        cut_list = []
        current = self._embedder(frame_img)
        last_cut = self._last_scene_cut
        previous = self._previous_embedding
        self._previous_embedding = current
        if last_cut is None or previous is None:
            self._last_scene_cut = position
            return cut_list
        
        score = self._distance(previous, current)
        if score >= self._threshold and self.meets_minimum(position  - last_cut, self._min_scene_len):
            cut_list.append(position)
            self._last_scene_cut = position

        return cut_list

