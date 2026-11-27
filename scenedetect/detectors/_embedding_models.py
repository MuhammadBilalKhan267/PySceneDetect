from typing import Any

import numpy as np

from abc import ABC, abstractmethod
import cv2
from cv2.dnn import readNetFromONNX, Net

_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    short = min(h, w)
    resized = cv2.resize(
        frame, (w * 256 // short, h * 256 // short),
        interpolation=cv2.INTER_CUBIC,
    )
    y, x = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
    rgb = cv2.cvtColor(resized[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - _RGB_MEAN) / _RGB_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

class _OpenCVEmbedder(ABC):
    def __init__(self):
        self._model: Net | None = None
    def embed(self, frame: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._model = self._load_model()
        return self._embed(self._model, frame)
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.embed(frame)
    @abstractmethod
    def _load_model(self)-> Net:
        ...
    @abstractmethod
    def _embed(self, model: Net, frame: np.ndarray) -> np.ndarray:
        ...

class DinoEmbedder(_OpenCVEmbedder):
    def __init__(self, model_path: str):
        super().__init__()
        self._model_path = model_path
    def _load_model(self) -> Net:
        return readNetFromONNX(self._model_path)
    def _embed(self, model: Net, frame: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        model.setInput(_preprocess(frame))
        output = model.forward()
        if output.shape != (1, 257, 384):
            raise ValueError(f"Expected DINOv2 token output (1, 257, 384), got {output.shape}")
        return output[0, 0]  # First image, first token (CLS): shape (384,)
        