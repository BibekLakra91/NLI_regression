from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BUCKET_RANGES = (
    ("0-40", 0, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
)


@dataclass
class ModelBundleMetadata:
    embedding_model: str
    device: str
    text_column: str
    target_column: str
    clip_min: float
    clip_max: float


class SentenceEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model: Any | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [normalize_text(text) for text in texts]
        embeddings = self._get_model().encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


class PriorityModel:
    def __init__(self, estimator: Any, metadata: ModelBundleMetadata) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self._embedder: SentenceEmbedder | None = None

    def predict_priority(self, texts: Sequence[str]) -> np.ndarray:
        features = self._get_embedder().embed(texts)
        predictions = self.estimator.predict(features)
        return np.clip(
            np.asarray(predictions, dtype=np.float32),
            self.metadata.clip_min,
            self.metadata.clip_max,
        )

    @classmethod
    def load(cls, path: Path | str) -> "PriorityModel":
        bundle_path = Path(path)
        with bundle_path.open("rb") as handle:
            payload = pickle.load(handle)
        metadata = dict(payload["metadata"])
        metadata.setdefault("device", "auto")
        return cls(payload["estimator"], ModelBundleMetadata(**metadata))

    def _get_embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder(self.metadata.embedding_model, device=self.metadata.device)
        return self._embedder


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def bucket_label(score: float) -> str:
    if score <= BUCKET_RANGES[0][2]:
        return BUCKET_RANGES[0][0]
    if score <= BUCKET_RANGES[1][2]:
        return BUCKET_RANGES[1][0]
    if score <= BUCKET_RANGES[2][2]:
        return BUCKET_RANGES[2][0]
    if score <= BUCKET_RANGES[3][2]:
        return BUCKET_RANGES[3][0]
    raise ValueError(f"Score {score} is outside supported bucket ranges.")


def load_summary(path: Path | str) -> dict[str, Any] | None:
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    try:
        import torch

        if requested_device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"
