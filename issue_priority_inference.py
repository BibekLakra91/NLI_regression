from __future__ import annotations

import json
import os
import pickle
import threading
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
        self.cache_dir = resolve_cache_dir()
        self._model: Any | None = None
        self._model_lock = threading.Lock()

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
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_dir),
                    )
        return self._model


class PriorityModel:
    def __init__(self, estimator: Any, metadata: ModelBundleMetadata) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self._embedder: SentenceEmbedder | None = None
        self._embedder_lock = threading.Lock()

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
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = SentenceEmbedder(self.metadata.embedding_model, device=self.metadata.device)
        return self._embedder

    def warmup(self) -> None:
        # Force the sentence-transformer weights to load before the first request.
        self._get_embedder()._get_model()


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


def resolve_cache_dir() -> Path:
    raw = (
        os.environ.get("MODEL_CACHE_DIR")
        or os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        or os.environ.get("HF_HOME")
        or ".cache/sentence-transformers"
    )
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


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
