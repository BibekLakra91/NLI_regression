from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = "issue_priority_dataset.csv"
DEFAULT_OUTPUT_DIR = "artifacts/stratified_minilm"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_COLUMN = "issue_description"
DEFAULT_TARGET_COLUMN = "priority_score"
DEFAULT_VAL_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_CLIP_MIN = 0.0
DEFAULT_CLIP_MAX = 100.0
BUCKET_RANGES = (
    ("0-40", 0, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
)


@dataclass
class Example:
    issue_id: str
    text: str
    target: float


@dataclass
class ModelBundleMetadata:
    embedding_model: str
    device: str
    text_column: str
    target_column: str
    clip_min: float
    clip_max: float


class SentenceEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._model: SentenceTransformer | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [normalize_text(text) for text in texts]
        cache_path = self._cache_path(prepared)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as payload:
                return payload["embeddings"]

        embeddings = self._get_model().encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, embeddings=embeddings)

        return embeddings

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_path(self, texts: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None

        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        filename = f"{slugify(self.model_name)}_{digest}.npz"
        return self.cache_dir / filename


class PriorityModel:
    def __init__(self, estimator: RidgeCV, metadata: ModelBundleMetadata) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self._embedder: SentenceEmbedder | None = None

    def predict_priority(self, texts: Sequence[str]) -> np.ndarray:
        features = self._get_embedder().embed(texts)
        predictions = self.estimator.predict(features)
        return np.clip(np.asarray(predictions, dtype=np.float32), self.metadata.clip_min, self.metadata.clip_max)

    def save(self, path: Path | str) -> None:
        bundle_path = Path(path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("wb") as handle:
            pickle.dump({"estimator": self.estimator, "metadata": asdict(self.metadata)}, handle)

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


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip("-") or "model"


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_examples(dataset_path: Path, text_column: str, target_column: str) -> list[Example]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header row: {dataset_path}")
        if text_column not in reader.fieldnames or target_column not in reader.fieldnames:
            raise ValueError(
                f"Dataset must contain '{text_column}' and '{target_column}'. Found: {reader.fieldnames}"
            )

        examples: list[Example] = []
        for index, row in enumerate(reader, start=1):
            text = normalize_text(row[text_column])
            if not text:
                raise ValueError(f"Row {index} has empty text.")
            examples.append(
                Example(
                    issue_id=str(row.get("issue_id", index)),
                    text=text,
                    target=float(row[target_column]),
                )
            )

    if not examples:
        raise ValueError("Dataset is empty.")
    return examples


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


def summarize_bucket_distribution(examples: Sequence[Example]) -> list[dict[str, Any]]:
    total = len(examples)
    summary: list[dict[str, Any]] = []
    for label, _, _ in BUCKET_RANGES:
        count = sum(1 for example in examples if bucket_label(example.target) == label)
        summary.append(
            {
                "bucket": label,
                "count": count,
                "share": round(count / total, 4) if total else 0.0,
            }
        )
    return summary


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        next_index = index + 1
        while next_index < len(values) and values[order[next_index]] == values[order[index]]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        ranks[order[index:next_index]] = average_rank
        index = next_index
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_ranks = rankdata(y_true)
    pred_ranks = rankdata(y_pred)
    if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0:
        return 0.0
    return float(np.corrcoef(true_ranks, pred_ranks)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "spearman": round(spearman(y_true, y_pred), 4),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def log(message: str) -> None:
    print(f"[train_issue_priority] {message}", flush=True)


def resolve_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "embedding_cache"

    log(f"Loading dataset from {dataset_path}")
    examples = load_examples(dataset_path, args.text_column, args.target_column)
    stratify_labels = [bucket_label(example.target) for example in examples]
    log(f"Loaded {len(examples)} rows")

    example_indexes = np.arange(len(examples))
    train_indexes, validation_indexes = train_test_split(
        example_indexes,
        test_size=args.val_size,
        random_state=args.random_seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    train_examples = [examples[index] for index in train_indexes]
    validation_examples = [examples[index] for index in validation_indexes]
    log(
        "Built stratified split: "
        f"{len(train_examples)} train rows, {len(validation_examples)} validation rows"
    )

    resolved_device = resolve_device(args.device)
    log(f"Embedding device resolved to {resolved_device}")
    embedder = SentenceEmbedder(
        args.embedding_model,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    log(f"Embedding full dataset with model {args.embedding_model}")
    full_embeddings = embedder.embed([example.text for example in examples])
    log(f"Computed embedding matrix with shape {tuple(full_embeddings.shape)}")

    x_train = full_embeddings[train_indexes]
    x_validation = full_embeddings[validation_indexes]
    y_train = np.asarray([example.target for example in train_examples], dtype=np.float32)
    y_validation = np.asarray([example.target for example in validation_examples], dtype=np.float32)

    log("Fitting validation model (RidgeCV)")
    estimator = RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0))
    estimator.fit(x_train, y_train)
    validation_predictions = np.clip(estimator.predict(x_validation), DEFAULT_CLIP_MIN, DEFAULT_CLIP_MAX)
    validation_predictions = np.asarray(validation_predictions, dtype=np.float32)
    validation_metrics = compute_metrics(y_validation, validation_predictions)
    log(
        "Validation complete: "
        f"MAE={validation_metrics['mae']}, "
        f"RMSE={validation_metrics['rmse']}, "
        f"Spearman={validation_metrics['spearman']}"
    )

    full_targets = np.asarray([example.target for example in examples], dtype=np.float32)
    log("Refitting final model on full dataset")
    final_estimator = RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0))
    final_estimator.fit(full_embeddings, full_targets)
    full_predictions = np.clip(final_estimator.predict(full_embeddings), DEFAULT_CLIP_MIN, DEFAULT_CLIP_MAX)
    full_predictions = np.asarray(full_predictions, dtype=np.float32)

    final_model = PriorityModel(
        estimator=final_estimator,
        metadata=ModelBundleMetadata(
            embedding_model=args.embedding_model,
            device=resolved_device,
            text_column=args.text_column,
            target_column=args.target_column,
            clip_min=DEFAULT_CLIP_MIN,
            clip_max=DEFAULT_CLIP_MAX,
        ),
    )
    model_path = output_dir / "final_model.pkl"
    final_model.save(model_path)
    log(f"Saved final model bundle to {model_path}")

    validation_rows: list[dict[str, Any]] = []
    for example, prediction in zip(validation_examples, validation_predictions, strict=True):
        validation_rows.append(
            {
                "issue_id": example.issue_id,
                "issue_description": example.text,
                "actual_score": round(example.target, 4),
                "predicted_score": round(float(prediction), 4),
                "absolute_error": round(abs(float(prediction) - example.target), 4),
                "actual_bucket": bucket_label(example.target),
                "predicted_bucket": bucket_label(float(prediction)),
            }
        )
    validation_rows.sort(key=lambda row: row["absolute_error"], reverse=True)

    full_rows: list[dict[str, Any]] = []
    for example, prediction in zip(examples, full_predictions, strict=True):
        full_rows.append(
            {
                "issue_id": example.issue_id,
                "issue": example.text,
                "score": round(example.target, 4),
                "predicted_score": round(float(prediction), 4),
                "absolute_error": round(abs(float(prediction) - example.target), 4),
                "actual_bucket": bucket_label(example.target),
                "predicted_bucket": bucket_label(float(prediction)),
            }
        )

    summary = {
        "dataset_path": str(dataset_path),
        "row_count": len(examples),
        "embedding_model": args.embedding_model,
        "device": resolved_device,
        "random_seed": args.random_seed,
        "validation_fraction": args.val_size,
        "train_distribution": summarize_bucket_distribution(train_examples),
        "validation_distribution": summarize_bucket_distribution(validation_examples),
        "full_distribution": summarize_bucket_distribution(examples),
        "validation_metrics": validation_metrics,
        "selected_alpha": round(float(final_estimator.alpha_), 6),
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
        "paths": {
            "model_bundle": str(model_path),
            "summary_json": str(output_dir / "training_summary.json"),
            "validation_predictions_csv": str(output_dir / "validation_predictions.csv"),
            "full_predictions_csv": str(output_dir / "full_predictions.csv"),
        },
    }

    write_json(output_dir / "training_summary.json", summary)
    write_csv(output_dir / "validation_predictions.csv", validation_rows)
    write_csv(output_dir / "full_predictions.csv", full_rows)
    log(
        "Wrote artifacts: "
        f"{output_dir / 'training_summary.json'}, "
        f"{output_dir / 'validation_predictions.csv'}, "
        f"{output_dir / 'full_predictions.csv'}"
    )
    log(f"Finished in {summary['elapsed_seconds']} seconds")
    return summary


def predict(args: argparse.Namespace) -> list[dict[str, Any]]:
    model = PriorityModel.load(Path(args.model_bundle))
    predictions = model.predict_priority(args.texts)
    return [
        {"text": text, "predicted_priority_score": round(float(score), 4)}
        for text, score in zip(args.texts, predictions, strict=True)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fresh issue-priority regression model with bucket-stratified train/validation splits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train with a bucket-stratified train/validation split.")
    train_parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Input CSV path.")
    train_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    train_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model.")
    train_parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Text column name.")
    train_parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN, help="Target column name.")
    train_parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE, help="Validation fraction.")
    train_parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    train_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Embedding device. 'auto' uses CUDA when available, otherwise CPU.",
    )

    predict_parser = subparsers.add_parser("predict", help="Score one or more texts with a saved model.")
    predict_parser.add_argument("--model-bundle", required=True, help="Path to final_model.pkl.")
    predict_parser.add_argument("--text", dest="texts", action="append", required=True, help="Text to score.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        print(json.dumps(train(args), indent=2))
        return 0
    if args.command == "predict":
        print(json.dumps(predict(args), indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
