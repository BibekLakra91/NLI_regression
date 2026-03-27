from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_DATASET_PATH = "issue_priority_dataset.csv"
DEFAULT_TEXT_COLUMN = "issue_description"
DEFAULT_TARGET_COLUMN = "priority_score"
DEFAULT_OUTPUT_DIR = "artifacts/issue_priority_regression"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ALTERNATIVE_EMBEDDING_MODEL = "intfloat/e5-small-v2"
DEFAULT_FOLDS = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_CLIP_MIN = 0.0
DEFAULT_CLIP_MAX = 100.0
DEFAULT_WORST_ERROR_COUNT = 20


class TrainingRuntimeError(RuntimeError):
    pass


@dataclass
class DatasetExample:
    row_number: int
    issue_id: str
    text: str
    target: float


@dataclass
class ModelArtifactMetadata:
    model_name: str
    embedding_model: str | None
    clip_min: float
    clip_max: float
    text_column: str
    target_column: str


class MeanRegressor:
    def __init__(self, constant_value: float) -> None:
        self.constant_value = constant_value

    def predict(self, features: Sequence[Any]) -> Any:
        np = _require_numpy()
        return np.full(len(features), self.constant_value, dtype=np.float32)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise TrainingRuntimeError(
            "Missing dependency 'numpy'. Install the CPU stack before training or inference."
        ) from exc
    return np


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise TrainingRuntimeError(
            "Missing dependency 'scikit-learn'. Install the CPU stack before training or inference."
        ) from exc

    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "KFold": KFold,
        "Pipeline": Pipeline,
        "RidgeCV": RidgeCV,
        "StratifiedKFold": StratifiedKFold,
        "TfidfVectorizer": TfidfVectorizer,
    }


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_texts(texts: Sequence[str]) -> list[str]:
    return [normalize_text(text) for text in texts]


def _load_examples(
    dataset_path: Path,
    text_column: str,
    target_column: str,
) -> list[DatasetExample]:
    if not dataset_path.exists():
        raise TrainingRuntimeError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TrainingRuntimeError(f"Dataset has no header row: {dataset_path}")
        if text_column not in reader.fieldnames:
            raise TrainingRuntimeError(
                f"Missing text column '{text_column}' in {dataset_path}. Available columns: {reader.fieldnames}"
            )
        if target_column not in reader.fieldnames:
            raise TrainingRuntimeError(
                f"Missing target column '{target_column}' in {dataset_path}. Available columns: {reader.fieldnames}"
            )

        examples: list[DatasetExample] = []
        for index, row in enumerate(reader, start=1):
            raw_text = row.get(text_column, "")
            raw_target = row.get(target_column, "")
            text = normalize_text(raw_text)
            if not text:
                raise TrainingRuntimeError(f"Row {index} has empty text in column '{text_column}'.")

            try:
                target = float(raw_target)
            except (TypeError, ValueError) as exc:
                raise TrainingRuntimeError(
                    f"Row {index} has invalid numeric target '{raw_target}' in column '{target_column}'."
                ) from exc

            examples.append(
                DatasetExample(
                    row_number=index,
                    issue_id=str(row.get("issue_id", index)),
                    text=text,
                    target=target,
                )
            )

    if not examples:
        raise TrainingRuntimeError(f"Dataset is empty: {dataset_path}")

    return examples


class SentenceEmbeddingStage:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def embed(self, texts: Sequence[str]) -> Any:
        np = _require_numpy()
        prepared_texts = self._prepare_texts(texts)
        cache_path = self._cache_path(prepared_texts)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as payload:
                return payload["embeddings"]

        model = self._load_model()
        embeddings = model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, embeddings=embeddings)

        return embeddings

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise TrainingRuntimeError(
                "Missing dependency 'sentence-transformers'. Install it to compute embeddings."
            ) from exc

        self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _prepare_texts(self, texts: Sequence[str]) -> list[str]:
        cleaned_texts = _normalize_texts(texts)
        if self._uses_e5_prefix():
            return [f"passage: {text}" for text in cleaned_texts]
        return cleaned_texts

    def _uses_e5_prefix(self) -> bool:
        return "e5" in self.model_name.lower()

    def _cache_path(self, prepared_texts: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None

        joined = "\n".join(prepared_texts).encode("utf-8")
        digest = hashlib.sha256(joined).hexdigest()[:16]
        filename = f"{_slugify(self.model_name)}_{digest}.npz"
        return self.cache_dir / filename


class PriorityRegressionPipeline:
    def __init__(self, estimator: Any, metadata: ModelArtifactMetadata) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self._embedder: SentenceEmbeddingStage | None = None

    def predict_priority(self, texts: Sequence[str]) -> Any:
        np = _require_numpy()
        normalized_texts = _normalize_texts(texts)

        if self.metadata.embedding_model:
            features = self._get_embedder().embed(normalized_texts)
            predictions = self.estimator.predict(features)
        else:
            predictions = self.estimator.predict(normalized_texts)

        predictions = np.asarray(predictions, dtype=np.float32)
        return np.clip(predictions, self.metadata.clip_min, self.metadata.clip_max)

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "estimator": self.estimator,
            "metadata": asdict(self.metadata),
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, source: Path) -> "PriorityRegressionPipeline":
        with source.open("rb") as handle:
            payload = pickle.load(handle)

        metadata = ModelArtifactMetadata(**payload["metadata"])
        return cls(estimator=payload["estimator"], metadata=metadata)

    def _get_embedder(self) -> SentenceEmbeddingStage:
        if self._embedder is None:
            self._embedder = SentenceEmbeddingStage(model_name=self.metadata.embedding_model or DEFAULT_EMBEDDING_MODEL)
        return self._embedder


def _build_regression_splits(targets: Any, fold_count: int, random_seed: int) -> list[tuple[Any, Any]]:
    np = _require_numpy()
    sklearn_parts = _require_sklearn()
    KFold = sklearn_parts["KFold"]
    StratifiedKFold = sklearn_parts["StratifiedKFold"]

    target_array = np.asarray(targets, dtype=np.float32)
    if len(target_array) < fold_count:
        raise TrainingRuntimeError(
            f"Dataset has {len(target_array)} rows, which is smaller than the requested {fold_count} folds."
        )

    max_bin_count = min(10, len(target_array) // fold_count)
    if max_bin_count >= 2:
        quantiles = np.quantile(target_array, np.linspace(0.0, 1.0, max_bin_count + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) >= 3:
            binned = np.digitize(target_array, quantiles[1:-1], right=True)
            bin_counts = np.bincount(binned)
            if len(bin_counts) > 0 and int(bin_counts.min()) >= fold_count:
                splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
                return list(splitter.split(np.zeros(len(target_array)), binned))

    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
    return list(splitter.split(np.zeros(len(target_array))))


def _rankdata(values: Any) -> Any:
    np = _require_numpy()
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    index = 0
    while index < len(array):
        next_index = index + 1
        while next_index < len(array) and array[order[next_index]] == array[order[index]]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        ranks[order[index:next_index]] = average_rank
        index = next_index
    return ranks


def _spearman_correlation(y_true: Any, y_pred: Any) -> float:
    np = _require_numpy()
    true_ranks = _rankdata(y_true)
    pred_ranks = _rankdata(y_pred)
    if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0:
        return 0.0
    return float(np.corrcoef(true_ranks, pred_ranks)[0, 1])


def _compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    np = _require_numpy()
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    spearman = _spearman_correlation(actual, predicted)
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "spearman": round(spearman, 4),
    }


def _clip_predictions(predictions: Any) -> Any:
    np = _require_numpy()
    return np.clip(predictions, DEFAULT_CLIP_MIN, DEFAULT_CLIP_MAX)


def _evaluate_mean_baseline(targets: Any, splits: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    np = _require_numpy()
    target_array = np.asarray(targets, dtype=np.float32)
    predictions = np.empty(len(target_array), dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_index, validation_index) in enumerate(splits, start=1):
        prediction = np.full(len(validation_index), target_array[train_index].mean(), dtype=np.float32)
        prediction = _clip_predictions(prediction)
        predictions[validation_index] = prediction
        metrics = _compute_metrics(target_array[validation_index], prediction)
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)

    return {
        "name": "mean_baseline",
        "oof_predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary_metrics": _compute_metrics(target_array, predictions),
        "embedding_model": None,
    }


def _evaluate_text_regressor(
    model_name: str,
    builder: Callable[[], Any],
    texts: Sequence[str],
    targets: Any,
    splits: Sequence[tuple[Any, Any]],
    embedding_model: str | None = None,
) -> dict[str, Any]:
    np = _require_numpy()
    text_list = list(texts)
    target_array = np.asarray(targets, dtype=np.float32)
    predictions = np.empty(len(target_array), dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_index, validation_index) in enumerate(splits, start=1):
        estimator = builder()
        train_texts = [text_list[i] for i in train_index]
        validation_texts = [text_list[i] for i in validation_index]
        estimator.fit(train_texts, target_array[train_index])
        fold_predictions = _clip_predictions(estimator.predict(validation_texts))
        predictions[validation_index] = fold_predictions
        metrics = _compute_metrics(target_array[validation_index], fold_predictions)
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)

    return {
        "name": model_name,
        "oof_predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary_metrics": _compute_metrics(target_array, predictions),
        "embedding_model": embedding_model,
    }


def _evaluate_feature_regressor(
    model_name: str,
    builder: Callable[[], Any],
    features: Any,
    targets: Any,
    splits: Sequence[tuple[Any, Any]],
    embedding_model: str | None,
) -> dict[str, Any]:
    np = _require_numpy()
    target_array = np.asarray(targets, dtype=np.float32)
    predictions = np.empty(len(target_array), dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_index, validation_index) in enumerate(splits, start=1):
        estimator = builder()
        estimator.fit(features[train_index], target_array[train_index])
        fold_predictions = _clip_predictions(estimator.predict(features[validation_index]))
        predictions[validation_index] = fold_predictions
        metrics = _compute_metrics(target_array[validation_index], fold_predictions)
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)

    return {
        "name": model_name,
        "oof_predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary_metrics": _compute_metrics(target_array, predictions),
        "embedding_model": embedding_model,
    }


def _bucketed_error_summary(targets: Any, predictions: Any) -> list[dict[str, Any]]:
    np = _require_numpy()
    target_array = np.asarray(targets, dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    ranges = [(0, 40), (41, 60), (61, 80), (81, 100)]
    summary: list[dict[str, Any]] = []

    for lower, upper in ranges:
        mask = (target_array >= lower) & (target_array <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        mae = float(np.mean(np.abs(prediction_array[mask] - target_array[mask])))
        summary.append(
            {
                "range": f"{lower}-{upper}",
                "count": count,
                "mae": round(mae, 4),
            }
        )

    return summary


def _build_worst_error_rows(
    examples: Sequence[DatasetExample],
    predictions: Any,
    top_n: int = DEFAULT_WORST_ERROR_COUNT,
) -> list[dict[str, Any]]:
    np = _require_numpy()
    prediction_array = np.asarray(predictions, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for example, prediction in zip(examples, prediction_array, strict=True):
        absolute_error = abs(float(prediction) - example.target)
        rows.append(
            {
                "issue_id": example.issue_id,
                "row_number": example.row_number,
                "issue_description": example.text,
                "actual_score": round(example.target, 4),
                "predicted_score": round(float(prediction), 4),
                "absolute_error": round(absolute_error, 4),
            }
        )
    rows.sort(key=lambda row: row["absolute_error"], reverse=True)
    return rows[:top_n]


def _fit_tfidf_ridge() -> Any:
    sklearn_parts = _require_sklearn()
    Pipeline = sklearn_parts["Pipeline"]
    RidgeCV = sklearn_parts["RidgeCV"]
    TfidfVectorizer = sklearn_parts["TfidfVectorizer"]

    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            ("regressor", RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0))),
        ]
    )


def _fit_embedding_ridge() -> Any:
    sklearn_parts = _require_sklearn()
    RidgeCV = sklearn_parts["RidgeCV"]
    return RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0))


def _fit_hist_gradient_boosting() -> Any:
    sklearn_parts = _require_sklearn()
    HistGradientBoostingRegressor = sklearn_parts["HistGradientBoostingRegressor"]
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=DEFAULT_RANDOM_SEED,
    )


def _fit_xgboost() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise TrainingRuntimeError(
            "Missing dependency 'xgboost'. Install it or omit --include-xgboost."
        ) from exc

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=1,
        random_state=DEFAULT_RANDOM_SEED,
    )


def train_and_evaluate(
    dataset_path: Path,
    output_dir: Path,
    text_column: str = DEFAULT_TEXT_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    alternative_embedding_model: str | None = None,
    embedding_batch_size: int = 64,
    folds: int = DEFAULT_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_xgboost: bool = False,
    include_alternative_embedding_model: bool = False,
) -> dict[str, Any]:
    np = _require_numpy()

    examples = _load_examples(dataset_path, text_column=text_column, target_column=target_column)
    texts = [example.text for example in examples]
    targets = np.asarray([example.target for example in examples], dtype=np.float32)
    splits = _build_regression_splits(targets, fold_count=folds, random_seed=random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir = output_dir / "embedding_cache"

    results: list[dict[str, Any]] = []
    skipped_models: list[dict[str, str]] = []

    results.append(_evaluate_mean_baseline(targets, splits))
    results.append(
        _evaluate_text_regressor(
            model_name="tfidf_ridge",
            builder=_fit_tfidf_ridge,
            texts=texts,
            targets=targets,
            splits=splits,
        )
    )

    default_embedder = SentenceEmbeddingStage(
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        cache_dir=embedding_cache_dir,
    )
    try:
        default_embeddings = default_embedder.embed(texts)
    except TrainingRuntimeError as exc:
        skipped_models.append({"model": "embedding_models", "reason": str(exc)})
        default_embeddings = None

    if default_embeddings is not None:
        results.append(
            _evaluate_feature_regressor(
                model_name="embedding_ridge",
                builder=_fit_embedding_ridge,
                features=default_embeddings,
                targets=targets,
                splits=splits,
                embedding_model=embedding_model,
            )
        )
        results.append(
            _evaluate_feature_regressor(
                model_name="embedding_hist_gradient_boosting",
                builder=_fit_hist_gradient_boosting,
                features=default_embeddings,
                targets=targets,
                splits=splits,
                embedding_model=embedding_model,
            )
        )

        if include_xgboost:
            try:
                results.append(
                    _evaluate_feature_regressor(
                        model_name="embedding_xgboost",
                        builder=_fit_xgboost,
                        features=default_embeddings,
                        targets=targets,
                        splits=splits,
                        embedding_model=embedding_model,
                    )
                )
            except TrainingRuntimeError as exc:
                skipped_models.append({"model": "embedding_xgboost", "reason": str(exc)})

    if include_alternative_embedding_model and alternative_embedding_model:
        alternative_embedder = SentenceEmbeddingStage(
            model_name=alternative_embedding_model,
            batch_size=embedding_batch_size,
            cache_dir=embedding_cache_dir,
        )
        try:
            alternative_embeddings = alternative_embedder.embed(texts)
            results.append(
                _evaluate_feature_regressor(
                    model_name="alternative_embedding_ridge",
                    builder=_fit_embedding_ridge,
                    features=alternative_embeddings,
                    targets=targets,
                    splits=splits,
                    embedding_model=alternative_embedding_model,
                )
            )
            results.append(
                _evaluate_feature_regressor(
                    model_name="alternative_embedding_hist_gradient_boosting",
                    builder=_fit_hist_gradient_boosting,
                    features=alternative_embeddings,
                    targets=targets,
                    splits=splits,
                    embedding_model=alternative_embedding_model,
                )
            )
        except TrainingRuntimeError as exc:
            skipped_models.append({"model": "alternative_embedding_models", "reason": str(exc)})

    if not results:
        raise TrainingRuntimeError("No models were evaluated.")

    for result in results:
        result["bucketed_error"] = _bucketed_error_summary(targets, result["oof_predictions"])
        result["worst_errors"] = _build_worst_error_rows(examples, result["oof_predictions"])

    results.sort(key=lambda item: (item["summary_metrics"]["mae"], item["summary_metrics"]["rmse"]))
    best_result = results[0]

    best_pipeline = _fit_final_pipeline(
        model_name=best_result["name"],
        texts=texts,
        targets=targets,
        embedding_model=best_result["embedding_model"],
        embedding_batch_size=embedding_batch_size,
        embedding_cache_dir=embedding_cache_dir,
        text_column=text_column,
        target_column=target_column,
    )

    artifact_path = output_dir / "best_model.pkl"
    best_pipeline.save(artifact_path)

    prediction_rows = _build_prediction_rows(examples, results)
    _write_json(output_dir / "evaluation_summary.json", _serialize_results(dataset_path, results, skipped_models, best_result))
    _write_csv(output_dir / "cv_predictions.csv", prediction_rows)
    _write_csv(output_dir / "best_model_worst_errors.csv", best_result["worst_errors"])

    return {
        "artifact_path": str(artifact_path),
        "best_model": best_result["name"],
        "best_model_metrics": best_result["summary_metrics"],
        "results": _serialize_results(dataset_path, results, skipped_models, best_result)["results"],
        "skipped_models": skipped_models,
    }


def _fit_final_pipeline(
    model_name: str,
    texts: Sequence[str],
    targets: Any,
    embedding_model: str | None,
    embedding_batch_size: int,
    embedding_cache_dir: Path,
    text_column: str,
    target_column: str,
) -> PriorityRegressionPipeline:
    np = _require_numpy()
    target_array = np.asarray(targets, dtype=np.float32)
    normalized_texts = _normalize_texts(texts)

    if model_name == "mean_baseline":
        mean_target = float(target_array.mean())
        estimator = MeanRegressor(mean_target)
        metadata = ModelArtifactMetadata(
            model_name=model_name,
            embedding_model=None,
            clip_min=DEFAULT_CLIP_MIN,
            clip_max=DEFAULT_CLIP_MAX,
            text_column=text_column,
            target_column=target_column,
        )
        return PriorityRegressionPipeline(estimator=estimator, metadata=metadata)

    if model_name == "tfidf_ridge":
        estimator = _fit_tfidf_ridge()
        estimator.fit(normalized_texts, target_array)
        metadata = ModelArtifactMetadata(
            model_name=model_name,
            embedding_model=None,
            clip_min=DEFAULT_CLIP_MIN,
            clip_max=DEFAULT_CLIP_MAX,
            text_column=text_column,
            target_column=target_column,
        )
        return PriorityRegressionPipeline(estimator=estimator, metadata=metadata)

    if embedding_model is None:
        raise TrainingRuntimeError(f"Model '{model_name}' requires an embedding model but none was provided.")

    embedder = SentenceEmbeddingStage(
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        cache_dir=embedding_cache_dir,
    )
    embeddings = embedder.embed(normalized_texts)

    if model_name in {"embedding_ridge", "alternative_embedding_ridge"}:
        estimator = _fit_embedding_ridge()
    elif model_name in {
        "embedding_hist_gradient_boosting",
        "alternative_embedding_hist_gradient_boosting",
    }:
        estimator = _fit_hist_gradient_boosting()
    elif model_name == "embedding_xgboost":
        estimator = _fit_xgboost()
    else:
        raise TrainingRuntimeError(f"Unsupported final model: {model_name}")

    estimator.fit(embeddings, target_array)
    metadata = ModelArtifactMetadata(
        model_name=model_name,
        embedding_model=embedding_model,
        clip_min=DEFAULT_CLIP_MIN,
        clip_max=DEFAULT_CLIP_MAX,
        text_column=text_column,
        target_column=target_column,
    )
    return PriorityRegressionPipeline(estimator=estimator, metadata=metadata)


def _build_prediction_rows(examples: Sequence[DatasetExample], results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    np = _require_numpy()
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        row = {
            "issue_id": example.issue_id,
            "row_number": example.row_number,
            "issue_description": example.text,
            "actual_score": round(example.target, 4),
        }
        for result in results:
            predictions = np.asarray(result["oof_predictions"], dtype=np.float32)
            row[f"{result['name']}_prediction"] = round(float(predictions[index]), 4)
        rows.append(row)
    return rows


def _serialize_results(
    dataset_path: Path,
    results: Sequence[dict[str, Any]],
    skipped_models: Sequence[dict[str, str]],
    best_result: dict[str, Any],
) -> dict[str, Any]:
    serialized_results: list[dict[str, Any]] = []
    for result in results:
        serialized_results.append(
            {
                "name": result["name"],
                "embedding_model": result["embedding_model"],
                "summary_metrics": result["summary_metrics"],
                "fold_metrics": result["fold_metrics"],
                "bucketed_error": result["bucketed_error"],
            }
        )

    return {
        "dataset_path": str(dataset_path),
        "best_model": {
            "name": best_result["name"],
            "embedding_model": best_result["embedding_model"],
            "summary_metrics": best_result["summary_metrics"],
        },
        "results": serialized_results,
        "skipped_models": list(skipped_models),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use a CPU-first regression pipeline for issue priority scoring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate regression models.")
    train_parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Input CSV path.")
    train_parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Text column name.")
    train_parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN, help="Numeric target column name.")
    train_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports and model artifacts.")
    train_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model used for the main embedding runs.",
    )
    train_parser.add_argument(
        "--alternative-embedding-model",
        default=DEFAULT_ALTERNATIVE_EMBEDDING_MODEL,
        help="Optional alternative embedding model to evaluate when enabled.",
    )
    train_parser.add_argument(
        "--include-alternative-embedding-model",
        action="store_true",
        help="Also evaluate the alternative embedding model with Ridge and HistGradientBoosting.",
    )
    train_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size used when computing sentence embeddings on CPU.",
    )
    train_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Cross-validation fold count.")
    train_parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    train_parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Also evaluate XGBoost on cached embeddings when the package is installed.",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference with a saved model bundle.")
    predict_parser.add_argument("--model-bundle", required=True, help="Path to best_model.pkl or another saved bundle.")
    predict_parser.add_argument(
        "--text",
        dest="texts",
        action="append",
        required=True,
        help="Issue description to score. Repeat --text to score multiple items.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            report = train_and_evaluate(
                dataset_path=Path(args.dataset),
                output_dir=Path(args.output_dir),
                text_column=args.text_column,
                target_column=args.target_column,
                embedding_model=args.embedding_model,
                alternative_embedding_model=args.alternative_embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                folds=args.folds,
                random_seed=args.random_seed,
                include_xgboost=args.include_xgboost,
                include_alternative_embedding_model=args.include_alternative_embedding_model,
            )
            print(json.dumps(report, indent=2))
            return 0

        if args.command == "predict":
            pipeline = PriorityRegressionPipeline.load(Path(args.model_bundle))
            predictions = pipeline.predict_priority(args.texts)
            payload = [
                {"text": text, "predicted_priority_score": round(float(score), 4)}
                for text, score in zip(args.texts, predictions, strict=True)
            ]
            print(json.dumps(payload, indent=2))
            return 0
    except TrainingRuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
