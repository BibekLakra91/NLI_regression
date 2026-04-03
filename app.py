from __future__ import annotations

import csv
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from issue_priority_inference import BUCKET_RANGES, PriorityModel, bucket_label, load_summary, normalize_text


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_BUNDLE = "artifacts/stratified_minilm_3200/final_model.pkl"
DEFAULT_SUMMARY_PATH = "artifacts/stratified_minilm_3200/training_summary.json"


def resolve_project_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


MODEL_BUNDLE_PATH = resolve_project_path(os.environ.get("MODEL_BUNDLE_PATH", DEFAULT_MODEL_BUNDLE))
SUMMARY_PATH = resolve_project_path(os.environ.get("SUMMARY_PATH", DEFAULT_SUMMARY_PATH))

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_model() -> PriorityModel:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}")
    return PriorityModel.load(MODEL_BUNDLE_PATH)


@lru_cache(maxsize=1)
def get_summary() -> dict[str, Any] | None:
    return load_summary(SUMMARY_PATH)


def score_texts(texts: list[str]) -> list[dict[str, Any]]:
    cleaned = [normalize_text(text) for text in texts if normalize_text(text)]
    if not cleaned:
        return []

    predictions = get_model().predict_priority(cleaned)
    rows: list[dict[str, Any]] = []
    for text, score in zip(cleaned, predictions, strict=True):
        numeric_score = round(float(score), 4)
        rows.append(
            {
                "issue": text,
                "predicted_score": numeric_score,
                "predicted_bucket": bucket_label(numeric_score),
            }
        )
    return rows


def parse_uploaded_csv() -> list[str]:
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return []

    content = uploaded_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("Uploaded CSV must include a header row.")

    if "issue" in reader.fieldnames:
        column = "issue"
    elif "issue_description" in reader.fieldnames:
        column = "issue_description"
    else:
        raise ValueError("CSV must contain either an 'issue' or 'issue_description' column.")

    return [row.get(column, "") for row in reader]


def parse_batch_input() -> list[str]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError("'issues' must be a list of strings.")
        return [str(issue) for issue in issues]

    rows: list[str] = []
    textarea_value = request.form.get("batch_text", "")
    if textarea_value.strip():
        rows.extend(line for line in textarea_value.splitlines() if line.strip())
    rows.extend(parse_uploaded_csv())
    return rows


@app.get("/")
def index() -> str:
    summary = get_summary()
    metrics = (summary or {}).get("validation_metrics", {})
    return render_template(
        "index.html",
        bucket_ranges=BUCKET_RANGES,
        summary=summary,
        metrics=metrics,
        model_bundle_path=str(MODEL_BUNDLE_PATH.relative_to(BASE_DIR)) if MODEL_BUNDLE_PATH.is_relative_to(BASE_DIR) else str(MODEL_BUNDLE_PATH),
        model_ready=MODEL_BUNDLE_PATH.exists(),
    )


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "model_ready": MODEL_BUNDLE_PATH.exists(),
            "model_bundle": str(MODEL_BUNDLE_PATH),
            "summary_path": str(SUMMARY_PATH),
        }
    )


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    issue = normalize_text(str(payload.get("issue", "")))
    if not issue:
        return jsonify({"error": "Provide a non-empty issue description."}), 400

    try:
        prediction = score_texts([issue])[0]
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(prediction)


@app.post("/api/batch")
def batch() -> Any:
    try:
        rows = parse_batch_input()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not rows:
        return jsonify({"error": "Provide batch text or upload a compatible CSV."}), 400

    try:
        predictions = score_texts(rows)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"count": len(predictions), "predictions": predictions})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
