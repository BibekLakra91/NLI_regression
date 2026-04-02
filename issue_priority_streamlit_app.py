from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from issue_priority_inference import BUCKET_RANGES, PriorityModel, bucket_label, load_summary as read_summary


DEFAULT_MODEL_BUNDLE = Path("artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl")
DEFAULT_SUMMARY_PATH = Path("artifacts/stratified_gte_modernbert_base_3200_gpu/training_summary.json")


def parse_streamlit_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-bundle", default=str(DEFAULT_MODEL_BUNDLE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    return parser.parse_args()


@st.cache_resource(show_spinner=False)
def load_model(model_bundle_path: str) -> PriorityModel:
    return PriorityModel.load(Path(model_bundle_path))


@st.cache_data(show_spinner=False)
def load_summary(summary_path: str) -> dict[str, Any] | None:
    return read_summary(Path(summary_path))


def predicted_bucket(score: float) -> str:
    return bucket_label(score)


def score_texts(model: PriorityModel, texts: list[str]) -> pd.DataFrame:
    cleaned = [" ".join(text.strip().split()) for text in texts if text and text.strip()]
    predictions = model.predict_priority(cleaned)
    rows: list[dict[str, Any]] = []
    for text, score in zip(cleaned, predictions, strict=True):
        rows.append(
            {
                "issue": text,
                "predicted_score": round(float(score), 4),
                "predicted_bucket": predicted_bucket(float(score)),
            }
        )
    return pd.DataFrame(rows)


def render_summary(summary: dict[str, Any] | None) -> None:
    st.sidebar.header("Model")
    if not summary:
        st.sidebar.warning("Training summary not found.")
        return

    metrics = summary.get("validation_metrics", {})
    st.sidebar.write(f"Rows: `{summary.get('row_count', 'unknown')}`")
    st.sidebar.write(f"Embedding model: `{summary.get('embedding_model', 'unknown')}`")
    st.sidebar.metric("Validation MAE", metrics.get("mae", "n/a"))
    st.sidebar.metric("Validation RMSE", metrics.get("rmse", "n/a"))
    st.sidebar.metric("Validation Spearman", metrics.get("spearman", "n/a"))

    st.sidebar.subheader("Bucket Ranges")
    for label, lower, upper in BUCKET_RANGES:
        st.sidebar.write(f"`{label}` = {lower} to {upper}")

    distributions = summary.get("validation_distribution", [])
    if distributions:
        st.sidebar.subheader("Validation Mix")
        distribution_df = pd.DataFrame(distributions)
        distribution_df["share"] = (distribution_df["share"] * 100).round(2).astype(str) + "%"
        st.sidebar.dataframe(distribution_df, use_container_width=True, hide_index=True)


def main() -> None:
    args = parse_streamlit_args()
    model_path = Path(args.model_bundle)
    summary_path = Path(args.summary)

    st.set_page_config(page_title="Issue Priority Scorer", page_icon=":bar_chart:", layout="wide")
    st.title("Issue Priority Scorer")
    st.caption("Interactive scoring UI for the bucket-stratified MiniLM + Ridge model.")

    if not model_path.exists():
        st.error(f"Model bundle not found: {model_path}")
        st.stop()

    summary = load_summary(str(summary_path))
    render_summary(summary)

    model = load_model(str(model_path))

    tab_single, tab_batch = st.tabs(["Single Issue", "Batch Scoring"])

    with tab_single:
        issue_text = st.text_area(
            "Issue description",
            height=160,
            placeholder="Paste one issue narrative here...",
        )
        if st.button("Predict score", type="primary"):
            if not issue_text.strip():
                st.warning("Enter an issue description first.")
            else:
                result_df = score_texts(model, [issue_text])
                row = result_df.iloc[0]
                col1, col2 = st.columns(2)
                col1.metric("Predicted score", row["predicted_score"])
                col2.metric("Predicted bucket", row["predicted_bucket"])
                st.dataframe(result_df, use_container_width=True, hide_index=True)

    with tab_batch:
        st.write("Paste multiple issues, one per line, or upload a CSV.")
        multiline_input = st.text_area(
            "Batch input",
            height=220,
            placeholder="One issue per line...",
            key="batch_text_input",
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        batch_df: pd.DataFrame | None = None
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            if "issue" in uploaded_df.columns:
                batch_df = uploaded_df.rename(columns={"issue": "issue"})
            elif "issue_description" in uploaded_df.columns:
                batch_df = uploaded_df.rename(columns={"issue_description": "issue"})
            else:
                st.error("CSV must contain either an 'issue' or 'issue_description' column.")

        if st.button("Run batch scoring"):
            rows: list[str] = []
            if multiline_input.strip():
                rows.extend(line for line in multiline_input.splitlines() if line.strip())
            if batch_df is not None:
                rows.extend(batch_df["issue"].astype(str).tolist())

            if not rows:
                st.warning("Provide batch text or upload a compatible CSV.")
            else:
                result_df = score_texts(model, rows)
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Download predictions CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="issue_priority_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
