# Issue Priority Scorer

This project scores free-form issue descriptions on a `0-100` priority scale and maps each score into a bucket:

- `0-40`
- `41-60`
- `61-80`
- `81-100`

The project now supports two ways to use the scorer:

- local Streamlit app for the current Windows workflow
- Vercel-ready Flask app for deployment

## What your lead can do

Once either app is running, you can:

- open the website in a browser
- paste any issue description into the text box
- click `Predict score`
- see the predicted numeric score and bucket immediately
- paste multiple issues at once or upload a CSV in the batch tab
- download batch predictions as a CSV

## Current default model

The app is currently configured to use:

- embedding model: `Alibaba-NLP/gte-modernbert-base`
- trained artifact folder: `artifacts/stratified_gte_modernbert_base_3200_gpu`
- validation MAE: `8.4056`
- validation RMSE: `11.0749`
- validation Spearman: `0.9048`

The app expects these files:

- `artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl`
- `artifacts/stratified_gte_modernbert_base_3200_gpu/training_summary.json`

## Fastest way to run locally

Use PowerShell from the project root:

```powershell
.\run_issue_priority_app.ps1
```

Then open:

```text
http://localhost:8501
```

This launcher will:

1. install app dependencies from `requirements-issue-priority-app.txt`
2. check whether `final_model.pkl` already exists
3. train a model if needed
4. start the Streamlit website

## Vercel deployment

This repo now includes a Vercel-compatible Flask entrypoint in `app.py`.

### Files used by Vercel

- `app.py`
- `templates/index.html`
- `issue_priority_inference.py`
- `requirements.txt`
- `.vercelignore`
- `.python-version`

### Before you deploy

Vercel deploys from Git, so the trained model file must be committed to the repo.

Required artifact:

- `artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl`

The default `.gitignore` now allows that specific model file to be committed.

### Deploy steps

1. Make sure these files exist locally:
   - `artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl`
   - `artifacts/stratified_gte_modernbert_base_3200_gpu/training_summary.json`
2. Commit and push the repo to GitHub.
3. Import the repo into Vercel.
4. Leave the framework detection on auto. Vercel will detect the root `app.py` as a Flask app.
5. Deploy.

Optional environment variables:

- `MODEL_BUNDLE_PATH`
- `SUMMARY_PATH`

If you do not set them, the deployed app uses the default artifact paths above.

No `vercel.json` is required for this setup. Recent Vercel Python support detects the root Flask entrypoint automatically.

### Vercel routes

- `/`: web UI
- `/health`: health check
- `/api/predict`: JSON prediction endpoint
- `/api/batch`: JSON batch endpoint

### Important runtime note

The deployment shape is now compatible with Vercel, but inference still depends on the `sentence-transformers` model. That means cold starts can be slow, and the first request may need internet access to fetch the embedding model weights unless they are already cached in the deployment environment.

## Important note

On another machine, the website will only start immediately if one of these is true:

1. you also give them the trained artifact folder
2. they run the app with retraining enabled so a new model is created locally

If your lead is using this same machine and the artifact already exists, the simple launcher command is enough.

## Recommended setup on another machine

### 1. Install Python

Use Python `3.11+` or `3.12+`.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-issue-priority-app.txt
```

### 4. Start the app

If `artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl` already exists:

```powershell
python -m streamlit run issue_priority_streamlit_app.py -- `
  --model-bundle artifacts/stratified_gte_modernbert_base_3200_gpu/final_model.pkl `
  --summary artifacts/stratified_gte_modernbert_base_3200_gpu/training_summary.json
```

If the model file does not exist, run:

```powershell
.\run_issue_priority_app.ps1 -Retrain
```

Then open `http://localhost:8501`.

## How to use the website

### Single issue scoring

1. Open the `Single Issue` tab.
2. Paste one issue description.
3. Click `Predict score`.
4. Review:
   - `Predicted score`
   - `Predicted bucket`

### Batch scoring

Use the `Batch Scoring` tab when you want to test several items.

You can either:

- paste one issue per line
- upload a CSV with an `issue` column
- upload a CSV with an `issue_description` column

After scoring, use `Download predictions CSV` to save the results.

## Retraining the model

If you want to rebuild the model from the dataset:

```powershell
python train_issue_priority_stratified.py train `
  --dataset issue_priority_dataset.csv `
  --output-dir artifacts/stratified_gte_modernbert_base_3200_gpu `
  --embedding-model Alibaba-NLP/gte-modernbert-base `
  --device auto
```

Artifacts written by training:

- `training_summary.json`
- `validation_predictions.csv`
- `full_predictions.csv`
- `final_model.pkl`

## Key project files

- `app.py`: Flask website and API for Vercel deployment
- `issue_priority_inference.py`: shared inference loader for Flask and Streamlit
- `issue_priority_streamlit_app.py`: local Streamlit website for interactive scoring
- `run_issue_priority_app.ps1`: PowerShell launcher for install, optional retrain, and app startup
- `train_issue_priority_stratified.py`: training and CLI prediction pipeline
- `issue_priority_dataset.csv`: labeled training dataset
- `artifacts/stratified_gte_modernbert_base_3200_gpu/`: current default trained artifacts

## Troubleshooting

### `Python was not found`

Your shell may not have Python on `PATH`. Either install Python properly or edit `run_issue_priority_app.ps1` so `$PythonExe` points to the correct interpreter.

### `Model bundle not found`

Run:

```powershell
.\run_issue_priority_app.ps1 -Retrain
```

### Streamlit opens but scoring fails

Make sure:

- dependencies installed successfully
- internet access is available the first time the embedding model is downloaded
- the artifact paths passed to the app are correct
