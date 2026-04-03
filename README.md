# Issue Priority Scorer

This project scores free-form issue descriptions on a `0-100` priority scale and maps each score into a bucket:

- `0-40`
- `41-60`
- `61-80`
- `81-100`

The project now supports two ways to use the scorer:

- local Flask app for the simplest browser-based workflow
- local Streamlit app for the older interactive workflow

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

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- trained artifact folder: `artifacts/stratified_minilm_3200`
- validation MAE: `10.0`
- validation RMSE: `12.9545`
- validation Spearman: `0.8859`

The app expects these files:

- `artifacts/stratified_minilm_3200/final_model.pkl`
- `artifacts/stratified_minilm_3200/training_summary.json`

## Fastest way to run locally

Use a normal Windows Python installation from the project root. The simplest local path is the Flask app in `app.py`.

### 1. Create a clean virtual environment

```powershell
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
```

If `py` is not available, replace it with the full path to your Python executable.

Important:

- use a standard Windows Python build from `python.org`
- do not reuse an MSYS or Git Bash Python virtualenv for this app

### 2. Install runtime dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the local web app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Useful routes:

- `/`: web UI
- `/health`: health check
- `/api/predict`: single prediction API
- `/api/batch`: batch prediction API

### 4. Stop the app

Press `Ctrl+C` in the terminal running `python app.py`.

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

- `artifacts/stratified_minilm_3200/final_model.pkl`

The default `.gitignore` now allows that specific model file to be committed while still ignoring `.env`.

### Deploy steps

1. Make sure these files exist locally:
   - `artifacts/stratified_minilm_3200/final_model.pkl`
   - `artifacts/stratified_minilm_3200/training_summary.json`
2. Commit and push the repo to GitHub.
3. Import the repo into Vercel.
4. Leave the framework detection on auto. Vercel will detect the root `app.py` as a Flask app.
5. Deploy.

Optional environment variables:

- `MODEL_BUNDLE_PATH`
- `SUMMARY_PATH`

If you do not set them, the deployed app uses the default artifact paths above.

`.env` is not used by this Flask deploy path and should stay out of Git. If you ever need runtime secrets on Vercel, add them in the Vercel project settings instead.

No `vercel.json` is required for this setup. Recent Vercel Flask support detects the root `app.py` automatically.

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

Use a standard Windows Python install such as Python `3.12` or `3.13`.

### 2. Create and activate a virtual environment

```powershell
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
```

If you already created `.venv` from MSYS, Git Bash, or another non-standard Python build, create a fresh venv instead of reusing it.

### 3. Install dependencies

For the Flask app:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional: if you specifically want the Streamlit workflow instead, install:

```powershell
python -m pip install -r requirements-issue-priority-app.txt
```

### 4. Start the app

Recommended local path:

```powershell
python app.py
```

Then open `http://127.0.0.1:5000`.

Optional Streamlit path:

```powershell
python -m streamlit run issue_priority_streamlit_app.py -- `
  --model-bundle artifacts/stratified_minilm_3200/final_model.pkl `
  --summary artifacts/stratified_minilm_3200/training_summary.json
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
  --output-dir artifacts/stratified_minilm_3200 `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 `
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
- `run_issue_priority_app.ps1`: older PowerShell launcher for the Streamlit workflow
- `train_issue_priority_stratified.py`: training and CLI prediction pipeline
- `issue_priority_dataset.csv`: labeled training dataset
- `artifacts/stratified_minilm_3200/`: current default trained artifacts

## Troubleshooting

### `Python was not found`

Your shell may not have Python on `PATH`. Use `py -3.13`, install Python properly, or call the interpreter by full path.

### `Model bundle not found`

Make sure this file exists:

```powershell
artifacts\stratified_minilm_3200\final_model.pkl
```

If it is missing, retrain the model or copy the trained artifact folder into the project.

### Local app starts but scoring fails

Make sure:

- dependencies installed successfully
- internet access is available the first time the embedding model is downloaded
- the artifact files under `artifacts/stratified_minilm_3200/` are present

### Existing `.venv` behaves strangely on Windows

This project may fail to install correctly if the virtual environment was created from MSYS or another non-standard Python distribution. In that case, create a fresh Windows venv and install dependencies there:

```powershell
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
