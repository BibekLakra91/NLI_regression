# Issue Priority Scorer

This project scores free-form issue descriptions on a `0-100` priority scale and maps each score into a bucket:

- `0-40`
- `41-60`
- `61-80`
- `81-100`

The easiest way for a reviewer or team lead to validate the model is to run the Streamlit web app, paste issue text, and compare the predicted score with their own judgment.

## What your lead can do

Once the app is running, you can:

- open the website locally in a browser
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

## Fastest way to run the website

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

## Important note 

The trained model file `final_model.pkl` is intentionally ignored by Git, so you clones this repo on another machine, the website will only start immediately if one of these is true:

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

- `issue_priority_streamlit_app.py`: Streamlit website for interactive scoring
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
