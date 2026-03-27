param(
    [string]$PythonExe = "C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe",
    [string]$Dataset = "issue_priority_dataset.csv",
    [string]$OutputDir = "artifacts\stratified_gte_modernbert_base_3200_gpu",
    [string]$EmbeddingModel = "Alibaba-NLP/gte-modernbert-base",
    [string]$Device = "auto",
    [int]$Port = 8501,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$modelBundle = Join-Path $OutputDir "final_model.pkl"
$summaryPath = Join-Path $OutputDir "training_summary.json"

Write-Host "Using Python: $PythonExe"
Write-Host "Output directory: $OutputDir"

Write-Host "Installing app dependencies if needed..."
& $PythonExe -m pip install -r "requirements-issue-priority-app.txt"

if ($Retrain -or -not (Test-Path $modelBundle)) {
    Write-Host "Training model..."
    & $PythonExe "train_issue_priority_stratified.py" train `
        --dataset $Dataset `
        --output-dir $OutputDir `
        --embedding-model $EmbeddingModel `
        --device $Device
}

if (-not (Test-Path $modelBundle)) {
    throw "Model bundle missing after training step: $modelBundle"
}

Write-Host "Launching Streamlit app on port $Port..."
& $PythonExe -m streamlit run "issue_priority_streamlit_app.py" `
    --server.port $Port `
    --server.headless true `
    --server.fileWatcherType none `
    -- `
    --model-bundle $modelBundle `
    --summary $summaryPath
