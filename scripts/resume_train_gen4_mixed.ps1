param(
    [int]$RestartDelaySeconds = 10
)

$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$trainArgs = @(
    "train.py",
    "model=rnndet",
    "hardware.gpus=0",
    "dataset=gen4x0.01_ss",
    "+experiment/gen4=tiny.yaml",
    "batch_size.train=2",
    "batch_size.eval=1",
    "hardware.num_workers.train=2",
    "hardware.num_workers.eval=0"
)

while ($true) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting training..." -ForegroundColor Cyan
    & python @trainArgs
    $exitCode = $LASTEXITCODE

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training exited with code $exitCode." -ForegroundColor Yellow
    Write-Host "Waiting $RestartDelaySeconds seconds before restart..." -ForegroundColor Yellow
    Start-Sleep -Seconds $RestartDelaySeconds
}
