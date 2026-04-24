param(
    [ValidateSet("tiny", "small", "base")]
    [string]$Size = "base",
    [string]$Gpu = "0",
    [int]$TrainBatchSize = 1,
    [int]$EvalBatchSize = 1,
    [int]$TrainWorkers = 0,
    [int]$EvalWorkers = 0,
    [bool]$SuppressKnownWarnings = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $RepoRoot

$PythonWarningArgs = @()
$PreviousPythonWarnings = $env:PYTHONWARNINGS
if ($SuppressKnownWarnings) {
    $WarningFilters = @(
        "ignore:pkg_resources is deprecated as an API:UserWarning",
        "ignore:The dataloader.*does not have many workers:UserWarning",
        "ignore:torch.meshgrid:UserWarning",
        "ignore::FutureWarning",
        "ignore:Detected call of ``lr_scheduler.step:UserWarning"
    )
    $env:PYTHONWARNINGS = $WarningFilters -join ","
    foreach ($Filter in $WarningFilters) {
        $PythonWarningArgs += @("-W", $Filter)
    }
}

try {
    python @PythonWarningArgs train.py `
        model=rnndet `
        hardware.gpus=$Gpu `
        dataset=gen4x0.01_ss `
        +experiment/gen4="efm_$Size.yaml" `
        batch_size.train=$TrainBatchSize `
        batch_size.eval=$EvalBatchSize `
        hardware.num_workers.train=$TrainWorkers `
        hardware.num_workers.eval=$EvalWorkers
}
finally {
    $env:PYTHONWARNINGS = $PreviousPythonWarnings
}
