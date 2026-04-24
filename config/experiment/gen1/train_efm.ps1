param(
    [ValidateSet("tiny", "small", "base")]
    [string]$Size = "base",
    [string]$Gpu = "0",
    [int]$TrainBatchSize = 1,
    [int]$EvalBatchSize = 1,
    [int]$TrainWorkers = 0,
    [int]$EvalWorkers = 0
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $RepoRoot

python train.py `
    model=rnndet `
    hardware.gpus=$Gpu `
    dataset=gen1x0.01_ss `
    +experiment/gen1="efm_$Size.yaml" `
    batch_size.train=$TrainBatchSize `
    batch_size.eval=$EvalBatchSize `
    hardware.num_workers.train=$TrainWorkers `
    hardware.num_workers.eval=$EvalWorkers
