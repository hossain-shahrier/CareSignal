# Sync CareSignal to a Hugging Face Docker Space and push.
# Usage:
#   .\scripts\push_hf_space.ps1
#   .\scripts\push_hf_space.ps1 -SpaceRepo "beardmoose/CareSignal"
#
# Requires: git, and HF write token as git password when prompted.

param(
    [string]$SpaceRepo = "beardmoose/CareSignal",
    [string]$CloneDir = "$env:TEMP\caresignal-hf-space"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Invoke-Git {
    param([string[]]$GitArgs)
    $prevNative = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    try {
        $out = & git.exe @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed: $out"
        }
        return $out
    } finally {
        $PSNativeCommandUseErrorActionPreference = $prevNative
    }
}
$Root = Split-Path $PSScriptRoot -Parent

$items = @(
    "pyproject.toml",
    "Dockerfile",
    ".dockerignore",
    "config",
    "src",
    "scripts",
    "data/reference",
    "artifacts/manifest.json",
    "artifacts/model.joblib.b64"
)

foreach ($item in $items) {
    if (-not (Test-Path (Join-Path $Root $item))) {
        throw "Missing required path for HF deploy: $item"
    }
}

if (-not (Test-Path (Join-Path $Root "config/train.docker.yaml"))) {
    throw "Missing config/train.docker.yaml"
}

$SpaceUrl = "https://huggingface.co/spaces/$SpaceRepo"
Write-Host "Target Space: $SpaceUrl"

if (-not (Test-Path $CloneDir)) {
    Write-Host "Cloning Space repo..."
    git clone $SpaceUrl $CloneDir
}

Push-Location $CloneDir
try {
    Write-Host "Syncing clone to origin/main..."
    Invoke-Git -GitArgs @("fetch", "origin") | Out-Null
    Invoke-Git -GitArgs @("checkout", "main") | Out-Null
    Invoke-Git -GitArgs @("reset", "--hard", "origin/main") | Out-Null
} finally {
    Pop-Location
}

# Clear old app files (keep .git)
Get-ChildItem $CloneDir -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

function Copy-TreeFiltered {
    param([string]$Source, [string]$Dest)
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    robocopy $Source $Dest /E /XD __pycache__ .pytest_cache .ruff_cache .venv venv .git /XF *.pyc *.pyo /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed copying $Source" }
}

foreach ($item in $items) {
    $src = Join-Path $Root $item
    $dest = Join-Path $CloneDir $item
    $destParent = Split-Path $dest -Parent
    if (-not (Test-Path $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }
    if (Test-Path $src -PathType Container) {
        Copy-TreeFiltered $src $dest
    } else {
        Copy-Item $src $dest -Force
    }
}

Copy-Item (Join-Path $Root "config/train.docker.yaml") (Join-Path $CloneDir "config/train.docker.yaml") -Force
Copy-Item (Join-Path $Root "deploy\huggingface\README.md") (Join-Path $CloneDir "README.md") -Force

@"
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ruff_cache/
artifacts/*.joblib
"@ | Set-Content (Join-Path $CloneDir ".gitignore") -Encoding utf8

Push-Location $CloneDir
try {
    Invoke-Git -GitArgs @("add", "-A") | Out-Null
    $status = git status --porcelain
    if (-not $status) {
        Write-Host "No changes to push."
        exit 0
    }

    Invoke-Git -GitArgs @("commit", "-m", "Fix artifacts path for Docker/Hugging Face") | Out-Null
    Invoke-Git -GitArgs @("push", "origin", "main") | Out-Null
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Pushed successfully. Space will build at: $SpaceUrl"
