# Sync CareSignal to a Hugging Face Docker Space and push.
# Usage:
#   .\scripts\push_hf_space.ps1
#   .\scripts\push_hf_space.ps1 -SpaceRepo "beardmoose/CareSignal"
#
# Requires: git, and `hf auth login` OR git credentials for huggingface.co

param(
    [string]$SpaceRepo = "beardmoose/CareSignal",
    [string]$CloneDir = "$env:TEMP\caresignal-hf-space"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

$items = @(
    "pyproject.toml",
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "config",
    "src",
    "scripts",
    "data/reference",
    "artifacts/manifest.json",
    "artifacts/model.joblib.b64",
    "config/train.docker.yaml"
)
# HF rejects binary model.joblib via git — ship model.joblib.b64 (text) instead.

foreach ($item in $items) {
    if (-not (Test-Path (Join-Path $Root $item))) {
        throw "Missing required path for HF deploy: $item"
    }
}

$SpaceUrl = "https://huggingface.co/spaces/$SpaceRepo"
Write-Host "Target Space: $SpaceUrl"

if (-not (Test-Path $CloneDir)) {
    Write-Host "Cloning Space repo..."
    git clone $SpaceUrl $CloneDir
} else {
    Write-Host "Updating existing clone at $CloneDir"
    Push-Location $CloneDir
    git pull --rebase
    Pop-Location
}

# Clear old app files (keep .git) — avoids nested config/config from incremental copies
Get-ChildItem $CloneDir -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

foreach ($item in $items) {
    $src = Join-Path $Root $item
    $dest = Join-Path $CloneDir $item
    $destParent = Split-Path $dest -Parent
    if (-not (Test-Path $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }
    if (Test-Path $src -PathType Container) {
        Copy-Item $src $dest -Recurse -Force
    } else {
        Copy-Item $src $dest -Force
    }
}

# HF Space card README (with YAML frontmatter)
Copy-Item (Join-Path $Root "deploy\huggingface\README.md") (Join-Path $CloneDir "README.md") -Force

Push-Location $CloneDir
git add -A
$status = git status --porcelain
if (-not $status) {
    Write-Host "No changes to push."
    Pop-Location
    exit 0
}

git commit -m "Deploy CareSignal app with UI and model bundle"
git push
Pop-Location

Write-Host ""
Write-Host "Pushed. Space will build at: $SpaceUrl"
Write-Host "Open the URL in ~2-5 minutes once the build finishes."
