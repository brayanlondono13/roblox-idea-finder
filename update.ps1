# Daily Roblox dashboard refresh — run from YOUR machine (residential IP).
# Roblox blocks most datacenter IPs, so this is the reliable updater (not CI).
# It re-harvests live data, rebuilds the dashboard, and pushes — GitHub Pages then
# serves the new version within ~1 minute.
#
# Run once:        powershell -ExecutionPolicy Bypass -File update.ps1
# Schedule daily:  powershell -ExecutionPolicy Bypass -File install_scheduler.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "[1/4] Harvesting live Roblox data..."
python roblox_research.py --sleep 0.5 harvest --pages 1 --out data/corpus.json

Write-Host "[2/4] Computing combos..."
python roblox_research.py combos --corpus data/corpus.json --json data/combos.json

Write-Host "[3/4] Rebuilding dashboard..."
python roblox_viz.py --corpus data/corpus.json --out docs/index.html

Write-Host "[4/4] Publishing..."
git add -A
if (git status --porcelain) {
    git commit -m "chore: daily Roblox data refresh" | Out-Null
    git push
    Write-Host "Done — pushed. Live site updates in ~1 min: https://brayanlondono13.github.io/roblox-idea-finder/"
} else {
    Write-Host "No changes to publish."
}
