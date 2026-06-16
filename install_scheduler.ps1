# Register a daily Windows Task that refreshes + publishes the dashboard.
# Run once:  powershell -ExecutionPolicy Bypass -File install_scheduler.ps1
# Remove:    Unregister-ScheduledTask -TaskName "RobloxIdeaFinderRefresh" -Confirm:$false

$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$task = "RobloxIdeaFinderRefresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\update.ps1`"" `
    -WorkingDirectory $dir
$trigger  = New-ScheduledTaskTrigger -Daily -At 9am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Settings $settings `
    -Description "Re-harvest Roblox data and publish the idea-finder dashboard (runs on your residential IP)." `
    -Force | Out-Null

Write-Host "Scheduled '$task' to run daily at 9:00 AM (runs when you're next online if the PC was off)."
Write-Host "Run it now to test:  Start-ScheduledTask -TaskName $task"
