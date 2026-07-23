param(
    [string]$TaskName = "CodexTabPFNColabRecoverySupervisor"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "launch_m5_colab_recovery_supervisor.ps1"
$lockPath = Join-Path $repoRoot "data\processed\m5_tabpfn_recovery_supervisor.lock"

function Get-LiveLockOwner {
    if (-not (Test-Path -LiteralPath $lockPath)) {
        return $null
    }
    try {
        $ownerPid = [int]((Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json).pid)
    }
    catch {
        $ownerPid = 0
    }
    if ($ownerPid -gt 0 -and (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)) {
        return $ownerPid
    }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    return $null
}

$liveOwner = Get-LiveLockOwner
if ($null -ne $liveOwner) {
    Write-Output "supervisor_already_live=true pid=$liveOwner"
    exit 0
}

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Missing committed Colab-only launcher: $launcher"
}

$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask -and $existingTask.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName
}

$argument = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$launcher`""
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argument `
    -WorkingDirectory $repoRoot
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

for ($sample = 1; $sample -le 15; $sample++) {
    Start-Sleep -Seconds 2
    $liveOwner = Get-LiveLockOwner
    if ($null -ne $liveOwner) {
        Write-Output "supervisor_started=true pid=$liveOwner task=$TaskName"
        exit 0
    }
}

throw "Scheduled recovery supervisor did not acquire its singleton lock"
