param(
  [string]$IdeId = $env:CODEX_TENCENT_IDE_ID,
  [string]$DomainId = $(if ($env:CODEX_TENCENT_DOMAIN_ID) { $env:CODEX_TENCENT_DOMAIN_ID } else { "447" }),
  [string]$TeamId = $(if ($env:CODEX_TENCENT_TEAM_ID) { $env:CODEX_TENCENT_TEAM_ID } else { "11585" }),
  [string]$IdeUrl = "",
  [string]$HealthUrl = "",
  [string]$StatusUrl = $(if ($env:CODEX_TENCENT_STATUS_URL) { $env:CODEX_TENCENT_STATUS_URL } else { "https://tencentarena.com/api/v5/Competition/GetWebIDE" }),
  [int]$IntervalSeconds = $(if ($env:CODEX_KEEPALIVE_INTERVAL) { [int]$env:CODEX_KEEPALIVE_INTERVAL } else { 180 }),
  [string]$AgentBrowser = $(if ($env:AGENT_BROWSER) { $env:AGENT_BROWSER } else { "agent-browser" }),
  [string]$Session = $(if ($env:AGENT_BROWSER_SESSION) { $env:AGENT_BROWSER_SESSION } else { "tencent-arena" }),
  [switch]$Status,
  [switch]$Stop,
  [switch]$Restart,
  [switch]$InteractiveRecovery,
  [switch]$NoSyntheticActivity,
  [switch]$SafeClickKeepalive,
  [int]$SafeClickX = 8,
  [int]$SafeClickY = 8,
  [switch]$TerminalKeepalive
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BridgeDir = Split-Path -Parent $ScriptDir
$RepoRoot = Resolve-Path (Join-Path $BridgeDir "..\..")
$StateDir = Join-Path $RepoRoot ".codex_rpc"
$PidFile = Join-Path $StateDir "keepalive_tencent_arena.pid"
$LogFile = Join-Path $StateDir "keepalive_tencent_arena.log"
$ErrorLogFile = Join-Path $StateDir "keepalive_tencent_arena.err.log"
$KeepaliveScript = Join-Path $BridgeDir "keepalive_tencent_arena.ps1"

if (-not $IdeId) {
  $IdeId = "11428"
}
if (-not $IdeUrl) {
  $IdeUrl = "https://tencentarena.com/p/common/competition/ide/$DomainId/$TeamId/$IdeId"
}
if (-not $HealthUrl) {
  $HealthUrl = "https://tencentarena.com/p5/ide/$IdeId/proxy/8765/api/health"
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Get-KeepaliveProcess {
  if (-not (Test-Path $PidFile)) {
    return $null
  }
  $pidText = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (-not $pidText) {
    return $null
  }
  return Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
}

function Show-Status {
  $proc = Get-KeepaliveProcess
  Write-Host "pid_file=$PidFile"
  Write-Host "log_file=$LogFile"
  Write-Host "error_log_file=$ErrorLogFile"
  Write-Host "ide_url=$IdeUrl"
  Write-Host "health_url=$HealthUrl"
  Write-Host "status_url=$StatusUrl"
  Write-Host "interactive_recovery=$InteractiveRecovery"
  Write-Host "terminal_keepalive=$TerminalKeepalive"
  Write-Host "synthetic_activity=$(-not $NoSyntheticActivity)"
  Write-Host "safe_click_keepalive=$SafeClickKeepalive"
  if ($proc) {
    Write-Host "status=alive"
    Write-Host "pid=$($proc.Id)"
  } else {
    Write-Host "status=not_alive"
    if (Test-Path $PidFile) {
      Write-Host "last_pid=$((Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1))"
    }
  }
  if (Test-Path $LogFile) {
    Write-Host "--- recent log ---"
    Get-Content $LogFile -Tail 40
  }
  if (Test-Path $ErrorLogFile) {
    $err = Get-Content $ErrorLogFile -Tail 20
    if ($err) {
      Write-Host "--- recent error log ---"
      $err
    }
  }
}

function Stop-Keepalive {
  $proc = Get-KeepaliveProcess
  if ($proc) {
    Stop-Process -Id $proc.Id
    Write-Host "stopped keepalive pid $($proc.Id)"
  } else {
    Write-Host "keepalive is not running"
  }
}

function Start-Keepalive {
  $existing = Get-KeepaliveProcess
  if ($existing) {
    Write-Host "keepalive already running: pid=$($existing.Id)"
    Show-Status
    return
  }

  $powershellExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
  if (-not $powershellExe) {
    $powershellExe = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
  }
  if (-not $powershellExe) {
    throw "cannot find pwsh or powershell.exe"
  }

  $processArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $KeepaliveScript,
    "-IntervalSeconds", "$IntervalSeconds",
    "-AgentBrowser", $AgentBrowser,
    "-Session", $Session,
    "-IdeUrl", $IdeUrl,
    "-HealthUrl", $HealthUrl,
    "-StatusUrl", $StatusUrl
  )
  if ($InteractiveRecovery) {
    $processArgs += "-InteractiveRecovery"
  }
  if ($NoSyntheticActivity) {
    $processArgs += "-NoSyntheticActivity"
  }
  if ($SafeClickKeepalive) {
    $processArgs += @("-SafeClickKeepalive", "-SafeClickX", "$SafeClickX", "-SafeClickY", "$SafeClickY")
  }
  if ($TerminalKeepalive) {
    $processArgs += "-TerminalKeepalive"
  }

  "" | Set-Content $LogFile
  "" | Set-Content $ErrorLogFile

  $proc = Start-Process `
    -FilePath $powershellExe `
    -ArgumentList $processArgs `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrorLogFile `
    -PassThru

  Set-Content -Path $PidFile -Value $proc.Id
  Start-Sleep -Seconds 3

  if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
    Write-Host "started keepalive pid $($proc.Id)"
    Show-Status
  } else {
    Write-Error "keepalive exited immediately; see log: $LogFile"
    Show-Status
    exit 1
  }
}

if ($Status) {
  Show-Status
} elseif ($Stop) {
  Stop-Keepalive
} elseif ($Restart) {
  Stop-Keepalive
  Start-Keepalive
} else {
  Start-Keepalive
}
