param(
  [switch]$Apply,
  [switch]$DryRun,
  [string]$Base = $env:CODEX_RPC_BASE,
  [string]$Session = $(if ($env:AGENT_BROWSER_SESSION) { $env:AGENT_BROWSER_SESSION } else { "tencent-arena" }),
  [string]$AgentBrowser = $env:AGENT_BROWSER,
  [string]$Source = ".",
  [string]$Dest = ".",
  [int]$ChunkSize = 4096,
  [switch]$IncludeIgnored,
  [switch]$PyCompile,
  [string]$Token = $env:CODEX_RPC_TOKEN,
  [string]$AdminToken = $env:CODEX_RPC_ADMIN_TOKEN,
  [string]$Bash = "bash"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BridgeDir = Split-Path -Parent $ScriptDir
$SyncScript = Join-Path $BridgeDir "sync_repo_to_container.sh"

if (-not (Get-Command $Bash -ErrorAction SilentlyContinue)) {
  throw "cannot find bash. Install Git for Windows or WSL, or pass -Bash <path-to-bash.exe>."
}

if ($Apply) {
  if (-not $Token) {
    throw "CODEX_RPC_TOKEN is required with -Apply. Pass -Token or set CODEX_RPC_TOKEN."
  }
  if (-not $AdminToken) {
    throw "CODEX_RPC_ADMIN_TOKEN is required with -Apply. Pass -AdminToken or set CODEX_RPC_ADMIN_TOKEN."
  }
  $env:CODEX_RPC_TOKEN = $Token
  $env:CODEX_RPC_ADMIN_TOKEN = $AdminToken
}

if ($Base) {
  $env:CODEX_RPC_BASE = $Base
}
if ($AgentBrowser) {
  $env:AGENT_BROWSER = $AgentBrowser
}
$env:AGENT_BROWSER_SESSION = $Session
$env:AGENT_BROWSER_SESSION_NAME = $Session

$bashArgs = @($SyncScript)

if ($Apply) {
  $bashArgs += "--apply"
} elseif ($DryRun) {
  $bashArgs += "--dry-run"
} else {
  $bashArgs += "--dry-run"
}

$bashArgs += @("--source", $Source)
$bashArgs += @("--dest", $Dest)
$bashArgs += @("--chunk-size", "$ChunkSize")

if ($Base) {
  $bashArgs += @("--base", $Base)
}
if ($Session) {
  $bashArgs += @("--session", $Session)
}
if ($AgentBrowser) {
  $bashArgs += @("--agent-browser", $AgentBrowser)
}
if ($IncludeIgnored) {
  $bashArgs += "--include-ignored"
}
if ($PyCompile) {
  $bashArgs += "--py-compile"
}

& $Bash @bashArgs
exit $LASTEXITCODE
