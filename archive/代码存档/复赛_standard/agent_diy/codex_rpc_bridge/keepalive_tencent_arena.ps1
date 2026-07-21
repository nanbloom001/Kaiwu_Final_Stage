param(
  [int]$IntervalSeconds = 180,
  [string]$AgentBrowser = "C:\Users\lenovo\AppData\Roaming\npm\agent-browser.cmd",
  [string]$Session = "tencent-arena",
  [string]$IdeUrl = "https://tencentarena.com/p/common/competition/ide/447/11585/11428",
  [string]$HealthUrl = "https://tencentarena.com/p5/ide/11428/proxy/8765/api/health",
  [string]$StatusUrl = "https://tencentarena.com/api/v5/Competition/GetWebIDE",
  [string]$KeepaliveCommand = 'printf ''[codex-keepalive] %s\n'' "$(date -Iseconds)"; pwd >/dev/null',
  [switch]$InteractiveRecovery,
  [switch]$NoSyntheticActivity,
  [switch]$SafeClickKeepalive,
  [int]$SafeClickX = 8,
  [int]$SafeClickY = 8,
  [switch]$TerminalKeepalive,
  [switch]$Once
)

$ErrorActionPreference = "Continue"
$env:AGENT_BROWSER_SESSION = $Session
$env:AGENT_BROWSER_SESSION_NAME = $Session

function Log($Message) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "[$ts] $Message"
}

function Run-AgentBrowser([string[]]$CommandArgs) {
  $maxAttempts = 4
  for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    try {
      $output = & $AgentBrowser @CommandArgs 2>&1
      if ($LASTEXITCODE -eq 0) {
        return $output
      }

      $text = ($output | Out-String).Trim()
      Log "agent-browser failed (attempt ${attempt}/${maxAttempts}): $text"
    } catch {
      Log "agent-browser exception (attempt ${attempt}/${maxAttempts}): $($_.Exception.Message)"
    }

    if ($attempt -lt $maxAttempts) {
      Start-Sleep -Seconds ([Math]::Min(2 * $attempt, 8))
    }
  }

  throw "agent-browser command failed: $($CommandArgs -join ' ')"
}

function Select-IdeTab {
  $tabOutput = Run-AgentBrowser @("tab", "list")
  $tabText = ($tabOutput | Out-String)
  $ideLine = $tabText -split "`r?`n" | Where-Object { $_ -match [Regex]::Escape($IdeUrl) } | Select-Object -First 1

  if ($ideLine -and $ideLine -match "\[(t\d+)\]") {
    $tabId = $Matches[1]
    Log "switching to IDE tab: $tabId"
    Run-AgentBrowser @("tab", $tabId) | Out-Null
    return
  }

  Log "IDE tab not found; opening IDE URL"
  Run-AgentBrowser @("open", $IdeUrl) | Out-Host
  Start-Sleep -Seconds 8
}

function Ensure-Page {
  Select-IdeTab

  $urlOutput = Run-AgentBrowser @("get", "url")
  $urlText = ($urlOutput | Out-String).Trim()
  Log "url: $urlText"

  if (-not $urlText -or $urlText -eq "about:blank" -or $urlText -ne $IdeUrl) {
    Log "opening IDE URL"
    Run-AgentBrowser @("open", $IdeUrl) | Out-Host
    Start-Sleep -Seconds 8
  }
}

function Handle-RecoveryButtons {
  Run-AgentBrowser @("snapshot", "-i") | Out-Null

  $js = @'
(() => {
  const iframe = document.querySelector('iframe');
  const docs = [document];
  if (iframe && iframe.contentDocument) docs.push(iframe.contentDocument);

  const actions = [
    { key: 'reconnect', label: '\u7acb\u5373\u91cd\u65b0\u8fde\u63a5', waitMs: 10000 },
    { key: 'reload_window', label: '\u91cd\u65b0\u52a0\u8f7d\u7a97\u53e3', waitMs: 15000 },
    { key: 'restart', label: '\u91cd\u542f', waitMs: 30000 },
  ];

  function normalize(value) {
    return String(value || '').replace(/\s+/g, '');
  }

  for (const action of actions) {
    const target = normalize(action.label);
    for (const doc of docs) {
      const nodes = Array.from(doc.querySelectorAll('button,a,[role="button"],[onclick]'));
      const found = nodes.find((node) => normalize(node.innerText || node.textContent || node.getAttribute('aria-label')).includes(target));
      if (found) {
        found.click();
        return { clicked: action.key, waitMs: action.waitMs };
      }
    }
  }

  return { clicked: null };
})()
'@
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($js))
  $clickOutput = Run-AgentBrowser @("eval", "-b", $b64)
  $clickText = ($clickOutput | Out-String).Trim()
  Log "recovery: $clickText"

  if ($clickText -match '"clicked":\s*"reconnect"') {
    Start-Sleep -Seconds 10
  } elseif ($clickText -match '"clicked":\s*"reload_window"') {
    Start-Sleep -Seconds 15
  } elseif ($clickText -match '"clicked":\s*"restart"') {
    Start-Sleep -Seconds 30
  }
}

function Send-TerminalKeepalive {
  $escaped = $KeepaliveCommand.Replace("\", "\\").Replace("`"", "\""")
  $js = @"
(() => {
  const iframe = document.querySelector('iframe');
  if (!iframe || !iframe.contentDocument) {
    return { ok: false, reason: 'iframe_not_ready' };
  }

  const doc = iframe.contentDocument;
  const terminalInput =
    doc.querySelector('textarea[aria-label*="终端"]') ||
    doc.querySelector('.terminal-wrapper textarea') ||
    doc.querySelector('.xterm-helper-textarea');

  if (!terminalInput) {
    return { ok: false, reason: 'terminal_input_not_found' };
  }

  terminalInput.focus();

  const text = "$escaped";
  terminalInput.value = '';
  terminalInput.dispatchEvent(new InputEvent('input', { bubbles: true, data: '' }));

  for (const ch of text) {
    terminalInput.dispatchEvent(new KeyboardEvent('keydown', { key: ch, bubbles: true }));
    terminalInput.value += ch;
    terminalInput.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch }));
    terminalInput.dispatchEvent(new KeyboardEvent('keyup', { key: ch, bubbles: true }));
  }

  terminalInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
  terminalInput.dispatchEvent(new KeyboardEvent('keypress', { key: 'Enter', code: 'Enter', bubbles: true }));
  terminalInput.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', code: 'Enter', bubbles: true }));

  return { ok: true, command: text };
})()
"@
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($js))
  $output = Run-AgentBrowser @("eval", "-b", $b64)
  $text = ($output | Out-String).Trim()
  Log "terminal-keepalive: $text"
  return $text
}

function Handle-RestartOnly {
  $js = @'
(() => {
  const iframe = document.querySelector('iframe');
  const docs = [document];
  if (iframe && iframe.contentDocument) docs.push(iframe.contentDocument);

  function normalize(value) {
    return String(value || '').replace(/\s+/g, '');
  }

  const target = normalize('\u91cd\u542f');
  for (const doc of docs) {
    const nodes = Array.from(doc.querySelectorAll('button,a,[role="button"],[onclick]'));
    const found = nodes.find((node) => normalize(node.innerText || node.textContent || node.getAttribute('aria-label')).includes(target));
    if (found) {
      found.click();
      return { clicked: 'restart', waitMs: 30000 };
    }
  }

  return { clicked: null };
})()
'@
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($js))
  $clickOutput = Run-AgentBrowser @("eval", "-b", $b64)
  $clickText = ($clickOutput | Out-String).Trim()
  Log "hard-recovery: $clickText"

  if ($clickText -match '"clicked":\s*"restart"') {
    Start-Sleep -Seconds 30
  }
}

function Test-Health {
  $js = "fetch('$HealthUrl').then(async r=>({status:r.status,text:await r.text()})).catch(e=>({error:String(e)}))"
  $healthOutput = Run-AgentBrowser @("eval", $js)
  $healthText = ($healthOutput | Out-String).Trim()
  Log "health: $healthText"
  return $healthText
}

function Invoke-BackgroundProbe {
  $healthB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($HealthUrl))
  $statusB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($StatusUrl))
  $syntheticActivity = if ($NoSyntheticActivity) { "false" } else { "true" }
  $js = @"
(() => {
  const decode = (value) => new TextDecoder().decode(Uint8Array.from(atob(value), c => c.charCodeAt(0)));
  const statusUrl = decode("$statusB64");
  const healthUrl = decode("$healthB64");
  const syntheticActivity = $syntheticActivity;
  const activity = { enabled: syntheticActivity, dispatched: 0, errors: [] };
  const dispatchActivity = (targetWindow, targetDocument, label) => {
    if (!targetWindow || !targetDocument) return;
    try {
      const event = new targetWindow.MouseEvent("mousemove", {
        bubbles: true,
        cancelable: true,
        clientX: 1 + Math.floor(Math.random() * 20),
        clientY: 1 + Math.floor(Math.random() * 20),
        movementX: 1,
        movementY: 0,
      });
      targetWindow.dispatchEvent(event);
      targetDocument.dispatchEvent(event);
      activity.dispatched += 2;
    } catch (error) {
      activity.errors.push(`${label}: ${String(error)}`);
    }
  };
  if (syntheticActivity) {
    dispatchActivity(window, document, "top");
    for (const iframe of Array.from(document.querySelectorAll("iframe"))) {
      try {
        dispatchActivity(iframe.contentWindow, iframe.contentDocument, iframe.src || "iframe");
      } catch (error) {
        activity.errors.push(`iframe: ${String(error)}`);
      }
    }
  }
  const getText = async (url) => {
    try {
      const response = await fetch(url, { credentials: "include", cache: "no-store" });
      const text = await response.text();
      return { ok: true, status: response.status, url: response.url, text };
    } catch (error) {
      return { ok: false, url, error: String(error) };
    }
  };
  return Promise.all([getText(statusUrl), getText(healthUrl)]).then(([webide, health]) => ({
    location: location.href,
    visibilityState: document.visibilityState,
    hasFocus: document.hasFocus(),
    activity,
    webide,
    health,
  }));
})()
"@
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($js))
  $probeOutput = Run-AgentBrowser @("eval", "-b", $b64)
  $probeText = ($probeOutput | Out-String).Trim()
  Log "background-probe: $probeText"
  return $probeText
}

function Send-SafeClickKeepalive {
  $urlOutput = Run-AgentBrowser @("get", "url")
  $urlText = ($urlOutput | Out-String).Trim()
  if ($urlText -ne $IdeUrl) {
    Log "safe-click skipped: active url is not IDE url: $urlText"
    return
  }

  Run-AgentBrowser @("mouse", "move", "$SafeClickX", "$SafeClickY") | Out-Null
  Run-AgentBrowser @("mouse", "down", "left") | Out-Null
  Run-AgentBrowser @("mouse", "up", "left") | Out-Null
  Log "safe-click: x=$SafeClickX, y=$SafeClickY"
}

Log "keepalive started; interval=${IntervalSeconds}s; interactive_recovery=$InteractiveRecovery; terminal_keepalive=$TerminalKeepalive; synthetic_activity=$(-not $NoSyntheticActivity); safe_click=$SafeClickKeepalive"

while ($true) {
  try {
    if ($InteractiveRecovery -or $TerminalKeepalive) {
      Ensure-Page
      Handle-RecoveryButtons
      if ($TerminalKeepalive) {
        $terminalResult = Send-TerminalKeepalive
        if ($terminalResult -match '"ok":\s*false') {
          Log "terminal keepalive did not confirm terminal input path"
        }
      } else {
        Log "terminal-keepalive: disabled"
      }
      $healthText = Test-Health
      if ($healthText -match 'WEBIDE_RECORD_NOT_FOUND') {
        Log "health indicates reclaimed IDE; attempting restart"
        if ($InteractiveRecovery) {
          Handle-RestartOnly
        } else {
          Log "interactive recovery disabled; not clicking restart"
        }
      }
    } else {
      $healthText = Invoke-BackgroundProbe
      if ($SafeClickKeepalive) {
        Send-SafeClickKeepalive
      }
      if ($healthText -match 'WEBIDE_RECORD_NOT_FOUND') {
        Log "health indicates reclaimed IDE; interactive recovery disabled"
      }
    }
  } catch {
    Log "error: $($_.Exception.Message)"
  }

  if ($Once) {
    Log "keepalive once completed"
    break
  }

  Start-Sleep -Seconds $IntervalSeconds
}
