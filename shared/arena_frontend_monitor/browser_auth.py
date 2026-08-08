"""Local agent-browser authentication configuration.

The browser profile contains session cookies and other credentials. It must
remain local and is stored under the Git-ignored runtime directory by default.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = ROOT / "runtime" / "browser_profile"
REPOSITORY_ROOT = ROOT.parents[1]
RUNTIME_ROOT = ROOT / "runtime"
DEFAULT_SESSION = "tencent-arena-persistent-v2"
LEGACY_SESSION = "tencent-arena"


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries without overriding the process env."""
    env_path = path or ROOT / ".env"
    if not env_path.is_file():
        return

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{env_path}:{line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise ValueError(f"{env_path}:{line_number}: invalid environment variable name")
        values = shlex.split(raw_value, comments=True, posix=True)
        if len(values) > 1:
            raise ValueError(f"{env_path}:{line_number}: quote values containing spaces")
        os.environ.setdefault(key, values[0] if values else "")


def agent_browser_env(session: str, session_name: str) -> dict[str, str]:
    """Return an environment using a persistent, private browser profile."""
    load_local_env()
    env = os.environ.copy()
    env["AGENT_BROWSER_SESSION"] = session
    env["AGENT_BROWSER_SESSION_NAME"] = session_name
    env.setdefault("AGENT_BROWSER_HEADED", "1")

    raw_profile = env.get("AGENT_BROWSER_PROFILE", "").strip()
    profile = Path(raw_profile or DEFAULT_PROFILE_DIR).expanduser()
    if not profile.is_absolute():
        profile = ROOT / profile
    profile = profile.resolve()
    if profile == ROOT or profile in ROOT.parents:
        raise ValueError(
            f"AGENT_BROWSER_PROFILE must be a dedicated subdirectory, not {profile}"
        )
    if profile.is_relative_to(REPOSITORY_ROOT) and not profile.is_relative_to(RUNTIME_ROOT):
        raise ValueError(
            "repository-local AGENT_BROWSER_PROFILE must be under "
            f"the Git-ignored runtime directory: {RUNTIME_ROOT}"
        )
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        profile.chmod(0o700)
    except OSError:
        pass
    env["AGENT_BROWSER_PROFILE"] = str(profile)
    return env


def default_session_from_env() -> str:
    """Migrate the old default session unless the caller explicitly opts out."""
    session = os.environ.get("AGENT_BROWSER_SESSION", DEFAULT_SESSION)
    allow_legacy = os.environ.get("AGENT_BROWSER_ALLOW_LEGACY_SESSION") == "1"
    if session == LEGACY_SESSION and not allow_legacy:
        print(
            f"[browser_auth] migrating legacy session {LEGACY_SESSION!r} "
            f"to {DEFAULT_SESSION!r}; set AGENT_BROWSER_ALLOW_LEGACY_SESSION=1 "
            "to keep the old daemon intentionally",
            file=sys.stderr,
        )
        return DEFAULT_SESSION
    return session


def default_session_name_from_env(session: str) -> str:
    session_name = os.environ.get("AGENT_BROWSER_SESSION_NAME", session)
    if session == DEFAULT_SESSION and session_name == LEGACY_SESSION:
        return DEFAULT_SESSION
    return session_name


def agent_browser_command(
    args: list[str],
    session: str,
    session_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Build a command that pins auth settings before contacting the daemon."""
    env = agent_browser_env(session, session_name)
    executable = (
        shutil.which("agent-browser")
        or shutil.which("agent-browser.cmd")
        or "agent-browser"
    )
    command = [
        executable,
        "--session",
        session,
        "--session-name",
        session_name,
        "--profile",
        env["AGENT_BROWSER_PROFILE"],
        *args,
    ]
    return command, env


def print_shell_env() -> None:
    session = default_session_from_env()
    session_name = default_session_name_from_env(session)
    env = agent_browser_env(session, session_name)
    for key in (
        "AGENT_BROWSER_SESSION",
        "AGENT_BROWSER_SESSION_NAME",
        "AGENT_BROWSER_PROFILE",
        "AGENT_BROWSER_HEADED",
    ):
        print(f"export {key}={shlex.quote(env[key])}")


load_local_env()


if __name__ == "__main__":
    if sys.argv[1:] == ["shell-env"]:
        print_shell_env()
    else:
        raise SystemExit("usage: browser_auth.py shell-env")
