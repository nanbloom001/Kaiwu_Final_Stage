"""Local agent-browser authentication configuration.

The browser profile contains session cookies and other credentials. It must
remain local and is stored under the Git-ignored runtime directory by default.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = ROOT / "runtime" / "browser_profile"


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

    profile = Path(env.get("AGENT_BROWSER_PROFILE", DEFAULT_PROFILE_DIR)).expanduser()
    if not profile.is_absolute():
        profile = ROOT / profile
    profile = profile.resolve()
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        profile.chmod(0o700)
    except OSError:
        pass
    env["AGENT_BROWSER_PROFILE"] = str(profile)
    return env


load_local_env()
