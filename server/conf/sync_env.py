"""Small, dependency-free loader for local IDE-sync credential files.

The sync scripts intentionally do not use ``python-dotenv``: their runtime
environment is the Tencent IDE image and may not contain extra packages.  This
module accepts only simple ``KEY=value`` lines; it neither evaluates shell
syntax nor expands command substitutions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(raw_path: str | None) -> dict[str, str]:
    """Read a simple dotenv file without executing arbitrary shell content."""
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(f"env file is not a regular file: {path}")

    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _KEY_RE.fullmatch(key):
            raise ValueError(f"invalid env file line {line_no} in {path}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def setting(
    cli_value: str | None,
    name: str,
    file_values: Mapping[str, str],
    default: str = "",
) -> str:
    """Resolve one setting with CLI > process environment > env file order."""
    if cli_value:
        return cli_value
    process_value = os.environ.get(name)
    if process_value:
        return process_value
    return file_values.get(name, default)
