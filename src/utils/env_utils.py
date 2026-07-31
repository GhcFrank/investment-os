"""Project-root environment loading that is independent of current cwd."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_environment(env_file: Path | str | None = None) -> Path:
    """Load a project env file without overriding explicit environment values."""

    path = Path(env_file) if env_file is not None else PROJECT_ENV_FILE
    load_dotenv(path, override=False)
    return path


def get_project_environment_value(
    name: str,
    *,
    env_file: Path | str | None = None,
    load_environment: bool = True,
) -> str:
    """Return one stripped value after optional project-root env loading."""

    if load_environment:
        load_project_environment(env_file)
    return os.environ.get(name, "").strip()
