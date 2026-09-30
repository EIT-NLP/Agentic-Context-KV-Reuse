"""CLI input-file checks."""
import sys
from pathlib import Path


def require_exists(path: Path | None, what: str) -> Path:
    """Exit with a clean error message if ``path`` does not exist.

    Call this from subcommand ``run()`` functions (not from argparse
    ``type=`` converters). Returns the validated ``Path`` on success.
    """
    if path is None:
        sys.exit(f"sragents: error: missing --{what}")
    p = Path(path)
    if not p.exists():
        sys.exit(f"sragents: error: {what} not found: {p}")
    return p
