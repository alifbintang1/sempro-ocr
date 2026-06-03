"""Pluggable extraction approaches.

Each module in this package registers itself in REGISTRY. To compare a new
approach, drop a new file here that calls ``register(name, fn)`` and
``run_all.py`` will pick it up automatically.

The function signature must be:

    def run(pdf_path: str) -> tuple[dict, float]

It must return ``(unified_output_dict, elapsed_seconds)`` where
``unified_output_dict`` follows the schema in ``idx_fin_parser.unified``.

If the approach fails to import (missing dependency) it should fail gracefully
at import-time so the registry remains usable for other approaches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

REGISTRY: dict[str, Callable[[str], tuple[dict, float]]] = {}


def register(name: str, fn: Callable[[str], tuple[dict, float]]) -> None:
    REGISTRY[name] = fn


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency on python-dotenv).

    Loads KEY=value lines from a project-root .env into os.environ, but only
    for keys not already set. Allows users to drop their OPENAI_API_KEY into
    a .env file without re-launching the shell.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


# Importing the modules triggers each one's `register(...)` call.
# Order matters only for display.
from . import native_pdf            # noqa: E402, F401
from . import ocr_full              # noqa: E402, F401
from . import baseline_regex        # noqa: E402, F401
from . import pdfplumber_tables     # noqa: E402, F401

try:
    from . import pymupdf_native    # noqa: E402, F401
except ImportError as exc:
    print(f"[approaches] pymupdf_native unavailable: {exc}")

try:
    from . import camelot_lattice   # noqa: E402, F401
except ImportError as exc:
    print(f"[approaches] camelot_lattice unavailable: {exc}")

try:
    import os as _os
    if _os.environ.get("OPENAI_API_KEY"):
        from . import vlm_openai    # noqa: E402, F401
    else:
        # Register only if key is set — avoid polluting comparison output
        # with traceback noise when the user hasn't configured the API.
        pass
except ImportError as exc:
    print(f"[approaches] vlm_openai unavailable: {exc}")
