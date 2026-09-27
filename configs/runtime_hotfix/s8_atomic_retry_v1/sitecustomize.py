"""Infrastructure-only Windows atomic replace retry for one explicitly scoped run."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


_original = getattr(os, "_capstone_original_replace", os.replace)
os._capstone_original_replace = _original
_scope_text = os.environ.get("CAPSTONE_ATOMIC_RETRY_ROOT", "")
_scope = Path(_scope_text).resolve() if _scope_text else None


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _replace(source, destination):
    if _scope is None or not _inside(destination, _scope):
        return _original(source, destination)
    deadline = time.monotonic() + 5.
    delay = .005
    while True:
        try:
            return _original(source, destination)
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.2, delay * 2)


os.replace = _replace
os.environ["CAPSTONE_ATOMIC_RETRY_ACTIVE"] = "1" if _scope is not None else "0"
os.environ["CAPSTONE_ATOMIC_RETRY_MODULE_SHA256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
