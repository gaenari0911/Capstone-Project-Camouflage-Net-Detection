from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from capstone_lab.config import canonical_json
from capstone_lab.errors import StateError


def atomic_replace(source, destination, *, timeout=5.0):
    source, destination = Path(source), Path(destination)
    deadline = time.monotonic() + timeout
    delay = .005
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.2, delay * 2)


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(value) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise StateError(f"immutable artifact differs: {path}")
        return
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/scanners may briefly hold the destination without
    # FILE_SHARE_DELETE. Retry only that transient replacement failure.
    try:
        atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class RunLock:
    """OS-held byte lock. Process death releases it without deleting user files."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if not self.stream.tell():
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise StateError("another supervisor already owns this run") from exc
        return self

    def __exit__(self, *_):
        self.stream.close()
