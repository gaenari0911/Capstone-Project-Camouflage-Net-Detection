"""Exercise real Windows job closure, without terminating any real SSH session."""
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil
from capstone_lab.campaign.detached import ExtendedLimits, job_info, kernel, spawn

ROOT = Path(__file__).resolve().parents[1]


def probe_child(path):
    path = Path(path)
    (path.with_suffix(".started")).write_text(json.dumps({"pid": os.getpid(), "job": job_info()}))
    time.sleep(3)
    path.with_suffix(".finished").write_text("survived")


def probe_parent(directory):
    k = kernel()
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.CreateJobObjectW.restype = w.HANDLE
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    job = k.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2800  # KILL_ON_JOB_CLOSE | BREAKAWAY_OK, same as actual SSH job.
    if not job or not k.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    directory = Path(directory)
    for name in ("inherited", "detached"):
        command = [sys.executable, "-m", "tests.test_s8_detached", "child", str(directory / name)]
        with (directory / (name + ".log")).open("wb") as log:
            if name == "detached":
                spawn(command, cwd=ROOT, stdout=log, stderr=log)
            else:
                subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 10
    while not all((directory / (name + ".started")).exists() for name in ("inherited", "detached")):
        if time.monotonic() > deadline:
            raise RuntimeError("probe child startup timeout")
        time.sleep(.05)
    os._exit(0)  # Closes our sole job handle, killing inherited children.


@unittest.skipUnless(os.name == "nt", "Windows process lifecycle test")
class WindowsDetachTests(unittest.TestCase):
    def test_session_job_close_kills_inherited_but_detached_child_finishes(self):
        base = ROOT / "artifacts/s8_detach_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            result = subprocess.run([sys.executable, "-m", "tests.test_s8_detached", "parent", directory],
                                    cwd=ROOT, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            inherited = json.loads((Path(directory) / "inherited.started").read_text())
            detached = json.loads((Path(directory) / "detached.started").read_text())
            self.assertTrue(inherited["job"]["kill_on_close"])
            self.assertFalse(detached["job"]["in_job"])
            deadline = time.monotonic() + 10
            while psutil.pid_exists(detached["pid"]) and time.monotonic() < deadline:
                time.sleep(.1)
            self.assertFalse(psutil.pid_exists(inherited["pid"]))
            self.assertTrue((Path(directory) / "detached.finished").exists())
            self.assertFalse((Path(directory) / "inherited.finished").exists())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "parent":
        probe_parent(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "child":
        probe_child(sys.argv[2])
    else:
        unittest.main()
