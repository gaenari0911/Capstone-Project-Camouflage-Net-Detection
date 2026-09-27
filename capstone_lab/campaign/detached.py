"""Windows job-object diagnostics and an explicitly detached process launcher."""
from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes as w


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_process_limit", w.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


def kernel():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    k.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]
    k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                           w.DWORD, ctypes.POINTER(w.DWORD)]
    return k


def job_info():
    if os.name != "nt":
        return {"platform": os.name, "in_job": False}
    k = kernel()
    member = w.BOOL()
    if not k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(member)):
        raise ctypes.WinError(ctypes.get_last_error())
    result = {"platform": "nt", "in_job": bool(member.value)}
    if member.value:
        limits = ExtendedLimits()
        if not k.QueryInformationJobObject(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise ctypes.WinError(ctypes.get_last_error())
        flags = limits.basic.flags
        result.update(flags=hex(flags), breakaway=bool(flags & 0x800),
                      silent_breakaway=bool(flags & 0x1000), kill_on_close=bool(flags & 0x2000))
    return result


def spawn(command, *, cwd, stdout, stderr, env=None):
    """Never silently fall back to a session-owned Windows child."""
    flags = 0
    if os.name == "nt":
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.CREATE_BREAKAWAY_FROM_JOB)
    return subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout,
                            stderr=stderr, env=env, shell=False, close_fds=True,
                            creationflags=flags, start_new_session=os.name != "nt")


def process_in_job(pid):
    """Inspect a live process without relying on parent PID ancestry."""
    if os.name != "nt":
        return False
    k = kernel()
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.CloseHandle.argtypes = [w.HANDLE]
    handle = k.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        member = w.BOOL()
        if not k.IsProcessInJob(handle, None, ctypes.byref(member)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(member.value)
    finally:
        k.CloseHandle(handle)


if __name__ == "__main__":
    import json
    import psutil
    import sys
    if len(sys.argv) > 1:
        print(json.dumps([{"pid": int(pid), "in_job": process_in_job(int(pid))}
                          for pid in sys.argv[1:]], indent=2))
        raise SystemExit(0)
    print(json.dumps({"pid": os.getpid(), "job": job_info(),
                      "parents": [{"pid": p.pid, "name": p.name()}
                                  for p in psutil.Process().parents()]}, indent=2))
