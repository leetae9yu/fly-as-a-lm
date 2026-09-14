"""Metric-free worker stream and Linux file-event download acknowledgement hold."""

import ctypes
import os
import re
import select
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import cast

MARKER = re.compile(
    r"""ALPN_CALIBRATION_(?:
    SEED_READY[ ]seed=(?:[7-9]|1[0-2])|
    STATUS[ ]status=(?:balanced|insufficient_common_support|failed))\n?
""",
    re.VERBOSE,
)


def emit(marker: str) -> None:
    """Flush one metric-free status marker."""
    _ = sys.stdout.write(marker + "\n")
    _ = sys.stdout.flush()


def hold(
    acknowledgement: Path, ready: Callable[[], None], timeout: float | None = None
) -> None:
    """Subscribe before announcing readiness; await file creation without polling."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    descriptor = cast("int", libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK))
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Cannot initialize acknowledgement watch")
    try:
        watch = cast(
            "int",
            libc.inotify_add_watch(
                descriptor, os.fsencode(acknowledgement.parent), 0x100 | 0x80 | 0x8
            ),
        )
        if watch < 0:
            raise OSError(ctypes.get_errno(), "Cannot watch acknowledgement directory")
        deadline = None if timeout is None else time.monotonic() + timeout
        ready()
        while not acknowledgement.is_file():
            remaining = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            if not select.select([descriptor], [], [], remaining)[0]:
                message = "Download acknowledgement timed out"
                raise TimeoutError(message)
            _ = os.read(descriptor, 65536)
    finally:
        os.close(descriptor)


def command(project: Path, destination: Path) -> list[str]:
    """Run only the installed sealed calibration worker module."""
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.alpn_causal_calibration_worker",
        str(project),
        str(project / "seal.json"),
        str(destination),
    ]


def execute(project: Path, destination: Path, acknowledgement: Path) -> int:
    """Keep all non-allowlisted output in a local log, including failure details."""
    project, destination, acknowledgement = (
        path.resolve() for path in (project, destination, acknowledgement)
    )
    if acknowledgement.exists():
        message = "Stale download acknowledgement exists"
        raise FileExistsError(message)
    with (
        (project / "calibration-execute.log").open("x") as log,
        subprocess.Popen[str](
            command(project, destination),
            cwd=project,
            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        ) as process,
    ):
        output = process.stdout
        if not isinstance(output, TextIOWrapper):
            message = "Worker stdout is not connected"
            raise TypeError(message)
        for line in output:
            if MARKER.fullmatch(line):
                emit(line.rstrip("\n"))
            else:
                _ = log.write(line)
        returncode = process.wait()
    hold(acknowledgement, lambda: emit("ALPN_CALIBRATION_HOLD_READY"))
    emit(f"ALPN_CALIBRATION_EXEC_EXIT {returncode}")
    return returncode


def main() -> None:
    """Take project, result ZIP and download acknowledgement paths."""
    project, destination, acknowledgement = map(Path, sys.argv[1:])
    try:
        returncode = execute(project, destination, acknowledgement)
    except (OSError, ValueError, RuntimeError):
        with (project / "calibration-wrapper-failure.log").open("a") as log:
            _ = log.write(traceback.format_exc())
        emit("ALPN_CALIBRATION_STATUS status=failed")
        returncode = 1
        emit("ALPN_CALIBRATION_EXEC_EXIT 1")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
