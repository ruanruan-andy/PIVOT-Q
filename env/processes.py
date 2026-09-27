"""Own only explicitly launched process groups; clean up on errors and signals."""
from contextlib import contextmanager
import os
import signal
import subprocess


def start(command, environment, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=environment, start_new_session=True)


def stop(process, grace_seconds=15):
    """The caller must have created this process with start_new_session=True."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before descendants; terminate the owned group too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def managed_processes(grace_seconds=15):
    processes = []
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in signals:
        signal.signal(sig, interrupted)
    try:
        yield processes
    finally:
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for process in reversed(processes):
                stop(process, grace_seconds=grace_seconds)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def run_worker(command, *, processes, env, cwd=None, stdout=None, stderr=None):
    services = list(processes)
    worker = subprocess.Popen(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                              start_new_session=True)
    processes.append(worker)
    while True:
        try:
            result = worker.wait(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if any(process.poll() is not None for process in services):
                raise RuntimeError("A required service exited; inspect its log")
    if result:
        raise subprocess.CalledProcessError(result, command)
