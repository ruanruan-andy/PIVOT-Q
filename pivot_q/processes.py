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


@contextmanager
def services(cfg, args, out, *, anchor=False, suite=None):
    from pathlib import Path
    from fcntl import flock, LOCK_EX, LOCK_NB
    import socket
    import sys

    root = Path(__file__).resolve().parents[1]
    paths = cfg["paths"]
    python = paths["simulator_python"]
    conda = Path(python).parent.parent
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
               MUJOCO_GL="egl", QT_QPA_PLATFORM="offscreen", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               LD_LIBRARY_PATH=str(conda / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
               MAGICK_HOME=str(conda))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    benchmark = getattr(args, "benchmark", "libero_plus")
    with (out / ".run.lock").open("a") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        with managed_processes() as processes:
            modes = [(False, args.port)] + ([(True, args.clean_port)] if anchor else [])
            for clean, port in modes:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", port))
                original = clean or benchmark == "libero"
                source = paths["libero" if original else "libero_plus"]
                command = [python, str(root / "pivot_q/libero_service.py"),
                           "--port", str(port), "--seed", str(cfg["train"]["seed"] if args.seed is None else args.seed)]
                if suite:
                    command += ["--suite", suite]
                if not original:
                    command += ["--manifest", cfg["benchmark"]["task_file"]]
                model_env = dict(env, PYTHONPATH=str(root) + ":" + source,
                                 LIBERO_CONFIG_PATH=paths["clean_config" if original else "plus_config"])
                processes.append(start(command, model_env, out / "logs" / ("clean.log" if clean else "environment.log")))
            yield processes
