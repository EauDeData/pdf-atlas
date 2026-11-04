"""Tiny file-based work queue: shards are claimed with O_EXCL lock files (dynamic load balancing,
any number of workers, safe restarts). Stale locks (dead pid on this host) are reclaimed."""
import os
import socket
import time

from . import config as C

HOST = socket.gethostname()
EXTRACT_DONE = C.SHARDS / "EXTRACT_DONE"


def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def claim(lock):
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{HOST} {os.getpid()}".encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            host, pid = open(lock).read().split()
            if host == HOST and not _alive(int(pid)):
                os.unlink(lock)
                return claim(lock)
        except Exception:
            pass
        return False


def release(lock):
    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass


def stream(prefix, out_dir, out_suffix=".npy", poll=20):
    """Yield (bid, shard_path) for every finished extraction shard lacking output, until extraction is done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        done_flag = EXTRACT_DONE.exists()
        got = False
        for d in sorted(C.SHARDS.glob("docs_*.parquet")):
            bid = d.stem.split("_")[1]
            out = out_dir / f"{prefix}_{bid}{out_suffix}"
            if out.exists():
                continue
            lock = out_dir / f"{prefix}_{bid}.lock"
            if not claim(lock):
                continue
            try:
                if not out.exists():
                    got = True
                    yield int(bid), C.SHARDS / f"{prefix}_{bid}.parquet", out
            finally:
                release(lock)
        if done_flag and not got:
            return
        if not got:
            time.sleep(poll)


def atomic_save_npy(path, arr):
    import numpy as np
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
