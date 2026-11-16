"""Password-protected admin actions: adding / removing watched folders from the explorer.

The password is never stored, only a salted scrypt hash, in $PDFATLAS_WORK/admin.hash (mode 600) or
in the PDFATLAS_ADMIN_HASH environment variable. It is checked on the server; the browser only sends
the password (in a POST body, never in a URL, so it does not end up in access logs).

    python -m pdfatlas.admin set-password       # prompts, writes the hash
    python -m pdfatlas.admin hash               # prompts, prints a hash for PDFATLAS_ADMIN_HASH
    python -m pdfatlas.admin list               # watched folders added from the explorer

Without a hash configured, admin actions are disabled. Repeated wrong passwords from one client lock
it out for a while.
"""
import fnmatch
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

from . import config as C

HASH_FILE = C.WORK / "admin.hash"
N, R, P = 2**15, 8, 1                  # scrypt cost: ~50 ms and 32 MB per check
MAX_FAILS, LOCK_S = 5, 600
_fails, _lock = {}, threading.Lock()   # client -> [failures, locked_until]


# ------------------------------------------------------------------------------- password
def make_hash(password):
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, maxmem=2**26, dklen=32)
    return f"scrypt${N}${R}${P}${salt.hex()}${h.hex()}"


def stored_hash():
    h = os.environ.get("PDFATLAS_ADMIN_HASH", "").strip()
    if not h and HASH_FILE.exists():
        h = HASH_FILE.read_text().strip()
    return h or None


def configured():
    return stored_hash() is not None


def _check(password, stored):
    try:
        alg, n, r, p, salt, h = stored.split("$")
        assert alg == "scrypt"
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=2**26, dklen=len(h) // 2)
    except Exception:
        return False
    return hmac.compare_digest(got.hex(), h)


class Denied(Exception):
    def __init__(self, msg, status=403):
        super().__init__(msg)
        self.status = status


def verify(password, client):
    """Raises Denied unless the password matches. Wrong guesses lock the client out after MAX_FAILS."""
    stored = stored_hash()
    if stored is None:
        raise Denied("admin actions are disabled: run `python -m pdfatlas.admin set-password` on the server", 503)
    now = time.time()
    with _lock:
        n, until = _fails.get(client, (0, 0))
        if until > now:
            raise Denied(f"too many wrong passwords, try again in {int(until - now) // 60 + 1} min", 429)
    if _check(password or "", stored):
        with _lock:
            _fails.pop(client, None)
        return
    time.sleep(0.5)  # slow down guessing a little more
    with _lock:
        n += 1
        _fails[client] = (0, now + LOCK_S) if n >= MAX_FAILS else (n, 0)
    raise Denied("wrong password")


# ------------------------------------------------------------------------------- watched folders
_NAME = re.compile(r"[^\w.\- ]+", re.U)


def _inside(a, b):
    """a is b or below b."""
    try:
        Path(a).relative_to(b)
        return True
    except ValueError:
        return False


def all_sources():
    try:
        return C.load_sources()
    except SystemExit:  # PDFATLAS_SOURCES not set in this process: only the added folders are known
        return C.load_extra_sources()


def sources_view():
    extra = {s["path"] for s in C.load_extra_sources()}
    return [dict(name=s["name"], path=s["path"], added=s["path"] in extra) for s in all_sources()]


def count_pdfs(path, budget_s=4.0, cap=1_000_000):
    """Quick look at a folder before adding it. -> (n, complete)"""
    t0, n = time.time(), 0
    for _, dns, fns in os.walk(path):
        n += sum(1 for f in fns if f.lower().endswith(".pdf"))
        if time.time() - t0 > budget_s or n >= cap:
            return n, False
    return n, True


def _write_extra(srcs):
    tmp = C.EXTRA_SOURCES.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sources": srcs}, indent=2, ensure_ascii=False))
    os.replace(tmp, C.EXTRA_SOURCES)


def add_folder(path, name="", exclude=(), metadata=None):
    """Validate and register a folder; the watcher indexes it on its next scans. -> source dict"""
    if not path or not os.path.isabs(os.path.expanduser(path)):
        raise ValueError("give an absolute path, e.g. /data/pdfs/new_collection")
    p = Path(os.path.expanduser(path)).resolve()
    if not p.is_dir():
        raise ValueError(f"not a directory (as seen from the server): {p}")
    if not os.access(p, os.R_OK | os.X_OK):
        raise ValueError(f"the server cannot read {p}")
    if _inside(C.WORK, p):
        raise ValueError("that folder contains the atlas work directory")
    spec = Path(C.SOURCES_SPEC).expanduser().resolve() if C.SOURCES_SPEC else None
    if spec and spec.is_dir() and p.parent == spec:
        raise ValueError(f"every sub-folder of {spec} is already watched as its own collection")
    name = _NAME.sub("", (name or p.name)).strip()[:60]
    if not name:
        raise ValueError("give a collection name")
    for s in all_sources():
        sp = Path(s["path"]).expanduser().resolve()
        if s["name"] == name:
            raise ValueError(f"a collection named '{name}' already exists")
        if _inside(p, sp) and s.get("recursive", True) or _inside(sp, p):
            # nested folders would list the same PDFs twice, as two documents
            raise ValueError(f"overlaps the watched folder {sp} ('{s['name']}')")
    n, complete = count_pdfs(p)
    src = dict(name=name, path=str(p), exclude=[e for e in exclude if e], added=time.strftime("%Y-%m-%d %H:%M:%S"))
    if metadata:  # e.g. {"type": "jsonl", "path": ".../metadata.jsonl", "key": "file", "fields": {...}}
        src["metadata"] = metadata
    elif (p / "metadata.jsonl").exists():  # harvester convention: one JSON row per PDF, keyed by "file"
        src["metadata"] = dict(type="jsonl", path=str(p / "metadata.jsonl"), key="file",
                               fields=dict(title="title", authors="authors", url="item", date="date_issued", year="year"))
    _write_extra([dict(s) for s in _raw_extra()] + [src])
    return dict(src, n_pdfs=n, n_complete=complete)


def _raw_extra():
    try:
        return json.loads(C.EXTRA_SOURCES.read_text())["sources"]
    except (FileNotFoundError, ValueError, KeyError):
        return []


def remove_folder(path):
    """Stop watching a folder added from the explorer. Documents already indexed stay."""
    srcs = _raw_extra()
    keep = [s for s in srcs if s["path"] != path]
    if len(keep) == len(srcs):
        raise ValueError("only folders added from the explorer can be removed here")
    _write_extra(keep)


# ------------------------------------------------------------------------------- CLI
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("set-password", "hash"):
        pw = getpass.getpass("admin password: ")
        if len(pw) < 8:
            sys.exit("use at least 8 characters")
        if getpass.getpass("again: ") != pw:
            sys.exit("passwords differ")
        h = make_hash(pw)
        if cmd == "hash":
            print(h)
            return
        C.WORK.mkdir(parents=True, exist_ok=True)
        fd = os.open(HASH_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(h + "\n")
        print(f"hash written to {HASH_FILE} (the server reads it on every check: no restart needed)")
    elif cmd == "list":
        for s in sources_view():
            print(f"{'+' if s['added'] else ' '} {s['name']:24s} {s['path']}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
