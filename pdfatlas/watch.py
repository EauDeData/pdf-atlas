"""Watch the sources for new PDFs and add them to the explorer, on the GPUs you name.

    python -m pdfatlas.watch --text-gpus 2 3 --image-gpu 1     # GPUs as physical indices
    python -m pdfatlas.watch                                    # no GPUs given: take idle ones per run

Every --interval seconds the sources (PDFATLAS_SOURCES: a directory, where a new sub-directory is a
new collection, or a sources JSON) are listed again. A new PDF is accepted once it has stopped
changing (same size and mtime over two scans, older than --settle seconds, and ending in %%EOF, so
files still being copied are not read half-written). Accepted PDFs are appended to the catalogue
with fresh doc_ids and new extraction batches, then a short pipeline run processes only those:

    extract -> embed_text (one per text GPU) + embed_images -> ingest --incremental -> topics

all streaming, exactly like the first run, and each exits when the new batches are done. The GPUs
are only held during that run. New documents appear in the explorer (and pulse on the map) when it
finishes; they get topics by nearest centroid and map positions by UMAP transform (see
PDFATLAS_REFIT_GROWTH in topics.py for when everything is refitted).

The watcher waits until the initial pipeline has finished (INGEST_DONE and TOPICS_DONE) before it
touches anything. Every step resumes from disk, so it can be stopped and restarted at any time; an
interrupted run is finished on the next start. Deleted or modified PDFs are not handled: a file
already in the catalogue is never re-read.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import pandas as pd

from . import config as C
from .catalog import append, iter_pdfs, make_row, read_metadata
from .extract import BATCHES, assign_batches
from .work import EXTRACT_DONE

INGEST_DONE = C.WORK / "INGEST_DONE"
TOPICS_DONE = C.WORK / "TOPICS_DONE"
RUNNING = C.WORK / "WATCH_RUNNING"   # present while a watcher run is unfinished
LOG = C.WORK / "logs"
_CHILDREN = []


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------- detection
def complete_pdf(path):
    """A PDF being copied has no %%EOF at its end yet (writers append sequentially)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 2048))
            return b"%%EOF" in f.read()
    except OSError:
        return False


class Scanner:
    def __init__(self, settle, twice=True):
        # twice: also require the same (size, mtime) at the previous scan (not possible with --once)
        self.settle, self.twice, self.seen = settle, twice, {}   # path -> (size, mtime) at the previous scan

    def scan(self, known):
        """-> [(path, source)] of new PDFs that are ready to be read."""
        ready, now, cur = [], time.time(), {}
        for src in C.load_sources():
            try:
                for path in iter_pdfs(src):
                    if path in known:
                        continue
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    sig = (st.st_size, st.st_mtime)
                    cur[path] = sig
                    age = now - st.st_mtime
                    if (st.st_size > 0 and (not self.twice or self.seen.get(path) == sig) and age >= self.settle
                            and (age >= 10 * self.settle or complete_pdf(path))):
                        ready.append((path, src))
            except OSError as e:
                log(f"! cannot list {src['path']}: {e}")
        waiting = len(cur) - len(ready)
        self.seen = cur
        return ready, waiting


def add_to_catalog(found, batch):
    """Append new PDFs to the catalogue and give them their own extraction batches."""
    rows, by_src = [], {}
    for path, src in found:
        by_src.setdefault(src["name"], (src, []))[1].append(path)
    for name, (src, paths) in by_src.items():
        meta = read_metadata(src["metadata"]) if src["metadata"] else {}
        for p in sorted(paths):
            try:
                rows.append(make_row(p, src, meta))
            except OSError as e:
                log(f"! skipping {p}: {e}")
        log(f"+ {len(paths)} new PDF(s) in '{name}'")
    if not rows:
        return 0
    append(rows)
    assign_batches(pd.read_parquet(C.CATALOG, columns=["doc_id"]), batch)
    return len(rows)


# ------------------------------------------------------------------------------- state
def initial_pipeline_done():
    return INGEST_DONE.exists() and TOPICS_DONE.exists()


def batches_without_shards():
    if not BATCHES.exists():
        return []
    bids = sorted(pd.read_parquet(BATCHES, columns=["bid"]).bid.unique().tolist())
    return [b for b in bids if not (C.SHARDS / f"docs_{b:05d}.parquet").exists()]


def batches_not_ingested():
    if not BATCHES.exists() or not C.DB.exists():
        return []
    import sqlite3
    con = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True, timeout=30)
    try:
        done = {b for (b,) in con.execute("SELECT bid FROM ingested WHERE kind='image'")}
    finally:
        con.close()
    bids = pd.read_parquet(BATCHES, columns=["bid"]).bid.unique().tolist()
    return [b for b in bids if b not in done]


# ------------------------------------------------------------------------------- the run
def pick_gpus(a, n_batches):
    """Declared GPUs win; otherwise the idle ones right now (polite on a shared box)."""
    text = list(a.text_gpus or [])
    img = a.image_gpu
    if not text or img is None:
        try:
            from .gpu import free_gpus
            free = free_gpus()
        except Exception as e:  # no nvidia-smi
            log(f"! cannot list GPUs: {e}")
            free = []
        if not text:
            cand = [g for g in free if g != img]
            # like run.sh: with nothing declared, the first idle GPU embeds images, the others text
            text = cand[1:] if img is None and len(cand) > 1 else cand
        if img is None:
            rest = [g for g in free if g not in text]
            img = rest[0] if rest else (text[0] if text else None)
    text = text[:max(1, min(a.max_text_gpus, n_batches))]  # a handful of PDFs needs one GPU, not four
    return text, img


def spawn(name, cmd, threads, cores=None):
    env = dict(os.environ, PDFATLAS_THREADS=str(threads))
    if cores:
        cmd = ["taskset", "-c", cores] + cmd
    LOG.mkdir(parents=True, exist_ok=True)
    out = open(LOG / f"watch.{name}.log", "a")
    out.write(f"\n==== {time.ctime()} {' '.join(cmd)}\n"); out.flush()
    p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.dirname(__file__)))
    p.name, p.log = name, out
    _CHILDREN.append(p)
    return p


def stop_children():
    for p in _CHILDREN:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 20
    for p in _CHILDREN:
        try:
            p.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
        p.log.close()
    _CHILDREN.clear()


def run_pipeline(a):
    todo = batches_without_shards()
    text_gpus, img_gpu = pick_gpus(a, max(1, len(todo)))
    if not text_gpus or img_gpu is None:
        raise RuntimeError("no GPU available: pass --text-gpus / --image-gpu (TEXT_GPUS / IMAGE_GPU in run.sh)")
    RUNNING.write_text(time.ctime())
    for f in (EXTRACT_DONE, INGEST_DONE, TOPICS_DONE):  # every stage streams until these reappear
        f.unlink(missing_ok=True)
    log(f"run: {len(todo)} batch(es) to extract · text GPUs {text_gpus} · image GPU {img_gpu}")
    py, t0 = [a.python, "-m"], time.time()
    spawn("extract", py + ["pdfatlas.extract", "--workers", str(a.extract_workers), "--batch", str(a.batch)], 1, a.extract_cores)
    for g in text_gpus:
        spawn(f"text{g}", py + ["pdfatlas.embed_text", "--gpu", str(g)], 2)
    spawn(f"img{img_gpu}", py + ["pdfatlas.embed_images", "--gpu", str(img_gpu), "--workers", str(a.image_workers)], 1)
    spawn("ingest", py + ["pdfatlas.ingest", "--incremental"], 2)
    spawn("topics", py + ["pdfatlas.topics"], 4)
    try:
        while True:
            alive = [p for p in _CHILDREN if p.poll() is None]
            bad = [p for p in _CHILDREN if p.poll() not in (None, 0)]
            if bad:
                raise RuntimeError(f"{bad[0].name} exited with {bad[0].returncode} (see {LOG}/watch.{bad[0].name}.log)")
            if not alive:
                break
            time.sleep(5)
    finally:
        stop_children()
    RUNNING.unlink(missing_ok=True)
    log(f"run done in {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-gpus", type=int, nargs="*", default=None, help="physical GPU indices for text embedding")
    ap.add_argument("--image-gpu", type=int, default=None, help="physical GPU index for image embedding")
    ap.add_argument("--max-text-gpus", type=int, default=3)
    ap.add_argument("--interval", type=float, default=120, help="seconds between scans of the sources")
    ap.add_argument("--settle", type=float, default=60, help="a new file must be unchanged for this long")
    ap.add_argument("--extract-workers", type=int, default=8)
    ap.add_argument("--extract-cores", default="", help="taskset core list for extraction (default: inherit)")
    ap.add_argument("--image-workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=100, help="docs per extraction batch (keep = extract --batch)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--once", action="store_true", help="one scan (+ run if needed), then exit (for cron)")
    a = ap.parse_args()

    def bye(*_):
        stop_children()
        sys.exit(0)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, bye)

    scanner, backoff, said = Scanner(a.settle, twice=not a.once), a.interval, None
    log(f"watching {C.SOURCES_SPEC} every {a.interval:.0f}s · GPUs: "
        f"{'text ' + str(a.text_gpus) if a.text_gpus else 'text: idle at run time'}, "
        f"{'image ' + str(a.image_gpu) if a.image_gpu is not None else 'image: idle at run time'}")
    while True:
        try:
            if not (initial_pipeline_done() or RUNNING.exists()):
                if said != "wait":
                    log("waiting for the initial pipeline to finish (INGEST_DONE + TOPICS_DONE)…"); said = "wait"
            else:
                if not BATCHES.exists():  # pin the batches of the existing index before anything is appended
                    assign_batches(pd.read_parquet(C.CATALOG, columns=["doc_id"]), a.batch)
                known = set(pd.read_parquet(C.CATALOG, columns=["path"]).path)
                ready, waiting = scanner.scan(known)
                if ready:
                    add_to_catalog(ready, a.batch)
                if waiting and said != f"settle{waiting}":
                    log(f"{waiting} new file(s) still settling / being copied"); said = f"settle{waiting}"
                if RUNNING.exists() or batches_not_ingested():
                    run_pipeline(a)
                    said = None
                backoff = a.interval
        except Exception as e:  # noqa: BLE001  everything resumes from disk: log, back off, retry
            log(f"! {e}")
            backoff = min(3600, backoff * 2)
            if a.once:
                raise
            time.sleep(backoff - a.interval)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
