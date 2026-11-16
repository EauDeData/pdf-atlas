"""Stage 1: read the PDF object tree with PyMuPDF (no OCR, no layout model, no pixels saved).

Per batch of documents we write three small parquet shards to WORK/shards:
  docs_XXXXX.parquet   doc_id, n_pages, n_text_pages, n_images, pdf_title, error
  text_XXXXX.parquet   doc_id, page, kind(0=body,1=metadata), text
  img_XXXXX.parquet    doc_id, page, xref, x0,y0,x1,y1 (fraction of page), w, h
Image pixels are decoded later, in memory, only to embed them.

Which document goes into which batch is fixed once, in shards/BATCHES.parquet (doc_id -> bid).
Documents appended to the catalogue later (pdfatlas.watch) get new batches after the last one,
so existing shards never change meaning.
"""
import argparse
import os
import pickle
import re
import select
import signal
import time
from multiprocessing import Pool

import pandas as pd

from . import config as C

_WS = re.compile(r"\s+")
_HYPH = re.compile(r"(\w)-\s*\n\s*(\w)")


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def chunk_words(text, n=C.CHUNK_WORDS):
    w = text.split()
    if len(w) <= n * 1.3:
        return [" ".join(w)]
    k = max(1, round(len(w) / n))
    step = -(-len(w) // k)
    return [" ".join(w[i:i + step]) for i in range(0, len(w), step)]


def one_doc(row):
    import pymupdf
    pymupdf.TOOLS.mupdf_display_errors(False)
    doc_id, path, title, abstract = row
    texts, imgs = [], []
    info = dict(doc_id=doc_id, n_pages=0, n_text_pages=0, n_images=0, pdf_title="", error="")
    meta_text = (title + ". " + abstract).strip()
    try:
        if os.path.getsize(path) > C.MAX_FILE_MB * 2**20:
            raise RuntimeError("too large")
        signal.alarm(180)
        with pymupdf.open(path) as d:
            info["n_pages"] = d.page_count
            info["pdf_title"] = ((d.metadata or {}).get("title") or "")[:300]
            for pno in range(min(d.page_count, C.MAX_PAGES)):
                page = d[pno]
                pw, ph = page.rect.width or 1, page.rect.height or 1
                t = page.get_text("text")
                t = _WS.sub(" ", _HYPH.sub(r"\1\2", t)).strip()
                if len(t.split()) >= C.MIN_CHUNK_WORDS:
                    info["n_text_pages"] += 1
                    for c in chunk_words(t):
                        if len(c.split()) >= C.MIN_CHUNK_WORDS:
                            texts.append((doc_id, pno, 0, c))
                kept = 0
                for im in page.get_image_info(xrefs=True):
                    if kept >= C.MAX_IMGS_PER_PAGE:
                        break
                    x0, y0, x1, y1 = im["bbox"]
                    x0, x1 = max(0, x0) / pw, min(pw, x1) / pw
                    y0, y1 = max(0, y0) / ph, min(ph, y1) / ph
                    frac = max(0, x1 - x0) * max(0, y1 - y0)
                    if (im.get("xref", 0) <= 0 or min(im["width"], im["height"]) < C.IMG_MIN_PX
                            or frac < C.IMG_MIN_FRAC or frac > C.IMG_MAX_FRAC):
                        continue
                    imgs.append((doc_id, pno, im["xref"], x0, y0, x1, y1, im["width"], im["height"]))
                    kept += 1
        signal.alarm(0)
    except _Timeout:
        info["error"] = "timeout"
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        info["error"] = str(e)[:200]
    info["n_images"] = len(imgs)
    if len(meta_text) > 20:  # every doc gets a metadata chunk -> scans stay findable
        texts.append((doc_id, -1, 1, meta_text[:3000]))
    return info, texts, imgs


def guarded_doc(row, limit=C.DOC_TIMEOUT_S):
    """Run one_doc in a forked child and SIGKILL it at the deadline.

    A Python signal cannot interrupt MuPDF stuck in a C loop, so a single malformed PDF could
    otherwise hang a worker forever and keep the pipeline from ever finishing. Also survives
    segfaults. A killed document keeps its metadata chunk, so it stays findable by title.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            with os.fdopen(w, "wb") as f:
                pickle.dump(one_doc(row), f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            os._exit(0)
    os.close(w)
    buf, deadline, why = bytearray(), time.monotonic() + limit, ""
    while True:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([r], [], [], left)[0]:
            os.kill(pid, signal.SIGKILL)
            why = "killed: exceeded time limit"
            break
        chunk = os.read(r, 1 << 20)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    os.waitpid(pid, 0)
    if not why:
        try:
            return pickle.loads(bytes(buf))
        except Exception:  # child died mid-write (segfault, OOM)
            why = "killed: worker crashed"
    doc_id, _, title, abstract = row
    meta = (title + ". " + abstract).strip()
    info = dict(doc_id=doc_id, n_pages=0, n_text_pages=0, n_images=0, pdf_title="", error=why)
    return info, ([(doc_id, -1, 1, meta[:3000])] if len(meta) > 20 else []), []


def run_batch(args):
    bid, rows = args
    out = C.SHARDS / f"docs_{bid:05d}.parquet"
    if out.exists():
        return bid, 0, 0
    signal.signal(signal.SIGALRM, _alarm)
    infos, texts, imgs = [], [], []
    for r in rows:
        i, t, m = guarded_doc(r)
        infos.append(i); texts += t; imgs += m
    pd.DataFrame(texts, columns=["doc_id", "page", "kind", "text"]).astype(
        {"doc_id": "int32", "page": "int16", "kind": "int8"}).to_parquet(C.SHARDS / f"text_{bid:05d}.parquet", compression="zstd")
    pd.DataFrame(imgs, columns=["doc_id", "page", "xref", "x0", "y0", "x1", "y1", "w", "h"]).astype(
        {"doc_id": "int32", "page": "int16", "xref": "int32", "x0": "float32", "y0": "float32", "x1": "float32",
         "y1": "float32", "w": "int32", "h": "int32"}).to_parquet(C.SHARDS / f"img_{bid:05d}.parquet")
    pd.DataFrame(infos).to_parquet(out)  # written last = batch done marker
    return bid, len(texts), len(imgs)


BATCHES = C.SHARDS / "BATCHES.parquet"


def assign_batches(cat, batch=100):
    """-> DataFrame(doc_id, bid), persisted. First call: random doc order (fixed seed), so every prefix
    of shards is a uniform sample of the corpus and the explorer is representative while it grows
    (identical to the original, manifest-less assignment, so existing work dirs stay valid).
    Later calls: catalogue rows without a batch are appended as new batches, in doc_id order."""
    if BATCHES.exists():
        m = pd.read_parquet(BATCHES)
        new = cat.loc[~cat.doc_id.isin(m.doc_id), "doc_id"].sort_values().to_numpy()
        if not len(new):
            return m
        b0 = int(m.bid.max()) + 1 if len(m) else 0
        add = pd.DataFrame(dict(doc_id=new, bid=b0 + pd.RangeIndex(len(new)) // batch))
        m = pd.concat([m, add], ignore_index=True)
    else:
        ids = cat.sample(frac=1.0, random_state=0).doc_id.to_numpy()
        m = pd.DataFrame(dict(doc_id=ids, bid=pd.RangeIndex(len(ids)) // batch))
    m = m.astype({"doc_id": "int64", "bid": "int32"})
    C.SHARDS.mkdir(parents=True, exist_ok=True)
    tmp = BATCHES.with_suffix(".tmp")
    m.to_parquet(tmp, index=False)
    os.replace(tmp, BATCHES)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only first N docs (debug)")
    a = ap.parse_args()
    C.SHARDS.mkdir(parents=True, exist_ok=True)
    cat = pd.read_parquet(C.CATALOG, columns=["doc_id", "path", "title", "abstract"])
    if a.limit:  # debug: no manifest, throwaway batches
        cat = cat.sample(a.limit, random_state=0).sort_values("doc_id").sample(frac=1.0, random_state=0)
        rows = list(cat.itertuples(index=False, name=None))
        batches = [(i // a.batch, rows[i:i + a.batch]) for i in range(0, len(rows), a.batch)]
    else:
        m = assign_batches(cat, a.batch)
        rowmap = {r[0]: r for r in cat.itertuples(index=False, name=None)}
        batches = [(int(b), [rowmap[d] for d in g.doc_id if d in rowmap]) for b, g in m.groupby("bid", sort=True)]
    todo = [b for b in batches if not (C.SHARDS / f"docs_{b[0]:05d}.parquet").exists()]
    (C.SHARDS / "N_BATCHES").write_text(str(len(batches)))
    print(f"{len(todo)}/{len(batches)} batches to do", flush=True)
    t0, nt, ni = time.time(), 0, 0
    with Pool(a.workers, maxtasksperchild=20) as pool:  # imap keeps bid order roughly ascending
        for k, (bid, t, m) in enumerate(pool.imap_unordered(run_batch, todo, chunksize=1), 1):
            nt += t; ni += m
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)}] chunks={nt} imgs={ni} {el/60:.1f}min eta={(len(todo)-k)*el/k/60:.1f}min", flush=True)
    if not a.limit:
        (C.SHARDS / "EXTRACT_DONE").write_text(str(len(batches)))
    (C.SHARDS / "N_BATCHES").write_text(str(len(batches)))


if __name__ == "__main__":
    main()
