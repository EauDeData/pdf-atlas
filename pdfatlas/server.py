"""FastAPI explorer.   uvicorn pdfatlas.server:app --host 0.0.0.0 --port 8800

Everything the browser sees is paginated server-side (offset/cursor + small pages); the full corpus is
never shipped. The only bulk payloads are the 2D maps, sent as compact binary (≈10 bytes / point).
"""
import json
import os
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config as C

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="PDF Atlas")
_local = threading.local()


def db():
    c = getattr(_local, "con", None)
    if c is None:
        c = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA mmap_size=8000000000")
        c.execute("PRAGMA busy_timeout=10000")
        _local.con = c
    return c


# ----------------------------------------------------------------------------- live vector stores
class LiveVectors:
    """Sealed Annoy segments + brute-forced unsealed tail, over an append-only fp16 matrix.
    `committed()` returns how many rows the DB has committed (the file may run slightly ahead)."""

    def __init__(self, kind, dim_fn, committed):
        self.kind, self.dim_fn, self.committed = kind, dim_fn, committed
        self.segs, self.sealed, self.tail, self.tail_start, self.n = [], 0, None, 0, 0
        self.checked, self.lock = 0.0, threading.Lock()

    def refresh(self, force=False):
        if not force and time.time() - self.checked < 10:
            return
        with self.lock:
            self.checked = time.time()
            dim = self.dim_fn()
            if not dim:
                return
            from annoy import AnnoyIndex
            rows = db().execute("SELECT seg, start, end, path FROM segments WHERE kind=? ORDER BY seg", (self.kind,)).fetchall()
            for seg, start, end, path in rows[len(self.segs):]:
                a = AnnoyIndex(dim, "angular"); a.load(str(C.IDX / path))
                self.segs.append((a, start, end))
            self.sealed = self.segs[-1][2] if self.segs else 0
            n = self.committed()
            if self.tail is None or self.tail_start != self.sealed:
                self.tail_start, self.tail = self.sealed, np.zeros((0, dim), np.float32)
            have = self.tail_start + len(self.tail)
            if n > have:
                X = np.memmap(C.IDX / f"{self.kind}.f16", np.float16, "r", shape=(n, dim))
                self.tail = np.concatenate([self.tail, np.asarray(X[have:n], np.float32)])
            self.n = n

    def vector(self, i):
        self.refresh()
        dim = self.dim_fn()
        X = np.memmap(C.IDX / f"{self.kind}.f16", np.float16, "r", shape=(self.n, dim))
        return np.asarray(X[i], np.float32)

    def search(self, v, k, valid=None):
        """-> (ids, cosine sims) best first, merged over segments + tail."""
        self.refresh()
        ids, sims = [], []
        for a, start, _ in self.segs:
            ii, dd = a.get_nns_by_vector(v, k, search_k=k * 40, include_distances=True)
            ids.append(np.asarray(ii, np.int64) + start)
            sims.append(1 - np.asarray(dd, np.float32) ** 2 / 2)
        if self.tail is not None and len(self.tail):
            sc = self.tail @ v
            if valid is not None:
                sc = np.where(valid(self.tail_start, self.tail_start + len(sc)), sc, -9)
            kk = min(k, len(sc))
            top = np.argpartition(-sc, kk - 1)[:kk]
            ids.append(top.astype(np.int64) + self.tail_start); sims.append(sc[top])
        if not ids:
            return np.array([], np.int64), np.array([], np.float32)
        ids, sims = np.concatenate(ids), np.concatenate(sims)
        o = np.argsort(-sims)[:k]
        keep = sims[o] > -2
        return ids[o][keep], sims[o][keep]


def _image_dim():
    f = C.IDX / "image.dim"
    return int(f.read_text()) if f.exists() else 0


def _committed(sql):
    return lambda: db().execute(sql).fetchone()[0]


class State:
    text = LiveVectors("text", lambda: C.TEXT_DIM, _committed("SELECT coalesce(max(id)+1,0) FROM chunks"))
    image = LiveVectors("image", _image_dim, _committed("SELECT coalesce(max(id)+1,0) FROM images"))
    docvec = None  # (matrix f32, n) for brute-force document similarity (<=150k x 512)
    docvec_checked = 0.0
    topics, topics_mtime = {}, 0.0
    _img_ok = None


S = State()


def image_ok_mask(lo, hi):
    rows = db().execute("SELECT id FROM images WHERE ok=0 AND id>=? AND id<?", (lo, hi)).fetchall()
    m = np.ones(hi - lo, bool)
    for (i,) in rows:
        m[i - lo] = False
    return m


def docvecs():
    if S.docvec is None or time.time() - S.docvec_checked > 10:
        S.docvec_checked = time.time()
        n = db().execute("SELECT coalesce(max(vrow)+1,0) FROM docs").fetchone()[0]
        old = S.docvec[0] if S.docvec is not None else np.zeros((0, C.TEXT_DIM), np.float32)
        if n > len(old):
            X = np.memmap(C.IDX / "docvec.f16", np.float16, "r", shape=(n, C.TEXT_DIM))
            old = np.concatenate([old, np.asarray(X[len(old):n], np.float32)])
        S.docvec = (old, n)
    return S.docvec[0]


def topics():
    f = C.MAP / "topics.json"
    if f.exists() and f.stat().st_mtime != S.topics_mtime:
        try:
            S.topics, S.topics_mtime = json.loads(f.read_text()), f.stat().st_mtime
        except Exception:
            pass
    return S.topics


def ensure():
    if not C.DB.exists():
        raise HTTPException(503, "index not built yet — the pipeline is still starting")


def qvec_text(q):
    from .embed_text import encode
    return _cached_vec(("t", q), lambda: encode([q], query=True, device=_device())[0].astype(np.float32))


def qvec_image(q):
    from .embed_images import encode_text
    return _cached_vec(("i", q), lambda: encode_text([q], device=_device())[0].astype(np.float32))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


_VEC = OrderedDict()
_VEC_LOCK = threading.Lock()


def _cached_vec(key, fn):
    with _VEC_LOCK:
        if key in _VEC:
            _VEC.move_to_end(key); return _VEC[key]
    v = fn()
    with _VEC_LOCK:
        _VEC[key] = v
        if len(_VEC) > 512:
            _VEC.popitem(last=False)
    return v


# result lists are cached per (query, filters) so paging = slicing, not re-searching
_RES = OrderedDict()
_RES_LOCK = threading.Lock()


def cached_results(key, fn):
    with _RES_LOCK:
        if key in _RES:
            _RES.move_to_end(key); return _RES[key]
    r = fn()
    with _RES_LOCK:
        _RES[key] = r
        while len(_RES) > 256:
            _RES.popitem(last=False)
    return r


# ----------------------------------------------------------------------------- helpers
DOC_COLS = "doc_id, collection, title, authors, year, n_pages, n_images, topic, url, file"


def docs_by_id(ids):
    if not ids:
        return {}
    q = f"SELECT {DOC_COLS} FROM docs WHERE doc_id IN ({','.join('?' * len(ids))})"
    return {r["doc_id"]: dict(r) for r in db().execute(q, list(ids))}


def where_filters(collection, topic, year_from, year_to, alias="d"):
    w, p = [], []
    if collection:
        cs = collection.split(",")
        w.append(f"{alias}.collection IN ({','.join('?' * len(cs))})"); p += cs
    if topic is not None and topic >= 0:
        w.append(f"{alias}.topic=?"); p.append(topic)
    if year_from:
        w.append(f"{alias}.year>=?"); p.append(year_from)
    if year_to:
        w.append(f"{alias}.year<=?"); p.append(year_to)
    return w, p


def doc_passes(d, collection, topic, year_from, year_to):
    if collection and d["collection"] not in collection.split(","):
        return False
    if topic is not None and topic >= 0 and d["topic"] != topic:
        return False
    if year_from and (d["year"] or 0) < year_from:
        return False
    if year_to and (d["year"] or 9999) > year_to:
        return False
    return True


def page_out(items, offset, limit, total=None, **extra):
    sl = items[offset:offset + limit]
    nxt = offset + limit if offset + limit < len(items) else None
    return dict(items=sl, offset=offset, next=nxt, total=len(items) if total is None else total, **extra)


# ----------------------------------------------------------------------------- API
@app.get("/api/progress")
def progress():
    f = C.WORK / "progress.json"
    if not f.exists():
        return JSONResponse(dict(indexed_pct=0, stages=[], waiting=True))
    return Response(f.read_bytes(), media_type="application/json", headers={"Cache-Control": "no-store"})


_STATS = [0.0, None]
_STATS_CAT = {}


@app.get("/api/stats")
def stats():
    ensure()
    if _STATS[1] is not None and time.time() - _STATS[0] < 20:
        return _STATS[1]
    con = db()
    r = con.execute("SELECT count(*), sum(n_pages), sum(n_images), sum(bytes) FROM docs").fetchone()
    colls = [dict(r) for r in con.execute(
        "SELECT collection, count(*) n, sum(n_pages) pages, sum(n_images) images, sum(n_text_pages=0) scans "
        "FROM docs GROUP BY collection ORDER BY n DESC")]
    years = [dict(r) for r in con.execute(
        "SELECT year, count(*) n FROM docs WHERE year BETWEEN 1700 AND 2030 GROUP BY year ORDER BY year")]
    if "cat" not in _STATS_CAT:
        import pandas as pd
        _STATS_CAT["cat"] = pd.read_parquet(C.CATALOG, columns=["collection"]).collection.value_counts().to_dict()
    out = dict(catalog_collections=_STATS_CAT["cat"], docs=r[0], pages=r[1] or 0, images=r[2] or 0, bytes=r[3] or 0, chunks=S.text.committed(),
               image_vectors=S.image.committed(), collections=colls, years=years, topics=topics(),
               title=C.TITLE, all_collections=C.collections(), models=dict(text=C.TEXT_MODEL, image=C.IMAGE_MODEL))
    _STATS[:] = [time.time(), out]
    return out


_MAPS = {}
_COLLS = []


def _collections():
    if not _COLLS:
        _COLLS.extend(C.collections())
    return _COLLS


def _map_bin(kind):
    """Binary map: u32 n | i32 ids[n] | u16 xy[n,2] | u8 topic+1[n] | u8 collection[n]"""
    con = db()
    colls = {c: i for i, c in enumerate(_collections())}
    if kind == "docs":
        rows = con.execute("SELECT doc_id, mx, my, topic, collection FROM docs WHERE mx IS NOT NULL").fetchall()
        extra = np.array([colls.get(r[4], 255) for r in rows], np.uint8)
    else:
        rows = con.execute("SELECT i.id, i.mx, i.my, i.topic, d.collection FROM images i JOIN docs d USING(doc_id) "
                           "WHERE i.mx IS NOT NULL").fetchall()
        extra = np.array([colls.get(r[4], 255) for r in rows], np.uint8)
    ids = np.array([r[0] for r in rows], np.int32)
    xy = (np.array([[r[1], r[2]] for r in rows], np.float32).reshape(-1, 2) * 65535).astype(np.uint16)
    top = np.array([max(r[3], -1) + 1 for r in rows], np.uint8)
    return struct.pack("<I", len(rows)) + ids.tobytes() + xy.tobytes() + top.tobytes() + extra.tobytes()


@app.get("/api/map/{kind}.bin")
def map_bin(kind: str):
    if kind not in ("docs", "images"):
        raise HTTPException(404)
    ensure()
    c = _MAPS.get(kind)
    if c is None or time.time() - c[0] > 30:  # the map grows: rebuild at most every 30 s
        c = _MAPS[kind] = (time.time(), _map_bin(kind))
    return Response(c[1], media_type="application/octet-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/search")
def search(q: str = "", mode: str = "semantic", offset: int = 0, limit: int = Query(20, le=60),
           collection: str = "", topic: int | None = None, year_from: int | None = None,
           year_to: int | None = None, like_chunk: int | None = None, like_image: int | None = None,
           like_doc: int | None = None):
    """Paginated search. mode: semantic (passages grouped per doc) | keyword (FTS5) | image | docs.
    Ranked lists are computed once per (query, filters, store size) and cached; pages are slices."""
    ensure()
    t0 = time.time()
    q = q.strip()
    filt = (collection, topic, year_from, year_to)
    if mode == "keyword":
        return _search_keyword(q, offset, limit, *filt) | dict(ms=int((time.time() - t0) * 1000))
    # the store grows; a cursor stays stable within one 'generation' of the cached list
    gen = {"image": S.image, "docs": None}.get(mode, S.text)
    if gen is not None:
        gen.refresh()
    else:
        docvecs()
    size = gen.n if gen is not None else (S.docvec[1] if S.docvec else 0)
    key = (q, mode, filt, like_chunk, like_image, like_doc)
    if offset == 0:  # a fresh query always sees the newest data
        with _RES_LOCK:
            _RES.pop(key, None)
    fn = {"image": lambda: _search_images(q, like_image, *filt),
          "docs": lambda: _search_docs(q, like_doc, *filt)}.get(mode, lambda: _search_passages(q, like_chunk, *filt))
    res = cached_results(key, fn)
    return page_out(res, offset, limit) | dict(ms=int((time.time() - t0) * 1000), mode=mode, searched=size)


def _fetch(table, cols, ids):
    out, con = {}, db()
    ids = [int(i) for i in ids]
    for i in range(0, len(ids), 900):
        part = ids[i:i + 900]
        for r in con.execute(f"SELECT {cols} FROM {table} WHERE id IN ({','.join('?' * len(part))})", part):
            out[r[0]] = r
    return out


def _filtered(collection, topic, yf, yt):
    return bool(collection or (topic is not None and topic >= 0) or yf or yt)


def _search_passages(q, like_chunk, collection, topic, yf, yt, depth=1000):
    if like_chunk is not None:
        v = S.text.vector(like_chunk)
    elif q:
        v = qvec_text(q)
    else:
        return []
    ids, sims = S.text.search(v, depth * (4 if _filtered(collection, topic, yf, yt) else 1))
    rows = _fetch("chunks", "id, doc_id, page, kind, text", ids)
    docs = docs_by_id({r[1] for r in rows.values()})
    grouped, order = {}, []
    for i, sc in zip(ids.tolist(), sims.tolist()):
        r = rows.get(i)
        if r is None or i == like_chunk:
            continue
        doc = docs.get(r[1])
        if doc is None or not doc_passes(doc, collection, topic, yf, yt):
            continue
        g = grouped.get(r[1])
        hit = dict(chunk=i, page=r[2], kind=r[3], text=r[4][:700], score=round(sc, 4))
        if g is None:
            g = grouped[r[1]] = dict(doc=doc, hits=[], score=hit["score"])
            order.append(r[1])
        if len(g["hits"]) < 4:
            g["hits"].append(hit)
    return [grouped[d] for d in order]


def _search_images(q, like_image, collection, topic, yf, yt, depth=900):
    if like_image is not None:
        v = S.image.vector(like_image)
    elif q:
        v = qvec_image(q)
    else:
        return []
    ids, sims = S.image.search(v, depth * (3 if _filtered(collection, None, yf, yt) else 1), valid=image_ok_mask)
    meta = {k: dict(zip(("id", "doc_id", "page", "x0", "y0", "x1", "y1", "w", "h", "topic", "ok"), r))
            for k, r in _fetch("images", "id, doc_id, page, x0, y0, x1, y1, w, h, topic, ok", ids).items()}
    docs = docs_by_id({m["doc_id"] for m in meta.values()})
    out, seen = [], set()
    for i, sc in zip(ids.tolist(), sims.tolist()):
        m = meta.get(i)
        if m is None or i == like_image or not m["ok"]:
            continue
        doc = docs.get(m["doc_id"])
        if doc is None or not doc_passes(doc, collection, None, yf, yt):
            continue
        if topic is not None and topic >= 0 and m["topic"] != topic:
            continue
        k = (m["doc_id"], m["page"], round(m["x0"], 2), round(m["y0"], 2))  # same picture reused on a page
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(image=m, doc=dict(doc_id=doc["doc_id"], title=doc["title"], collection=doc["collection"],
                                          year=doc["year"]), score=round(sc, 4)))
    return out


def _search_docs(q, like_doc, collection, topic, yf, yt, k=1000):
    V = docvecs()
    if not len(V):
        return []
    if like_doc is not None:
        r = db().execute("SELECT vrow FROM docs WHERE doc_id=?", (like_doc,)).fetchone()
        if not r or r[0] is None or r[0] >= len(V):
            return []
        v = V[r[0]]
    elif q:
        v = qvec_text(q)
    else:
        return []
    sims = V @ v
    kk = min(len(sims), k * 3)
    top = np.argpartition(-sims, kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    con = db()
    vr = top.tolist()
    rows = {}
    for i in range(0, len(vr), 900):
        part = vr[i:i + 900]
        for r in con.execute(f"SELECT vrow, {DOC_COLS} FROM docs WHERE vrow IN ({','.join('?' * len(part))})", part):
            rows[r[0]] = {k2: r[k2] for k2 in r.keys() if k2 != "vrow"}
    out = []
    for t in vr:
        d = rows.get(t)
        if d is None or d["doc_id"] == like_doc or not doc_passes(d, collection, topic, yf, yt):
            continue
        out.append(dict(doc=d, score=round(float(sims[t]), 4), hits=[]))
        if len(out) >= k:
            break
    return out


def _fts_query(q):
    # plain words -> AND of quoted tokens; power users may pass quoted phrases / OR / NOT / prefix*
    if any(t in q for t in ('"', " OR ", " NOT ", "*", "NEAR(")):
        return q
    toks = [t for t in "".join(ch if ch.isalnum() else " " for ch in q).split() if t]
    return " ".join(f'"{t}"' for t in toks)


def _search_keyword(q, offset, limit, collection, topic, yf, yt):
    if not q:
        return dict(items=[], offset=0, next=None, total=0, mode="keyword")
    w, p = where_filters(collection, topic, yf, yt)
    extra = (" AND " + " AND ".join(w)) if w else ""
    fq = _fts_query(q)
    # bm25 over at most 5000 matches, then page through them (bounded cost even for very common words)
    sql = f"""
      WITH m AS (SELECT rowid id, rank, snippet(chunks_fts, 0, char(2), char(3), ' … ', 40) snip
                 FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 5000)
      SELECT c.id, c.doc_id, c.page, c.kind, m.snip, m.rank
      FROM m JOIN chunks c ON c.id = m.id JOIN docs d ON d.doc_id = c.doc_id
      WHERE 1 {extra} ORDER BY m.rank LIMIT ? OFFSET ?"""
    try:
        rows = db().execute(sql, [fq, *p, limit + 1, offset]).fetchall()
    except sqlite3.OperationalError as e:
        raise HTTPException(400, f"bad query: {e}")
    has_more = len(rows) > limit
    rows = rows[:limit]
    docs = docs_by_id({r[1] for r in rows})
    items = [dict(doc=docs[r[1]], score=round(-r[5], 3),
                  hits=[dict(chunk=r[0], page=r[2], kind=r[3], text=r[4], score=round(-r[5], 3))]) for r in rows]
    return dict(items=items, offset=offset, next=offset + limit if has_more else None, total=None, mode="keyword")


@app.get("/api/brief")
def brief(ids: str = "", kind: str = "docs"):
    """Small hover cards for map points (ids comma separated, <=50)."""
    ii = [int(x) for x in ids.split(",") if x.strip().lstrip("-").isdigit()][:50]
    if kind == "images":
        if not ii:
            return []
        rows = db().execute(
            "SELECT i.id, i.doc_id, i.page, d.title, d.collection, d.year, i.topic FROM images i JOIN docs d USING(doc_id) "
            f"WHERE i.id IN ({','.join('?' * len(ii))})", ii).fetchall()
        return [dict(id=r[0], doc_id=r[1], page=r[2], title=r[3], collection=r[4], year=r[5], topic=r[6]) for r in rows]
    return list(docs_by_id(ii).values())


_SAMPLES = {}


@app.get("/api/topic/{kind}/{k}")
def topic_samples(kind: str, k: int, n: int = Query(6, le=24)):
    """A few random members of a topic (titles for text topics, crops for image topics)."""
    ver = topics().get("text" if kind == "docs" else "image", {}).get("version")
    key = (kind, k, n, ver)
    if key in _SAMPLES and time.time() - _SAMPLES[key][0] < 120:
        return _SAMPLES[key][1]
    con = db()
    if kind == "images":
        rows = con.execute("SELECT i.id, i.doc_id, i.page, i.w, i.h FROM images i WHERE i.topic=? AND i.ok=1 "
                           "ORDER BY random() LIMIT ?", (k, n)).fetchall()
        out = [dict(id=r[0], doc_id=r[1], page=r[2], w=r[3], h=r[4]) for r in rows]
    else:
        out = [dict(r) for r in con.execute(f"SELECT {DOC_COLS} FROM docs WHERE topic=? ORDER BY random() LIMIT ?", (k, n))]
    _SAMPLES[key] = (time.time(), out)
    return out


@app.get("/api/docs")
def browse_docs(cursor: int = -1, limit: int = Query(40, le=100), collection: str = "", topic: int | None = None,
                year_from: int | None = None, year_to: int | None = None, has_images: bool = False):
    """Keyset pagination over the catalogue (cursor = last doc_id seen)."""
    w, p = where_filters(collection, topic, year_from, year_to)
    w.append("d.doc_id > ?"); p.append(cursor)
    if has_images:
        w.append("d.n_images > 0")
    rows = [dict(r) for r in db().execute(
        f"SELECT {DOC_COLS.replace('doc_id', 'd.doc_id')} FROM docs d WHERE {' AND '.join(w)} ORDER BY d.doc_id LIMIT ?",
        [*p, limit + 1])]
    nxt = rows[limit - 1]["doc_id"] if len(rows) > limit else None
    return dict(items=rows[:limit], next=nxt)


@app.get("/api/doc/{doc_id}")
def doc(doc_id: int):
    con = db()
    r = con.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    if not r:
        raise HTTPException(404, "not indexed yet")
    d = dict(r)
    d.pop("path", None)
    pages = [p for (p,) in con.execute("SELECT DISTINCT page FROM chunks WHERE doc_id=? AND kind=0 ORDER BY page", (doc_id,))]
    imgs = [dict(x) for x in con.execute("SELECT id, page, x0,y0,x1,y1,w,h, topic FROM images WHERE doc_id=? AND ok=1 ORDER BY page, y0 LIMIT 400", (doc_id,))]
    d["text_pages"] = pages
    d["images"] = imgs
    return d


@app.get("/api/doc/{doc_id}/text")
def doc_text(doc_id: int, page: int):
    rows = db().execute("SELECT id, text FROM chunks WHERE doc_id=? AND page=? AND kind=0 ORDER BY id", (doc_id, page)).fetchall()
    return dict(chunks=[dict(id=r[0], text=r[1]) for r in rows])


# ----------------------------------------------------------------------------- rendering (separate processes: MuPDF is not thread-safe)
_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(int(os.environ.get("PDFATLAS_RENDER_WORKERS", 3)))
    return _POOL


def _render(path, page, width, clip):
    import pymupdf
    pymupdf.TOOLS.mupdf_display_errors(False)
    with pymupdf.open(path) as d:
        if page < 0 or page >= d.page_count:
            return None
        pg = d[page]
        r = pg.rect
        if clip:
            r = pymupdf.Rect(r.x0 + clip[0] * r.width, r.y0 + clip[1] * r.height,
                             r.x0 + clip[2] * r.width, r.y0 + clip[3] * r.height)
        z = max(0.1, min(4.0, width / max(1.0, r.width)))
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r if clip else None, alpha=False)
        return pix.tobytes("jpeg", jpg_quality=82)


_IMGCACHE = OrderedDict()
_IMGCACHE_BYTES = [0]
_IMG_LOCK = threading.Lock()
MAX_CACHE = 256 * 2**20


async def _render_cached(key, path, page, width, clip):
    import asyncio
    with _IMG_LOCK:
        if key in _IMGCACHE:
            _IMGCACHE.move_to_end(key); return _IMGCACHE[key]
    b = await asyncio.get_running_loop().run_in_executor(pool(), _render, path, page, width, clip)
    if b is None:
        raise HTTPException(404)
    with _IMG_LOCK:
        _IMGCACHE[key] = b; _IMGCACHE_BYTES[0] += len(b)
        while _IMGCACHE_BYTES[0] > MAX_CACHE:
            _, old = _IMGCACHE.popitem(last=False); _IMGCACHE_BYTES[0] -= len(old)
    return b


def _path(doc_id):
    r = db().execute("SELECT path FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    return r[0]


IMG_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


@app.get("/api/render/{doc_id}/{page}.jpg")
async def render_page(doc_id: int, page: int, w: int = Query(480, ge=64, le=1800)):
    w = min(1800, (w + 63) // 64 * 64)  # bucket widths -> better cache hits
    b = await _render_cached(("p", doc_id, page, w), _path(doc_id), page, w, None)
    return Response(b, media_type="image/jpeg", headers=IMG_HEADERS)


@app.get("/api/image/{img_id}.jpg")
async def render_image(img_id: int, w: int = Query(320, ge=64, le=1200)):
    r = db().execute("SELECT i.doc_id, i.page, x0, y0, x1, y1, d.path FROM images i JOIN docs d USING(doc_id) WHERE i.id=?", (img_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    w = (w + 63) // 64 * 64
    b = await _render_cached(("i", img_id, w), r[6], r[1], w, (r[2], r[3], r[4], r[5]))
    return Response(b, media_type="image/jpeg", headers=IMG_HEADERS)


@app.get("/api/pdf/{doc_id}")
def pdf(doc_id: int):
    return FileResponse(_path(doc_id), media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


@app.get("/progress")
def progress_page():
    return FileResponse(STATIC / "progress.html")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
