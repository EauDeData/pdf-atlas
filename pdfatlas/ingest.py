"""Stage 4 (streaming): merge finished shards into the live store while the explorer is serving.

  docs.sqlite (WAL)   docs / chunks / images / chunks_fts, readable by the server at any time
  index/text.f16      append-only [N_chunks, TEXT_DIM] fp16, row i == chunks.id i
  index/image.f16     append-only [N_images, D] fp16, row i == images.id i
  index/docvec.f16    append-only per-document mean vectors, row == docs.vrow
  index/{kind}_seg_XXX.ann   immutable Annoy segments over [start, end) rows; the unsealed tail is
                      brute-forced by the server. Segments are listed in the `segments` table.
"""
import argparse
import sqlite3
import time

import numpy as np
import pandas as pd

from . import config as C
from .work import EXTRACT_DONE

SEG = {"text": 400_000, "image": 200_000}
TREES = 32

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(doc_id INTEGER PRIMARY KEY, path TEXT, collection TEXT, file TEXT, bytes INTEGER,
    title TEXT, authors TEXT, url TEXT, abstract TEXT, year INTEGER,
    n_pages INTEGER, n_text_pages INTEGER, n_images INTEGER, error TEXT,
    vrow INTEGER, topic INTEGER DEFAULT -1, mx REAL, my REAL, t_ingest REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, kind INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, xref INTEGER,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, w INTEGER, h INTEGER, ok INTEGER, topic INTEGER DEFAULT -1, mx REAL, my REAL);
CREATE TABLE IF NOT EXISTS ingested(kind TEXT, bid INTEGER, t REAL, PRIMARY KEY(kind, bid));
CREATE TABLE IF NOT EXISTS segments(kind TEXT, seg INTEGER, start INTEGER, end INTEGER, path TEXT, PRIMARY KEY(kind, seg));
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, page);
CREATE INDEX IF NOT EXISTS images_doc ON images(doc_id, page);
CREATE INDEX IF NOT EXISTS images_topic ON images(topic, id);
CREATE INDEX IF NOT EXISTS docs_coll ON docs(collection, doc_id);
CREATE INDEX IF NOT EXISTS docs_topic ON docs(topic, doc_id);
CREATE INDEX IF NOT EXISTS docs_vrow ON docs(vrow);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
"""


def connect(readonly=False):
    if readonly:
        con = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True, check_same_thread=False, timeout=60)
    else:
        con = sqlite3.connect(C.DB, timeout=120)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=120000")
    return con


def _rows(path):
    return path.stat().st_size if path.exists() else 0


def _append(path, arr):
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def _truncate_to(path, nbytes):
    """Crash safety: vector files may be ahead of the DB after a crash -> cut them back."""
    if path.exists() and path.stat().st_size > nbytes:
        with open(path, "r+b") as f:
            f.truncate(nbytes)


def recover(con):
    n_chunks = con.execute("SELECT coalesce(max(id)+1,0) FROM chunks").fetchone()[0]
    n_imgs = con.execute("SELECT coalesce(max(id)+1,0) FROM images").fetchone()[0]
    n_vrow = con.execute("SELECT coalesce(max(vrow)+1,0) FROM docs").fetchone()[0]
    _truncate_to(C.IDX / "text.f16", n_chunks * C.TEXT_DIM * 2)
    _truncate_to(C.IDX / "docvec.f16", n_vrow * C.TEXT_DIM * 2)
    if n_imgs:
        d = image_dim()
        _truncate_to(C.IDX / "image.f16", n_imgs * d * 2)
    return n_chunks, n_imgs, n_vrow


def image_dim():
    f = C.IDX / "image.dim"
    return int(f.read_text()) if f.exists() else 0


def ingest_text(con, cat, bid, state):
    t = pd.read_parquet(C.SHARDS / f"text_{bid:05d}.parquet")
    e = np.load(C.EMB / f"text_{bid:05d}.npy")
    assert len(e) == len(t), bid
    d = pd.read_parquet(C.SHARDS / f"docs_{bid:05d}.parquet").set_index("doc_id")
    meta = cat.loc[d.index]
    # per-document mean vector (body chunks + metadata chunk)
    did = t.doc_id.to_numpy()
    uniq, inv = np.unique(did, return_inverse=True)
    dv = np.zeros((len(uniq), C.TEXT_DIM), np.float32)
    np.add.at(dv, inv, e.astype(np.float32))
    dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8
    vrow = {int(u): state["vrow"] + i for i, u in enumerate(uniq)}
    c0 = state["chunks"]
    ids = range(c0, c0 + len(t))
    now = time.time()
    with con:
        con.executemany(
            "INSERT OR REPLACE INTO docs(doc_id,path,collection,file,bytes,title,authors,url,abstract,year,n_pages,"
            "n_text_pages,n_images,error,vrow,t_ingest) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(int(i), m.path, m.collection, m.file, int(m.bytes), _title(m, d.loc[i]), m.authors, m.url, m.abstract,
              None if pd.isna(m.year) else int(m.year), int(d.loc[i, "n_pages"]), int(d.loc[i, "n_text_pages"]),
              int(d.loc[i, "n_images"]), d.loc[i, "error"], vrow.get(int(i)), now)
             for i, m in meta.iterrows()])
        con.executemany("INSERT INTO chunks(id,doc_id,page,kind,text) VALUES (?,?,?,?,?)",
                        zip(ids, t.doc_id.tolist(), t.page.tolist(), t.kind.tolist(), t.text.tolist()))
        con.executemany("INSERT INTO chunks_fts(rowid,text) VALUES (?,?)", zip(ids, t.text.tolist()))
        con.execute("INSERT INTO ingested VALUES ('text',?,?)", (bid, now))
        # vectors are appended inside the transaction window; recover() trims them if we die before commit
        _append(C.IDX / "text.f16", e.astype(np.float16))
        _append(C.IDX / "docvec.f16", dv.astype(np.float16))
    state["chunks"] += len(t)
    state["vrow"] += len(uniq)
    return len(t)


def _title(m, d):
    """Harvester title, unless it is just the filename and the PDF carries a better one."""
    pt = (d.get("pdf_title") or "").strip()
    fn_title = m.file[:-4].split("_", 1)[-1].replace("_", " ")
    if pt and len(pt) > 8 and m.title.strip() == fn_title.strip() and not pt.lower().startswith(("microsoft", "untitled")):
        return pt
    return m.title


def ingest_images(con, bid, state):
    df = pd.read_parquet(C.SHARDS / f"img_{bid:05d}.parquet")
    p = C.EMB / f"img_{bid:05d}.npy"
    with con:
        if len(df):
            e = np.load(p)
            ok = np.load(C.EMB / f"img_{bid:05d}.ok.npy")
            if not image_dim():
                (C.IDX / "image.dim").write_text(str(e.shape[1]))
            i0 = state["images"]
            con.executemany("INSERT INTO images(id,doc_id,page,xref,x0,y0,x1,y1,w,h,ok) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            [(i0 + j, int(r.doc_id), int(r.page), int(r.xref), float(r.x0), float(r.y0), float(r.x1),
                              float(r.y1), int(r.w), int(r.h), int(ok[j])) for j, r in enumerate(df.itertuples())])
            _append(C.IDX / "image.f16", e.astype(np.float16))
            state["images"] += len(df)
        con.execute("INSERT INTO ingested VALUES ('image',?,?)", (bid, time.time()))
    return len(df)


def seal_segments(con, kind, n_total, final=False):
    """Freeze full SEG-sized ranges (or, at the very end, the remainder) into Annoy segments."""
    from annoy import AnnoyIndex
    dim = C.TEXT_DIM if kind == "text" else image_dim()
    if not dim:
        return
    raw = C.IDX / f"{kind}.f16"
    while True:
        r = con.execute("SELECT coalesce(max(seg)+1,0), coalesce(max(end),0) FROM segments WHERE kind=?", (kind,)).fetchone()
        seg, start = r
        end = min(n_total, start + SEG[kind])
        if end - start < (1 if final else SEG[kind]):
            return
        t0 = time.time()
        X = np.memmap(raw, np.float16, "r", shape=(n_total, dim))
        ok = None
        if kind == "image":
            ok = np.zeros(end - start, bool)
            for (i,) in con.execute("SELECT id FROM images WHERE ok=1 AND id>=? AND id<?", (start, end)):
                ok[i - start] = True
        out = C.IDX / f"{kind}_seg_{seg:03d}.ann"
        a = AnnoyIndex(dim, "angular")
        a.on_disk_build(str(out) + ".tmp")
        blk = np.asarray(X[start:end], np.float32)
        for j, v in enumerate(blk):
            if ok is None or ok[j]:
                a.add_item(j, v)
        a.build(TREES, n_jobs=8)
        a.unload()
        (C.IDX / (out.name + ".tmp")).replace(out)
        with con:
            con.execute("INSERT INTO segments VALUES (?,?,?,?,?)", (kind, seg, start, end, out.name))
        print(f"sealed {kind} segment {seg} [{start},{end}) in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=15)
    a = ap.parse_args()
    C.IDX.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.executescript(SCHEMA)
    n_chunks, n_imgs, n_vrow = recover(con)
    state = dict(chunks=n_chunks, images=n_imgs, vrow=n_vrow)
    cat = pd.read_parquet(C.CATALOG).set_index("doc_id")
    print("resuming at", state, flush=True)
    while True:
        done = {(k, b) for k, b in con.execute("SELECT kind, bid FROM ingested")}
        did = 0
        for f in sorted(C.EMB.glob("text_*.npy")):
            if ".tmp" in f.name:
                continue
            bid = int(f.stem.split("_")[1])
            if ("text", bid) not in done:
                ingest_text(con, cat, bid, state); did += 1
        done = {(k, b) for k, b in con.execute("SELECT kind, bid FROM ingested")}
        for f in sorted(C.EMB.glob("img_*.npy")):
            if ".tmp" in f.name or ".ok" in f.name:
                continue
            bid = int(f.stem.split("_")[1])
            # images reference docs -> only after that shard's text is in
            if ("image", bid) not in done and ("text", bid) in done:
                ingest_images(con, bid, state); did += 1
        if did:
            print(f"ingested {did} shards -> chunks={state['chunks']} images={state['images']} docs={state['vrow']}", flush=True)
        n_batches = int((C.SHARDS / "N_BATCHES").read_text()) if (C.SHARDS / "N_BATCHES").exists() else -1
        all_done = EXTRACT_DONE.exists() and con.execute(
            "SELECT count(*) FROM ingested WHERE kind='image'").fetchone()[0] >= n_batches
        seal_segments(con, "text", state["chunks"], final=all_done)
        seal_segments(con, "image", state["images"], final=all_done)
        if all_done:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            con.commit()
            (C.WORK / "INGEST_DONE").write_text(time.ctime())
            print("all shards ingested; final segments sealed", flush=True)
            return
        if not did:
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
