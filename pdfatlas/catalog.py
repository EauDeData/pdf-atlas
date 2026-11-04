"""Stage 0 — catalogue: list every PDF of every source, attach whatever metadata exists.
No PDF is opened here (titles embedded in the PDFs are picked up later, during extraction).

Metadata is optional and declared per source (see examples/sources.example.json):

    {"type": "jsonl",  "path": ".../metadata.jsonl", "key": "file", "fields": {"title": "title", "url": "item"}}
    {"type": "csv",    "path": ".../meta.csv",       "key": "filename"}
    {"type": "sqlite", "path": ".../manifest.db",
     "query": "SELECT filepath AS file, title, authors, published AS date, abstract, pdf_url AS url FROM papers"}

Rows are matched to PDFs by the *basename* of the key column. Recognised output fields:
title, authors, date/year, abstract, url. Without metadata the title comes from the file name.
    python -m pdfatlas.catalog
"""
import fnmatch
import json
import os
import re
import sqlite3

import pandas as pd

from . import config as C

FIELDS = ("title", "authors", "date", "year", "abstract", "url")


def read_metadata(spec):
    if not spec:
        return {}
    kind, path = spec.get("type", "jsonl"), os.path.expanduser(spec["path"])
    if not os.path.exists(path):
        print(f"  ! metadata file not found: {path}")
        return {}
    key, fmap = spec.get("key", "file"), spec.get("fields", {})
    if kind == "jsonl":
        rows = []
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    elif kind == "csv":
        rows = pd.read_csv(path).to_dict("records")
    elif kind == "sqlite":
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = con.execute(spec["query"])
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur]
        key = spec.get("key", "file")
    else:
        raise ValueError(f"unknown metadata type {kind}")
    out = {}
    for r in rows:
        k = r.get(key)
        if not k:
            continue
        m = {}
        for f in FIELDS:
            v = r.get(fmap.get(f, f))
            if isinstance(v, list):
                v = "; ".join(map(str, v))
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                m[f] = str(v)
        out[os.path.basename(str(k))] = m
    return out


def iter_pdfs(src):
    root = os.path.expanduser(src["path"])
    excl = src["exclude"]
    if src["recursive"]:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not any(fnmatch.fnmatch(d, e) for e in excl)]
            for fn in fns:
                if fn.lower().endswith(".pdf") and not any(fnmatch.fnmatch(fn, e) for e in excl):
                    yield os.path.join(dp, fn)
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if fn.lower().endswith(".pdf") and os.path.isfile(p):
                yield p


def title_from_name(fn):
    t = re.sub(r"^\d+[_-]", "", fn[:-4]).replace("_", " ").strip()
    return t or fn


def year_of(s):
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", s or "")
    return int(m.group(1)) if m else None


def build():
    C.WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in C.load_sources():
        meta = read_metadata(src["metadata"])
        n0 = len(rows)
        for path in iter_pdfs(src):
            fn = os.path.basename(path)
            m = meta.get(fn, {})
            title = (m.get("title") or "").strip() or title_from_name(fn)
            rows.append(dict(
                path=path, collection=src["name"], file=fn, bytes=os.path.getsize(path),
                title=title[:400], authors=m.get("authors", "")[:400], url=m.get("url", ""),
                abstract=m.get("abstract", "")[:4000],
                year=year_of(m.get("year") or m.get("date")) or year_of(title)))
        print(f"{src['name']:20s} {len(rows) - n0:8d} pdfs  ({len(meta)} metadata rows)", flush=True)
    df = pd.DataFrame(rows).sort_values(["collection", "path"]).reset_index(drop=True)
    df["year"] = df["year"].astype("Int64")
    df.insert(0, "doc_id", range(len(df)))
    df.to_parquet(C.CATALOG, index=False)
    print("total", len(df), "->", C.CATALOG)


if __name__ == "__main__":
    build()
