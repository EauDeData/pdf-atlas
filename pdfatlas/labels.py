"""User labels (categories) drawn on the map, stored per document *path* in WORK/labels.sqlite.

A label is attached to a whole PDF, not to a passage or a figure box. One document can carry any
number of labels (1:N): lasso a region of the figure map and every document owning a figure inside
it gets the label; lasso the document map and every document inside it does.

Kept in its own SQLite file, written only by the server:
  * the pipeline never contends with it (ingest holds long write locks on docs.sqlite);
  * keyed by path, so labels survive a rebuild of the index, where doc_ids may be renumbered.

  selections(id, label, kind, polygon, n_points, n_docs, t, map_version, note)
      one row per lasso (or manual add); polygon = [[x,y],...] in map coordinates (0..1),
      map_version = topics.json version of the layout it was drawn on
  doc_labels(path, label, selection_id, doc_id, t)
      which selection put which label on which document
  doc_categories (view): path, label, doc_id, first_t, n_selections  -> one row per (document, label)

Query it from anywhere:
    sqlite3 $PDFATLAS_WORK/labels.sqlite "SELECT label, count(*) FROM doc_categories GROUP BY label"
"""
import json
import sqlite3
import time

from . import config as C

DB = C.WORK / "labels.sqlite"
MAX_LABEL = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS selections(id INTEGER PRIMARY KEY, label TEXT NOT NULL, kind TEXT NOT NULL,
    polygon TEXT, n_points INTEGER, n_docs INTEGER, t REAL, map_version INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS doc_labels(path TEXT NOT NULL, label TEXT NOT NULL, selection_id INTEGER NOT NULL,
    doc_id INTEGER, t REAL, PRIMARY KEY(path, label, selection_id));
CREATE INDEX IF NOT EXISTS doc_labels_label ON doc_labels(label, path);
CREATE INDEX IF NOT EXISTS doc_labels_sel ON doc_labels(selection_id);
CREATE VIEW IF NOT EXISTS doc_categories AS
    SELECT path, label, max(doc_id) doc_id, min(t) first_t, count(*) n_selections FROM doc_labels GROUP BY path, label;
"""


def connect():
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(SCHEMA)
    return con


def clean_label(s):
    s = " ".join((s or "").split())
    if not s or len(s) > MAX_LABEL:
        raise ValueError(f"label must be 1-{MAX_LABEL} characters")
    return s


def add(label, kind, docs, n_points, polygon=None, map_version=None, note=""):
    """docs: {doc_id: path}. Returns the selection row as a dict (n_new = documents that did not
    carry this label before)."""
    label = clean_label(label)
    now = time.time()
    con = connect()
    try:
        with con:
            before = {p for (p,) in con.execute(
                "SELECT DISTINCT path FROM doc_labels WHERE label=?", (label,))} if docs else set()
            cur = con.execute(
                "INSERT INTO selections(label, kind, polygon, n_points, n_docs, t, map_version, note) VALUES (?,?,?,?,?,?,?,?)",
                (label, kind, json.dumps(polygon) if polygon else None, n_points, len(docs), now, map_version, note))
            sid = cur.lastrowid
            con.executemany("INSERT OR IGNORE INTO doc_labels(path, label, selection_id, doc_id, t) VALUES (?,?,?,?,?)",
                            [(p, label, sid, int(d), now) for d, p in docs.items()])
        return dict(selection_id=sid, label=label, kind=kind, n_points=n_points, n_docs=len(docs),
                    n_new=len(set(docs.values()) - before))
    finally:
        con.close()


def summary():
    con = connect()
    try:
        labels = [dict(label=l, n_docs=n, n_selections=s, last_t=t) for l, n, s, t in con.execute(
            "SELECT label, count(DISTINCT path), count(DISTINCT selection_id), max(t) FROM doc_labels "
            "GROUP BY label ORDER BY max(t) DESC")]
        sels = [dict(id=i, label=l, kind=k, n_points=p, n_docs=n, t=t, note=note) for i, l, k, p, n, t, note in con.execute(
            "SELECT id, label, kind, n_points, n_docs, t, note FROM selections "
            "WHERE id IN (SELECT DISTINCT selection_id FROM doc_labels) ORDER BY id DESC LIMIT 20")]
        return dict(labels=labels, selections=sels)
    finally:
        con.close()


def polygons(kind):
    """Lassoes still in effect (not undone) drawn on one map: [{id, label, polygon, map_version, t, n_docs}]."""
    con = connect()
    try:
        return [dict(id=i, label=l, polygon=json.loads(p), map_version=v, t=t, n_docs=n) for i, l, p, v, t, n in con.execute(
            "SELECT id, label, polygon, map_version, t, n_docs FROM selections WHERE kind=? AND polygon IS NOT NULL "
            "AND id IN (SELECT DISTINCT selection_id FROM doc_labels) ORDER BY id", (kind,))]
    finally:
        con.close()


def paths_with(label):
    con = connect()
    try:
        return [p for (p,) in con.execute("SELECT DISTINCT path FROM doc_labels WHERE label=?", (label,))]
    finally:
        con.close()


def for_path(path):
    con = connect()
    try:
        return [l for (l,) in con.execute("SELECT label FROM doc_categories WHERE path=? ORDER BY first_t", (path,))]
    finally:
        con.close()


def remove(label=None, path=None, selection_id=None):
    """Delete by selection (undo a lasso), by (path, label) (one document), or by label (everywhere)."""
    con = connect()
    try:
        with con:
            if selection_id is not None:
                n = con.execute("DELETE FROM doc_labels WHERE selection_id=?", (selection_id,)).rowcount
            elif path is not None and label is not None:
                n = con.execute("DELETE FROM doc_labels WHERE path=? AND label=?", (path, label)).rowcount
            elif label is not None:
                n = con.execute("DELETE FROM doc_labels WHERE label=?", (label,)).rowcount
            else:
                raise ValueError("nothing to remove")
        return n
    finally:
        con.close()


def rows():
    """(path, label, doc_id, first_t) for CSV export, one row per (document, label)."""
    con = connect()
    try:
        return con.execute("SELECT path, label, doc_id, first_t FROM doc_categories ORDER BY label, path").fetchall()
    finally:
        con.close()
