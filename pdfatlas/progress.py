"""Pipeline progress: writes WORK/progress.json every few seconds (served at /api/progress) and,
with --watch, draws a live terminal dashboard (used in the tmux 'progress' window).
"""
import argparse
import shutil
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import deque

import pandas as pd

from . import config as C

HIST = deque(maxlen=400)  # (t, {stage: done})


class ShardStats:
    """Incrementally sums per-shard doc stats (pages / images) without re-reading old shards."""

    def __init__(self):
        self.seen, self.pages, self.imgs, self.docs, self.scans, self.errors = set(), 0, 0, 0, 0, 0

    def update(self):
        for f in C.SHARDS.glob("docs_*.parquet"):
            if f.name in self.seen:
                continue
            try:
                d = pd.read_parquet(f, columns=["n_pages", "n_images", "n_text_pages", "error"])
            except Exception:
                continue
            self.seen.add(f.name)
            self.docs += len(d); self.pages += int(d.n_pages.sum()); self.imgs += int(d.n_images.sum())
            self.scans += int((d.n_text_pages == 0).sum()); self.errors += int((d.error != "").sum())


def _count(glob, bad=(".tmp", ".ok")):
    return sum(1 for f in glob if not any(b in f.name for b in bad))


def gpu_lines(pids):
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except Exception:
        return []
    uu = {}
    for g in gpus:
        i, u, util, used, tot = [x.strip() for x in g.split(",")]
        uu[u] = dict(index=int(i), util=int(util), used=int(used), total=int(tot), ours=0)
    for a in apps:
        pid, u, mem = [x.strip() for x in a.split(",")]
        if int(pid) in pids and u in uu:
            uu[u]["ours"] += int(mem)
    return sorted([g for g in uu.values() if g["ours"]], key=lambda g: g["index"])


def cpu_usage(pids):
    """Our processes' CPU (sum of %CPU, 100 = one core) and thread count, plus machine load."""
    tot = th = 0.0
    try:
        out = subprocess.run(["ps", "-o", "pcpu=,nlwp=", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
        for line in out.split("\n"):
            if line.strip():
                a, b = line.split(); tot += float(a); th += int(b)
    except Exception:
        pass
    return dict(pct=round(tot), threads=int(th), procs=len(pids), load1=round(os.getloadavg()[0], 1), ncpu=os.cpu_count())


def our_pids():
    try:
        out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "pdfatlas"], capture_output=True, text=True).stdout
        return {int(x) for x in out.split()}
    except Exception:
        return set()


def snapshot(ss):
    ss.update()
    total_docs = len(pd.read_parquet(C.CATALOG, columns=["doc_id"]))
    nb = int((C.SHARDS / "N_BATCHES").read_text()) if (C.SHARDS / "N_BATCHES").exists() else max(1, -(-total_docs // 100))
    ext = len(ss.seen)
    te = _count(C.EMB.glob("text_*.npy")) if C.EMB.exists() else 0
    ie = _count(C.EMB.glob("img_*.npy")) if C.EMB.exists() else 0
    db = dict(docs=0, chunks=0, images=0, text_shards=0, image_shards=0, topic_docs=0, segments=[])
    if C.DB.exists():
        try:
            con = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True, timeout=5)
            db["docs"] = con.execute("SELECT count(*) FROM docs").fetchone()[0]
            db["chunks"] = con.execute("SELECT coalesce(max(id)+1,0) FROM chunks").fetchone()[0]
            db["images"] = con.execute("SELECT coalesce(max(id)+1,0) FROM images").fetchone()[0]
            db["text_shards"] = con.execute("SELECT count(*) FROM ingested WHERE kind='text'").fetchone()[0]
            db["image_shards"] = con.execute("SELECT count(*) FROM ingested WHERE kind='image'").fetchone()[0]
            db["topic_docs"] = con.execute("SELECT count(*) FROM docs WHERE topic>=0").fetchone()[0]
            db["segments"] = [dict(kind=k, seg=s, start=a, end=b) for k, s, a, b in
                              con.execute("SELECT kind, seg, start, end FROM segments ORDER BY kind, seg")]
            con.close()
        except sqlite3.Error:
            pass
    topics = {}
    if (C.MAP / "topics.json").exists():
        try:
            t = json.loads((C.MAP / "topics.json").read_text())
            topics = {k: dict(k=v["k"], fitted_on=v["fitted_on"], version=v["version"]) for k, v in t.items()}
        except Exception:
            pass
    stages = [
        dict(key="extract", label="Read PDF object trees", done=ext, total=nb, unit="shards"),
        dict(key="text_emb", label=f"Embed text · {C.TEXT_MODEL.split('/')[-1]}", done=te, total=nb, unit="shards"),
        dict(key="image_emb", label=f"Embed images · {C.IMAGE_MODEL.split('/')[-1]}", done=ie, total=nb, unit="shards"),
        dict(key="ingest", label="Ingest → SQLite + Annoy", done=db["image_shards"], total=nb, unit="shards"),
        dict(key="topics", label="Topics + map placement", done=db["topic_docs"], total=total_docs, unit="docs"),
    ]
    now = time.time()
    HIST.append((now, {s["key"]: s["done"] for s in stages}))
    for s in stages:
        s["pct"] = round(100 * s["done"] / max(1, s["total"]), 2)
        # rate over the last ~10 minutes
        old = next(((t, d) for t, d in HIST if now - t <= 600), None)
        if old and now - old[0] > 20 and s["done"] > old[1][s["key"]]:
            rate = (s["done"] - old[1][s["key"]]) / (now - old[0])
            s["rate_per_min"] = round(rate * 60, 2)
            s["eta_min"] = round((s["total"] - s["done"]) / rate / 60, 1)
    indexed = db["docs"]
    return dict(
        t=now, total_docs=total_docs, indexed_docs=indexed, indexed_pct=round(100 * indexed / max(1, total_docs), 2),
        extracted=dict(docs=ss.docs, pages=ss.pages, images=ss.imgs, scans=ss.scans, errors=ss.errors),
        db=db, topics=topics, stages=stages,
        done=dict(extract=(C.SHARDS / "EXTRACT_DONE").exists(), ingest=(C.WORK / "INGEST_DONE").exists(),
                  topics=(C.WORK / "TOPICS_DONE").exists()),
        gpus=gpu_lines(pids := our_pids()), cpu=cpu_usage(pids),
    )


def bar(p, w=36):
    f = p / 100 * w
    full = int(f)
    part = " ▏▎▍▌▋▊▉"[int((f - full) * 8)] if full < w else ""
    return "█" * full + part + " " * (w - full - len(part))


def render(s):
    R, D, B, X = "\033[38;2;255;59;47m", "\033[2m", "\033[1m", "\033[0m"
    o = ["\033[H\033[2J", f"{B}DOC EXPLORER · pipeline{X}   {D}{time.strftime('%H:%M:%S')}{X}\n"]
    o.append(f"  {R}{s['indexed_pct']:6.2f}%{X} searchable   {s['indexed_docs']:,} / {s['total_docs']:,} documents\n\n")
    for st in s["stages"]:
        eta = f"eta {st['eta_min']:.0f} min" if "eta_min" in st else ""
        rate = f"{st['rate_per_min']:.1f} {st['unit']}/min" if "rate_per_min" in st else ""
        bw = max(10, min(40, shutil.get_terminal_size((120, 40)).columns - 100))
        o.append(f"  {st['label'][:34]:34s} {R}{bar(st['pct'], bw)}{X} {st['pct']:6.2f}%  {D}{st['done']:,}/{st['total']:,} {rate}  {eta}{X}\n")
    e, d = s["extracted"], s["db"]
    o.append(f"\n  {D}extracted{X} {e['docs']:,} docs · {e['pages']:,} pages · {e['images']:,} image refs · {e['scans']:,} text-less · {e['errors']:,} errors\n")
    o.append(f"  {D}store    {X} {d['chunks']:,} chunks · {d['images']:,} images · {len(d['segments'])} annoy segments\n")
    for k, v in s["topics"].items():
        o.append(f"  {D}topics   {X} {k}: k={v['k']} fitted on {v['fitted_on']:,}\n")
    c = s.get("cpu", {})
    o.append(f"  {D}cpu      {X} ours {c.get('pct',0)/100:.1f} cores · {c.get('threads',0)} threads · {c.get('procs',0)} procs · machine load {c.get('load1')}/{c.get('ncpu')}\n")
    for g in s["gpus"]:
        o.append(f"  {D}gpu{g['index']}     {X} ours {g['ours']/1024:.1f} GB · total {g['used']/1024:.1f}/{g['total']/1024:.0f} GB · util {g['util']}%\n")
    sys.stdout.write("".join(o)); sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=float, default=5)
    a = ap.parse_args()
    ss = ShardStats()
    out = C.WORK / "progress.json"
    while True:
        try:
            s = snapshot(ss)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=5))
            os.replace(tmp, out)
            if a.watch:
                render(s)
        except Exception as e:  # keep the monitor alive no matter what
            print("progress error:", e, flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
