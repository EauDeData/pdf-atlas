"""Stage 5 (streaming, a posteriori): topics + 2D maps that follow the growing store.

Refits at milestones (the corpus is ingested in random order, so each milestone is a uniform sample):
  text : per-document vectors -> KMeans -> c-TF-IDF labels -> UMAP (fit on <=50k, transform rest)
  image: MiniBatchKMeans -> zero-shot SigLIP labels -> UMAP on a fixed 1/32 subsample (id % 32 == 0)
Between refits, new rows get nearest-centroid topics and UMAP.transform coordinates.
    python -m pdfatlas.topics
"""
import json
import os
import pickle
import re
import time
from collections import Counter

import numpy as np

from . import config as C
from .ingest import connect, image_dim

K_TEXT, K_IMG = 40, 30
MILESTONES = {"text": [1500, 8000, 30000, 80000], "image": [3000, 25000, 150000, 600000]}
IMG_MAP_MOD = 32
if os.environ.get("PDFATLAS_SMALL"):  # smoke tests
    MILESTONES, IMG_MAP_MOD = {"text": [100], "image": [500]}, 2
FINAL = 10**12  # milestone id of the last refit (after everything is ingested)
UMAP_FIT_MAX = 50_000

STOP = set("""
a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its
may more most no not of on or our she should so such than that the their them then there these they this those to
was we were what when where which while who will with would also used use using between both each other over only
same some well within without about after all any before being during under up very one two three new et al fig
figure table vol pp per de la el en y los las del se por con para una un es al lo como más o su sus le les des du
et est dans par pour sur au aux ne pas qui que ce cette sont il elle ou mais avec da do dos das em um uma não na no
os ao à é que és amb els pel per als d l s que i les una uns del dels va han ha més com o seu seva aquest aquesta
och att det som en ett är av för på med till den har de inte om ett vid kan eller från sig också samt även dess
van het een en is in op te dat die voor met zijn aan er niet ook als bij door om dan wordt worden zijn naar of uit
der die und das ist im den dem zu von mit sich des auf für nicht ein eine als auch es an werden aus er hat dass sie
nach wird bei einer um am sind noch wie einem über einen so zum war haben nur oder aber vor zur bis mehr durch man
http https www doi org pdf page
""".split())

# Zero-shot vocabulary used to *name* image clusters (not to detect anything). Domain-neutral by
# default; point PDFATLAS_IMAGE_CONCEPTS to a text file (one concept per line) to specialise it.
_DEFAULT_CONCEPTS = """bar chart|line chart|scatter plot|pie chart|box plot|histogram|heatmap|network graph|tree diagram|
flowchart|schematic diagram|block diagram|timeline|table of numbers|mathematical equation|map|satellite or aerial image|
floor plan|engineering drawing|electronic circuit|chemical structure|molecular model|microscope image|medical scan|
photo of people|portrait of a person|group photo|crowd|people in a meeting|classroom|historical black and white photo|
building exterior|interior of a room|city street|landscape|mountains|sea or river|forest|plants|flowers|food|
animals|birds|insects|vehicles|machinery|tools|laboratory equipment|computer screenshot|user interface|
logo|seal or coat of arms|signature|handwritten text|stamp|newspaper clipping|advertisement|poster|book cover|
cartoon|illustration|drawing or sketch|painting|decorative ornament|pattern or texture|3D rendering|photo of an object|
sports|art or sculpture|blank page or noise|scanned text page""".replace("\n", "")


def _concepts():
    f = os.environ.get("PDFATLAS_IMAGE_CONCEPTS")
    if f and os.path.exists(f):
        return [l.strip() for l in open(f) if l.strip()]
    return [c.strip() for c in _DEFAULT_CONCEPTS.split("|") if c.strip()]


IMG_CONCEPTS = _concepts()


def colors(k):
    return [f"hsl({int(360 * i / k + 12) % 360} 68% 62%)" for i in range(k)]


def norm01(Y, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(Y, 0.5, 0), np.percentile(Y, 99.5, 0)
    return np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1), lo, hi


def _save(name, obj):
    tmp = C.MAP / (name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, C.MAP / name)


def _load(name):
    p = C.MAP / name
    return pickle.load(open(p, "rb")) if p.exists() else None


def write_meta(kind, meta):
    f = C.MAP / "topics.json"
    allm = json.loads(f.read_text()) if f.exists() else {}
    allm[kind] = meta
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(allm, indent=5, ensure_ascii=False))
    os.replace(tmp, f)


def docvecs():
    n = (C.IDX / "docvec.f16").stat().st_size // (2 * C.TEXT_DIM) if (C.IDX / "docvec.f16").exists() else 0
    return np.memmap(C.IDX / "docvec.f16", np.float16, "r", shape=(n, C.TEXT_DIM)) if n else None


def ctfidf_labels(con, vrows_by_topic, k, per=200, rng=np.random.default_rng(0)):
    tok = re.compile(r"[^\W\d_]{3,}", re.U)
    tf = []
    for t in range(k):
        members = vrows_by_topic.get(t, np.array([], int))
        pick = rng.choice(members, min(per, len(members)), replace=False) if len(members) else []
        c = Counter()
        for vr in pick:
            for (txt,) in con.execute("SELECT c.text FROM docs d JOIN chunks c USING(doc_id) WHERE d.vrow=? "
                                      "ORDER BY c.kind DESC, c.page LIMIT 4", (int(vr),)):
                c.update(w for w in (x.lower() for x in tok.findall(txt)) if w not in STOP and len(w) < 25)
        tf.append(c)
    total = Counter()
    for c in tf:
        total.update(c)
    A = sum(total.values()) / max(1, k)
    out = []
    for c in tf:
        s = sum(c.values()) or 1
        sc = {w: (v / s) * np.log(1 + A / total[w]) for w, v in c.items() if v >= 3}
        out.append([w for w, _ in sorted(sc.items(), key=lambda x: -x[1])[:8]])
    return out


# ------------------------------------------------------------------------------------------ text
def fit_text(con, milestone):
    import umap
    from sklearn.cluster import KMeans
    t0 = time.time()
    V = np.asarray(docvecs(), np.float32)
    n = len(V)
    k = int(min(K_TEXT, max(8, n // 80)))
    km = KMeans(k, n_init=3, random_state=0).fit(V)
    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    lab = km.labels_
    rng = np.random.default_rng(0)
    fit_idx = np.sort(rng.choice(n, min(n, UMAP_FIT_MAX), replace=False))
    um = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=0, low_memory=True).fit(V[fit_idx])
    Y = np.zeros((n, 2), np.float32)
    Y[fit_idx] = um.embedding_
    rest = np.setdiff1d(np.arange(n), fit_idx)
    if len(rest):
        Y[rest] = um.transform(V[rest])
    Y, lo, hi = norm01(Y[:, :2])
    with con:
        con.executemany("UPDATE docs SET topic=?, mx=?, my=? WHERE vrow=?",
                        [(int(l), float(a), float(b), int(i)) for i, (l, (a, b)) in enumerate(zip(lab, Y))])
    by = {t: np.nonzero(lab == t)[0] for t in range(k)}
    labels = ctfidf_labels(con, by, k)
    _save("text_model.pkl", dict(cent=cent, umap=um, lo=lo, hi=hi, milestone=milestone, n=n))
    write_meta("text", dict(labels=labels, k=k, colors=colors(k), fitted_on=n, milestone=milestone,
                            version=int(time.time()), counts=np.bincount(lab, minlength=k).tolist()))
    print(f"text refit @ {n} docs, k={k} in {time.time()-t0:.0f}s", flush=True)


def assign_text(con, model):
    rows = con.execute("SELECT vrow FROM docs WHERE vrow IS NOT NULL AND topic=-1 ORDER BY vrow LIMIT 20000").fetchall()
    if not rows:
        return 0
    V = docvecs()
    vr = np.array([r[0] for r in rows])
    X = np.asarray(V[vr], np.float32)
    top = (X @ model["cent"].T).argmax(1)
    Y, _, _ = norm01(model["umap"].transform(X)[:, :2], model["lo"], model["hi"])
    with con:
        con.executemany("UPDATE docs SET topic=?, mx=?, my=? WHERE vrow=?",
                        [(int(t), float(a), float(b), int(i)) for i, t, (a, b) in zip(vr, top, Y)])
    return len(rows)


# ------------------------------------------------------------------------------------------ image
_SIG = None


def concept_vecs():
    global _SIG
    if _SIG is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from . import limit_torch
        from .embed_images import encode_text
        limit_torch()
        _SIG = encode_text(IMG_CONCEPTS, device="cpu").astype(np.float32)
    return _SIG


def image_matrix():
    d = image_dim()
    if not d:
        return None
    n = (C.IDX / "image.f16").stat().st_size // (2 * d)
    return np.memmap(C.IDX / "image.f16", np.float16, "r", shape=(n, d)) if n else None


def fit_image(con, milestone):
    import umap
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    X = image_matrix()
    ok = np.array([r[0] for r in con.execute("SELECT id FROM images WHERE ok=1 ORDER BY id")], int)
    ok = ok[ok < len(X)]
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(ok, min(len(ok), 250_000), replace=False))
    k = int(min(K_IMG, max(6, len(ok) // 150)))
    km = MiniBatchKMeans(k, batch_size=8192, n_init=3, random_state=0).fit(np.asarray(X[samp], np.float32))
    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    top = np.full(len(X), -1, np.int16)
    for i in range(0, len(ok), 200_000):
        ids = ok[i:i + 200_000]
        top[ids] = (np.asarray(X[ids], np.float32) @ cent.T).argmax(1)
    mp = ok[ok % IMG_MAP_MOD == 0]
    fit_idx = np.sort(rng.choice(mp, min(len(mp), 30_000), replace=False)) if len(mp) else mp
    um = umap.UMAP(n_neighbors=20, min_dist=0.08, metric="cosine", random_state=0, low_memory=True).fit(
        np.asarray(X[fit_idx], np.float32))
    Y = np.zeros((len(mp), 2), np.float32)
    pos = {int(j): i for i, j in enumerate(mp)}
    Y[[pos[int(j)] for j in fit_idx]] = um.embedding_
    rest = np.setdiff1d(mp, fit_idx)
    if len(rest):
        Y[[pos[int(j)] for j in rest]] = um.transform(np.asarray(X[rest], np.float32))
    Y, lo, hi = norm01(Y)
    with con:
        con.executemany("UPDATE images SET topic=? WHERE id=?", [(int(top[i]), int(i)) for i in ok])
        con.executemany("UPDATE images SET mx=?, my=? WHERE id=?", [(float(a), float(b), int(i)) for i, (a, b) in zip(mp, Y)])
    T = concept_vecs()
    sim = cent @ T.T
    sim = sim - sim.mean(0, keepdims=True)
    labels = [[IMG_CONCEPTS[j] for j in np.argsort(-sim[t])[:3]] for t in range(k)]
    _save("image_model.pkl", dict(cent=cent, umap=um, lo=lo, hi=hi, milestone=milestone))
    write_meta("image", dict(labels=labels, k=k, colors=colors(k), fitted_on=int(len(ok)), milestone=milestone,
                             version=int(time.time()), counts=np.bincount(top[ok], minlength=k).tolist()))
    print(f"image refit @ {len(ok)} imgs, k={k} in {time.time()-t0:.0f}s", flush=True)


def assign_image(con, model):
    rows = con.execute("SELECT id FROM images WHERE ok=1 AND topic=-1 ORDER BY id LIMIT 50000").fetchall()
    if not rows:
        return 0
    X = image_matrix()
    ids = np.array([r[0] for r in rows])
    ids = ids[ids < len(X)]
    V = np.asarray(X[ids], np.float32)
    top = (V @ model["cent"].T).argmax(1)
    mp = ids % IMG_MAP_MOD == 0
    with con:
        con.executemany("UPDATE images SET topic=? WHERE id=?", [(int(t), int(i)) for i, t in zip(ids, top)])
        if mp.any():
            Y, _, _ = norm01(model["umap"].transform(V[mp]), model["lo"], model["hi"])
            con.executemany("UPDATE images SET mx=?, my=? WHERE id=?", [(float(a), float(b), int(i)) for i, (a, b) in zip(ids[mp], Y)])
    return len(ids)


def main():
    C.MAP.mkdir(parents=True, exist_ok=True)
    while not C.DB.exists():
        time.sleep(30)
    con = connect()
    while True:
        final = (C.WORK / "INGEST_DONE").exists()
        V = docvecs()
        n_docs = 0 if V is None else len(V)
        tm = _load("text_model.pkl")
        due = [m for m in MILESTONES["text"] if n_docs >= m and (tm is None or tm["milestone"] < m)]
        if final and (tm is None or tm["milestone"] != FINAL):
            due = [FINAL]
        if due:
            fit_text(con, due[-1]); tm = _load("text_model.pkl")
        elif tm is not None:
            n = assign_text(con, tm)
            if n:
                print(f"text: placed {n} new docs", flush=True)
        X = image_matrix()
        n_img = 0 if X is None else len(X)
        im = _load("image_model.pkl")
        due = [m for m in MILESTONES["image"] if n_img >= m and (im is None or im["milestone"] < m)]
        if final and n_img and (im is None or im["milestone"] != FINAL):
            due = [FINAL]
        if due:
            fit_image(con, due[-1]); im = _load("image_model.pkl")
        elif im is not None:
            n = assign_image(con, im)
            if n:
                print(f"image: placed {n} new images", flush=True)
        if final and tm and tm["milestone"] == FINAL and (not n_img or (im and im["milestone"] == FINAL)):
            if not con.execute("SELECT 1 FROM docs WHERE vrow IS NOT NULL AND topic=-1 LIMIT 1").fetchone():
                print("topics final.", flush=True)
                (C.WORK / "TOPICS_DONE").write_text(time.ctime())
                return
        time.sleep(30)


if __name__ == "__main__":
    main()
