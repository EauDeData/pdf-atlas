"""Stage 3: embed the images referenced in img_XXXXX.parquet with SigLIP2.
Pixels are decoded from the PDF xref in RAM (CPU worker pool), resized, embedded and discarded.
Output: WORK/emb/img_XXXXX.npy (float16, L2-normalised) aligned with shard rows (zeros + ok=0 if undecodable).
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as C

RES = 256
DIM = 768
_M = None


def load(device="cuda"):
    global _M
    if _M is None:
        import torch
        from transformers import AutoModel, AutoProcessor
        m = AutoModel.from_pretrained(C.IMAGE_MODEL, torch_dtype=torch.float16).to(device).eval()
        p = AutoProcessor.from_pretrained(C.IMAGE_MODEL)
        _M = (m, p, device)
    return _M


def encode_text(queries, device="cuda"):
    import torch
    m, p, dev = load(device)
    inp = p(text=[q.lower() for q in queries], padding="max_length", max_length=64, truncation=True,
            return_tensors="pt").to(dev)
    with torch.no_grad():
        e = m.get_text_features(**inp)
        if not torch.is_tensor(e):
            e = e.pooler_output
    e = torch.nn.functional.normalize(e.float(), dim=-1)
    return e.cpu().numpy().astype(np.float16)


def encode_pixels(arr, device="cuda"):
    """arr: uint8 [N,RES,RES,3]"""
    import torch
    m, _, dev = load(device)
    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).half().div_(127.5).sub_(1.0)
    with torch.no_grad():
        e = m.get_image_features(pixel_values=x)
        if not torch.is_tensor(e):
            e = e.pooler_output
    e = torch.nn.functional.normalize(e.float(), dim=-1)
    return e.cpu().numpy().astype(np.float16)


def decode_group(args):
    """Open one PDF, decode its listed xrefs -> RES x RES uint8. Returns (row_idx list, array, ok mask)."""
    import pymupdf
    from PIL import Image
    pymupdf.TOOLS.mupdf_display_errors(False)
    path, idx, xrefs = args
    out = np.zeros((len(xrefs), RES, RES, 3), np.uint8)
    ok = np.zeros(len(xrefs), bool)
    cache = {}
    try:
        with pymupdf.open(path) as d:
            for j, x in enumerate(xrefs):
                try:
                    if x in cache:
                        out[j] = cache[x]; ok[j] = True; continue
                    pix = pymupdf.Pixmap(d, int(x))
                    if pix.n - pix.alpha >= 4 or pix.colorspace is None or pix.colorspace.n != 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.alpha:
                        pix = pymupdf.Pixmap(pix, 0)
                    # shrink huge images cheaply before PIL
                    while pix.width > 4 * RES and pix.height > 4 * RES:
                        pix.shrink(1)
                    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    a = np.asarray(im.resize((RES, RES), Image.BICUBIC))
                    cache[x] = a; out[j] = a; ok[j] = True
                except Exception:
                    pass
    except Exception:
        pass
    return idx, out, ok


def main():
    import torch
    from .gpu import setup
    from .work import atomic_save_npy, stream
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    ap.add_argument("--workers", type=int, default=6, help="CPU processes decoding pixels")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap-gb", type=float, default=8)
    a = ap.parse_args()
    from . import limit_torch
    limit_torch()
    dev = setup(a.gpu, a.cap_gb)
    load(dev)
    dim = encode_text(["x"], dev).shape[1]
    paths = pd.read_parquet(C.CATALOG, columns=["doc_id", "path"]).set_index("doc_id")["path"]
    t0, n = time.time(), 0
    with Pool(a.workers, maxtasksperchild=200) as pool:
        for bid, shard, out in stream("img", C.EMB):
            df = pd.read_parquet(shard).reset_index(drop=True)
            emb = np.zeros((len(df), dim), np.float16)
            ok_all = np.zeros(len(df), bool)
            groups = []  # <=64 images per task keeps pickled payloads small (64*196KB)
            for d, g in df.groupby("doc_id"):
                ix, xr = g.index.to_numpy(), g["xref"].to_numpy()
                groups += [(paths[d], ix[i:i + 64], xr[i:i + 64]) for i in range(0, len(ix), 64)]
            buf_i, buf_a = [], []

            def flush():
                if buf_i:
                    ii, aa = np.concatenate(buf_i), np.concatenate(buf_a)
                    for j in range(0, len(ii), a.batch):
                        while True:
                            try:
                                emb[ii[j:j + a.batch]] = encode_pixels(aa[j:j + a.batch], dev); break
                            except torch.OutOfMemoryError:
                                torch.cuda.empty_cache(); print("OOM, backing off 30s", flush=True); time.sleep(30)
                    buf_i.clear(); buf_a.clear()

            for idx, arr, ok in pool.imap_unordered(decode_group, groups):
                ok_all[idx[ok]] = True
                if ok.any():
                    buf_i.append(idx[ok]); buf_a.append(arr[ok])
                if sum(len(b) for b in buf_i) >= a.batch:
                    flush()
            flush()
            np.save(out.with_name(out.stem + ".ok.npy"), ok_all)
            atomic_save_npy(out, emb)
            n += len(df)
            print(f"[gpu{a.gpu}] shard {bid} imgs={n} {n/(time.time()-t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
