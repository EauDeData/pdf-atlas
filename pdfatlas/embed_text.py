"""Stage 2 (streaming): embed text chunks as extraction shards appear. Start one per GPU:
    python -m pdfatlas.embed_text --gpu 4
Output: WORK/emb/text_XXXXX.npy (float16, L2-normalised, TEXT_DIM), aligned with text_XXXXX.parquet rows.
"""
import argparse
import time

import numpy as np
import pandas as pd

from . import config as C

_MODEL = None


def model(device="cuda"):
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(C.TEXT_MODEL, device=device, model_kwargs={"dtype": torch.float16})
        _MODEL.max_seq_length = C.TEXT_MAX_TOKENS
    return _MODEL


def encode(texts, query=False, batch_size=64, device="cuda"):
    m = model(device)
    kw = {"prompt": C.QUERY_INSTRUCT} if query else {}  # e5-instruct: queries get the instruction, passages nothing
    e = m.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False, **kw)
    return e.astype(np.float16)


def encode_polite(texts, batch_size, device):
    """Shared GPUs: on OOM free our cache, halve the batch, wait, retry."""
    import torch
    while True:
        try:
            return encode(texts, batch_size=batch_size, device=device), batch_size
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(4, batch_size // 2)
            print(f"OOM -> batch {batch_size}, backing off 30s", flush=True)
            time.sleep(30)


def main():
    from .gpu import setup
    from .work import atomic_save_npy, stream
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cap-gb", type=float, default=10)
    a = ap.parse_args()
    from . import limit_torch
    limit_torch()
    dev = setup(a.gpu, a.cap_gb)
    model(dev)
    t0, n, batch = time.time(), 0, a.batch
    for bid, shard, out in stream("text", C.EMB):
        texts = pd.read_parquet(shard, columns=["text"])["text"].tolist()
        if texts:
            order = np.argsort([-len(t) for t in texts])  # length-sorted batches = less padding
            e, batch = encode_polite([texts[i] for i in order], batch, dev)
            emb = np.empty_like(e); emb[order] = e
        else:
            emb = np.zeros((0, C.TEXT_DIM), np.float16)
        atomic_save_npy(out, emb)
        n += len(texts)
        print(f"[gpu{a.gpu}] shard {bid} chunks={n} {n/(time.time()-t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
