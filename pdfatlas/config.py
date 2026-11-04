"""Configuration: where the PDFs are, where outputs go, which models to use.

Everything is driven by environment variables so the same code serves any PDF collection:

    PDFATLAS_SOURCES  path to a sources JSON file (see examples/sources.example.json), or
                      a plain directory -> every sub-directory becomes one "collection"
    PDFATLAS_WORK     output directory (SQLite, vectors, indexes, logs). Use a LOCAL disk:
                      SQLite in WAL mode is not reliable over NFS.
    PDFATLAS_TITLE    title shown in the explorer (default "PDF Atlas")
    HF_HOME           Hugging Face cache (models are downloaded on first use)

Source PDFs are only ever read. Nothing is written next to them.
"""
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("PDFATLAS_WORK", "./atlas-data")).expanduser().resolve()
TITLE = os.environ.get("PDFATLAS_TITLE", "PDF Atlas")
SOURCES_SPEC = os.environ.get("PDFATLAS_SOURCES", "")

CATALOG = WORK / "catalog.parquet"   # one row per PDF (+ harvested metadata)
SHARDS = WORK / "shards"             # extraction output: text chunks + image refs (no pixels)
EMB = WORK / "emb"                   # per-shard embeddings (fp16 .npy)
DB = WORK / "docs.sqlite"            # metadata + chunk text + FTS5 (WAL, read live by the server)
IDX = WORK / "index"                 # append-only vector files + sealed Annoy segments
MAP = WORK / "map"                   # topic models, 2D layouts, topics.json

# ------------------------------------------------------------------------------- models
# multilingual-e5-large-instruct: strong multilingual retrieval (~1 MTEB point below
# Qwen3-Embedding-0.6B) at ~2.7x its throughput. Any sentence-transformers model works;
# set TEXT_DIM / QUERY_INSTRUCT accordingly (empty instruct for symmetric models).
TEXT_MODEL = os.environ.get("PDFATLAS_TEXT_MODEL", "intfloat/multilingual-e5-large-instruct")
TEXT_DIM = int(os.environ.get("PDFATLAS_TEXT_DIM", 1024))
TEXT_MAX_TOKENS = 512
QUERY_INSTRUCT = os.environ.get(
    "PDFATLAS_QUERY_INSTRUCT",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
# SigLIP-2 (a CLIP-style model): figures <-> text queries in many languages.
IMAGE_MODEL = os.environ.get("PDFATLAS_IMAGE_MODEL", "google/siglip2-base-patch16-256")

# ------------------------------------------------------------------------------- extraction limits
MAX_PAGES = 600          # pages per document beyond this are skipped
CHUNK_WORDS = 220        # target words per text chunk (a page yields ~1-3 chunks)
MIN_CHUNK_WORDS = 12     # shorter page text is ignored (headers, page numbers…)
MAX_FILE_MB = 400        # larger files are skipped
IMG_MIN_PX = 120         # min intrinsic side (px) for an embedded image to count as a figure
IMG_MIN_FRAC = 0.02      # min area as a fraction of the page
IMG_MAX_FRAC = 0.80      # larger = a scanned-page background, not a figure
MAX_IMGS_PER_PAGE = 8


# ------------------------------------------------------------------------------- sources
def load_sources():
    """-> list of dicts {name, path, glob, exclude, metadata}. See examples/sources.example.json."""
    spec = SOURCES_SPEC
    if not spec:
        raise SystemExit("Set PDFATLAS_SOURCES to a sources JSON file or to a directory of PDFs.")
    p = Path(spec).expanduser()
    if p.is_dir():   # every sub-directory is a collection; loose PDFs at the top go to "root"
        subs = sorted(d for d in p.iterdir() if d.is_dir())
        out = [dict(name=d.name, path=str(d)) for d in subs]
        if any(f.suffix.lower() == ".pdf" for f in p.iterdir() if f.is_file()):
            out.append(dict(name="root", path=str(p), recursive=False))
        return [_defaults(s) for s in out]
    return [_defaults(s) for s in json.loads(p.read_text())["sources"]]


def _defaults(s):
    s = dict(s)
    s.setdefault("recursive", True)
    s.setdefault("exclude", [])
    s.setdefault("metadata", None)
    return s


def collections():
    """Collection names, read from the catalogue (so the server never needs the sources file)."""
    import pandas as pd
    if CATALOG.exists():
        return sorted(pd.read_parquet(CATALOG, columns=["collection"]).collection.unique().tolist())
    return []
