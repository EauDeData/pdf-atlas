# PDF Atlas

A live explorer for large collections of born-digital PDFs: semantic, keyword and image search,
automatically discovered topics, and a zoomable map of the whole collection. It is built to
**grow while you use it**: indexing streams in the background, and the explorer is usable from the
first minute, with a progress bar showing how much of the collection is searchable.

It was built for a ~150k-PDF, ~1 TB, multilingual collection (≈9M pages) on a shared GPU server.
Nothing in it is domain-specific: point it at folders of PDFs.

```
   PDFs (read-only)
       │  PyMuPDF object tree: page text + image xrefs/bboxes  (no OCR, no layout model, no pixels saved)
       ▼
  ┌──────────┐   shards/   ┌──────────────┐  emb/   ┌──────────┐  docs.sqlite (WAL) + FTS5   ┌────────┐
  │ extract  │────────────▶│ embed_text   │────────▶│  ingest  │──────────────────────────▶ │ server │◀── browser
  │ CPU pool │             │ embed_images │         │          │  index/*.f16 + Annoy segs  │FastAPI │
  └──────────┘             │ (1 per GPU)  │         └──────────┘                            └────────┘
                           └──────────────┘               │                                     ▲
                                                          ▼                                     │
                                                    ┌──────────┐  map/ (topics.json, layouts)   │
                                                    │  topics  │────────────────────────────────┘
                                                    └──────────┘
```

## What you get

* **Passage search** (default): multilingual dense retrieval over ~220-word chunks, grouped by
  document, with the matching pages and passages. Query in any language.
* **Document search**: whole documents ranked by their mean passage vector ("papers about…").
* **Image search**: figures embedded in the PDFs, text→image with SigLIP-2, plus "similar figures".
* **Keyword search**: SQLite FTS5 / BM25 with phrases, `OR`, `NOT`, `prefix*`.
* **Topics**: k-means over document vectors, named with c-TF-IDF words; image clusters named
  zero-shot. Refitted as the collection grows.
* **Fig. 1, the map**: UMAP of every document (or a 1/32 sample of figures), coloured by topic or
  collection. Hover to preview, scroll to zoom, drag to pan, click to open. Search results light up
  on it, and newly indexed documents pulse in.
* **Viewer**: pages rendered on demand, figure boxes from the PDF overlaid (click → similar figures),
  page text with "similar passages", similar documents, link to the PDF and its source URL.
* **Browse**: the catalogue with collection / topic / year filters.
* **Live progress**: a bar in the explorer, a `/progress` page and a terminal dashboard.

Everything the browser loads is paginated server-side: search results are offset slices of a cached
ranked list (20 per page, 36 for images), and browsing uses keyset cursors (40 rows). The page never
holds the corpus; the only bulk payload is the map, at ~10 bytes per point.

## Quick start

```bash
pip install -r requirements.txt          # or reuse an env with torch + the packages listed there

export PDFATLAS_SOURCES=/path/to/pdfs     # a directory (each sub-directory = one collection)
                                         # or a sources JSON file (see below)
export PDFATLAS_WORK=/local/disk/atlas    # outputs; must be a LOCAL disk (SQLite WAL)
export PDFATLAS_TITLE="Medieval charters atlas"
export HF_HOME=/local/disk/hf_cache       # models are downloaded on first use

./run.sh start      # tmux session "pdfatlas"
./run.sh attach     # live progress (window 0); Ctrl-b n / p to cycle through the stages
# explorer: http://<host>:8800    progress: http://<host>:8800/progress
```

`./run.sh stop` stops everything. Every stage resumes from what is on disk, so stop/start is
always safe. `./run.sh restart-server` / `restart-extract` restart a single stage (e.g. after
changing `PORT` or `EXTRACT_WORKERS`).

### Sources file

```json
{"sources": [
  {"name": "theses", "path": "/data/pdfs/theses", "exclude": ["*_draft*"],
   "metadata": {"type": "jsonl", "path": "/data/pdfs/theses/metadata.jsonl", "key": "file",
                "fields": {"title": "title", "authors": "creator", "date": "issued", "url": "handle_url"}}},
  {"name": "arxiv", "path": "/data/pdfs/arxiv",
   "metadata": {"type": "sqlite", "path": "/data/pdfs/arxiv/manifest.db",
                "query": "SELECT filepath AS file, title, authors, published AS date, abstract, pdf_url AS url FROM papers"}},
  {"name": "misc", "path": "/data/pdfs/misc"}
]}
```

Metadata is optional (`jsonl`, `csv` or `sqlite`). Rows are matched to PDFs by the basename of the
key column. Recognised fields are `title, authors, date|year, abstract, url`, and `fields` maps them
to your column names. Without metadata, the title comes from the PDF's own metadata or its file
name, and the year from the title. Each document also gets a searchable metadata chunk (title +
abstract), so scanned PDFs without a text layer are still findable by title. See
[`examples/sources.example.json`](examples/sources.example.json).

## How it works

| stage | module | what it does |
|---|---|---|
| catalogue | `pdfatlas/catalog.py` | lists PDFs, attaches metadata → `catalog.parquet` (no PDF is opened) |
| extract | `pdfatlas/extract.py` | CPU pool. Per page: text (de-hyphenated, whitespace-normalised) split into ~220-word chunks; embedded images taken *from the object tree* (xref + bbox), skipping tiny images and full-page scan backgrounds. Documents are processed in a fixed random order, so every prefix of the index is a uniform sample of the collection. → `shards/{docs,text,img}_XXXXX.parquet` |
| embed text | `pdfatlas/embed_text.py` | `intfloat/multilingual-e5-large-instruct` (1024-d), one process per GPU, length-sorted batches. Shards are claimed with lock files, so any number of workers can join or leave. → `emb/text_*.npy` |
| embed images | `pdfatlas/embed_images.py` | SigLIP-2. Pixels are decoded from the xref in RAM by a CPU pool, resized, embedded and dropped. → `emb/img_*.npy` |
| ingest | `pdfatlas/ingest.py` | appends to SQLite (WAL, so readers never block), FTS5 and append-only fp16 vector files. Every 400k text / 200k image vectors are sealed into an immutable Annoy segment. Crash-safe: on restart the vector files are trimmed back to what the DB committed. |
| topics | `pdfatlas/topics.py` | refits at milestones (1.5k, 8k, 30k, 80k docs, then once at the end). Between refits, new documents get the nearest centroid and `umap.transform` coordinates, so the map grows instead of reshuffling. |
| progress | `pdfatlas/progress.py` | `progress.json` (served at `/api/progress`) + terminal dashboard, including our own CPU/GPU usage |
| server | `pdfatlas/server.py` | FastAPI. Vector search = Annoy over sealed segments + brute force over the unsealed tail, merged. Pages and figure crops are rendered on demand by MuPDF in worker processes (in-memory LRU, nothing written). |
| UI | `pdfatlas/static/` | vanilla JS, no build step. |

### API

| endpoint | |
|---|---|
| `GET /api/search?q=&mode=semantic\|docs\|image\|keyword&offset=&limit=&collection=&topic=&year_from=&year_to=` | paginated; returns `items`, `next` (offset or `null`), `total`, `searched` (how many vectors were in the index at query time) |
| `…&like_chunk=ID` / `like_image=ID` / `like_doc=ID` | "more like this" |
| `GET /api/docs?cursor=&limit=&…filters` | keyset-paginated catalogue |
| `GET /api/doc/{id}`, `/api/doc/{id}/text?page=` | document + figures, page text |
| `GET /api/render/{doc}/{page}.jpg?w=`, `/api/image/{id}.jpg?w=`, `/api/pdf/{doc}` | rendered page, figure crop, original PDF |
| `GET /api/map/{docs\|images}.bin` | `u32 n · i32 ids[n] · u16 xy[n,2] · u8 topic+1[n] · u8 collection[n]` |
| `GET /api/stats`, `/api/progress`, `/api/topic/{docs\|images}/{k}` | |

## Being a good neighbour on a shared machine

Library thread pools are the real danger: numpy, torch, numba and scikit-learn each default to one
thread per core, *per process*. A handful of processes with pools can overload a 128-core box.
PDF Atlas therefore:

* caps OMP/MKL/OpenBLAS/numba/tokenizer threads in `pdfatlas/__init__.py`, before any heavy import
  (`PDFATLAS_THREADS`, default 2), and calls `torch.set_num_threads`;
* pins every process with `taskset`: by default the last ~20% of cores (`CORES`) for everything,
  and the last ~31% (`EXTRACT_CORES`) for the CPU-bound extraction;
* runs everything at `nice 19` and `ionice -c3`;
* only takes GPUs that are idle at launch (`TEXT_GPUS` / `IMAGE_GPU` to choose), caps each process's
  CUDA allocator, and backs off (halves the batch and waits) on out-of-memory errors.

On the reference run (128 cores, 4 GPUs) the whole pipeline used about 25 cores and 2.5 GB per GPU.

## Configuration

| variable | default | |
|---|---|---|
| `PDFATLAS_SOURCES` | — | directory or sources JSON |
| `PDFATLAS_WORK` | `./atlas-data` | outputs (local disk) |
| `PDFATLAS_TITLE` | `PDF Atlas` | shown in the UI |
| `PDFATLAS_TEXT_MODEL`, `PDFATLAS_TEXT_DIM`, `PDFATLAS_QUERY_INSTRUCT` | e5-large-instruct, 1024 | any sentence-transformers model |
| `PDFATLAS_IMAGE_MODEL` | `google/siglip2-base-patch16-256` | |
| `PDFATLAS_IMAGE_CONCEPTS` | built-in, domain-neutral | text file, one concept per line, used to name image clusters |
| `PDFATLAS_THREADS` | 2 | per-process library thread cap |
| `PY`, `PORT`, `CORES`, `EXTRACT_CORES`, `EXTRACT_WORKERS`, `IMAGE_WORKERS`, `TEXT_GPUS`, `IMAGE_GPU`, `MAX_TEXT_GPUS`, `SESSION` | see `run.sh` | launcher |

Extraction limits (pages per document, chunk size, figure size thresholds) are in
`pdfatlas/config.py`.

## Throughput (reference run)

On an A40, text embedding runs at ~180 chunks/s; an RTX PRO 6000 Blackwell is ~2.5× that. Text
embedding is compute-bound, roughly *chunks × tokens × model size*. At ~13M chunks the full text
pass takes about 4 h on 3 GPUs. Swap in `intfloat/multilingual-e5-base` for ~3× faster indexing, or
`Qwen/Qwen3-Embedding-0.6B` for slightly better retrieval at ~2.7× the cost. Extraction runs at
~10 PDFs/s with 28 PyMuPDF workers. Image embedding is limited by pixel decoding (~200 figures/s
with 6 decoders).

## Limitations

* No OCR, by design: pages without a text layer contribute only their document's metadata chunk.
* Figures are whatever the PDF stores as images. Vector graphics (many charts) are not extracted.
* The image map shows a 1/32 sample of figures; all of them are searchable.
* Annoy segments are approximate; the unsealed tail (the newest vectors) is searched exactly.
