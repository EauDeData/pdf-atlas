#!/usr/bin/env bash
# PDF Atlas: run the whole streaming pipeline + explorer in a tmux session, inside a hard CPU budget.
#
#   PDFATLAS_SOURCES=sources.json PDFATLAS_WORK=/local/disk/atlas ./run.sh start
#   ./run.sh attach            # window 0 = live progress dashboard
#   ./run.sh serve             # index already built: explorer + progress + new-PDF watcher only (no GPU until new PDFs)
#   ./run.sh restart-server    # e.g. after changing PORT
#   ./run.sh restart-extract   # e.g. after changing EXTRACT_CORES / EXTRACT_WORKERS
#   ./run.sh restart-watch     # (re)start the new-PDF watcher, e.g. TEXT_GPUS="2 3" IMAGE_GPU=1 ./run.sh restart-watch
#   ./run.sh stop
#
# Every process is pinned with taskset, runs at nice 19 / ionice idle, and has its library thread
# pools capped (pdfatlas/__init__.py). Whatever numpy/torch/numba decide, they cannot leave CORES.
# All stages resume from what is on disk, so stop/start is always safe.
set -euo pipefail
cd "$(dirname "$0")"

: "${PDFATLAS_WORK:?set PDFATLAS_WORK to an output directory on a LOCAL disk}"
export PDFATLAS_WORK PDFATLAS_SOURCES=${PDFATLAS_SOURCES:-} PDFATLAS_TITLE=${PDFATLAS_TITLE:-PDF Atlas}
[ -n "${HF_HOME:-}" ] && export HF_HOME
PY=${PY:-python}
NCPU=$(nproc)
CORES=${CORES:-$((NCPU - NCPU / 5))-$((NCPU - 1))}           # last ~20% of cores: GPU feeders, ingest, topics, server
EXTRACT_CORES=${EXTRACT_CORES:-$((NCPU - NCPU * 5 / 16))-$((NCPU - 1))}  # last ~31%: extraction (CPU-bound)
EXTRACT_WORKERS=${EXTRACT_WORKERS:-$((NCPU * 7 / 32))}
IMAGE_WORKERS=${IMAGE_WORKERS:-6}
TEXT_GPUS=${TEXT_GPUS:-}          # e.g. "2 3"; empty = pick idle GPUs at launch
IMAGE_GPU=${IMAGE_GPU:-}
MAX_TEXT_GPUS=${MAX_TEXT_GPUS:-3}
PORT=${PORT:-8800}
WATCH=${WATCH:-1}                 # 1 = watch the sources and index PDFs added later (pdfatlas/watch.py)
WATCH_INTERVAL=${WATCH_INTERVAL:-120}
WATCH_TEXT_GPUS=$TEXT_GPUS WATCH_IMAGE_GPU=$IMAGE_GPU   # as declared; empty = idle GPUs at each run
LOG=$PDFATLAS_WORK/logs; mkdir -p "$LOG"
S=${SESSION:-pdfatlas}

wrap() {  # wrap <threads> <cmd...>   (CPUSET overrides CORES)
  local th=$1; shift
  printf 'cd %q && export PDFATLAS_WORK=%q PDFATLAS_SOURCES=%q PDFATLAS_TITLE=%q PDFATLAS_THREADS=%s %s && exec taskset -c %s nice -n 19 ionice -c3 %s' \
    "$PWD" "$PDFATLAS_WORK" "$PDFATLAS_SOURCES" "$PDFATLAS_TITLE" "$th" "${HF_HOME:+HF_HOME=$HF_HOME}" "${CPUSET:-$CORES}" "$*"
}
server_cmd() { echo "$(wrap 4 env CUDA_VISIBLE_DEVICES= PDFATLAS_RENDER_WORKERS=3 $PY -m uvicorn pdfatlas.server:app --host 0.0.0.0 --port $PORT) 2>&1 | tee -a $LOG/server.log"; }
topics_cmd() { echo "$(wrap 4 $PY -m pdfatlas.topics) 2>&1 | grep --line-buffered -v -i warn | tee -a $LOG/topics.log"; }
watch_cmd() {
  local g=""
  [ -n "$WATCH_TEXT_GPUS" ] && g="--text-gpus $WATCH_TEXT_GPUS"
  [ -n "$WATCH_IMAGE_GPU" ] && g="$g --image-gpu $WATCH_IMAGE_GPU"
  echo "$(wrap 1 $PY -m pdfatlas.watch $g --max-text-gpus $MAX_TEXT_GPUS --interval $WATCH_INTERVAL --extract-workers $EXTRACT_WORKERS --extract-cores $EXTRACT_CORES --image-workers $IMAGE_WORKERS) 2>&1 | tee -a $LOG/watch.log"
}
extract_cmd() { echo "$(CPUSET=$EXTRACT_CORES wrap 1 $PY -m pdfatlas.extract --workers $EXTRACT_WORKERS) 2>&1 | tee -a $LOG/extract.log"; }

case "${1:-start}" in
  stop) tmux kill-session -t "$S" 2>/dev/null && echo stopped || echo "not running" ;;
  attach) exec tmux attach -t "$S" ;;
  restart-server)
    tmux kill-window -t "$S:server" 2>/dev/null || true
    tmux new-window -d -t "$S" -n server "$(server_cmd)"; echo "server on :$PORT" ;;
  restart-topics)
    tmux kill-window -t "$S:topics" 2>/dev/null || true
    tmux new-window -d -t "$S" -n topics "$(topics_cmd)"; echo "topics restarted" ;;
  serve)
    tmux has-session -t "$S" 2>/dev/null && { echo "already running: ./run.sh attach"; exit 1; }
    [ -f "$PDFATLAS_WORK/catalog.parquet" ] || { echo "no index in $PDFATLAS_WORK: use ./run.sh start"; exit 1; }
    tmux new-session -d -s "$S" -n progress "$(wrap 1 $PY -m pdfatlas.progress --watch)"
    tmux new-window -t "$S" -n server "$(server_cmd)"
    [ "$WATCH" = 1 ] && tmux new-window -d -t "$S" -n watch "$(watch_cmd)"
    tmux select-window -t "$S:progress"
    echo "serving · http://$(hostname):$PORT · watcher GPUs: ${WATCH_TEXT_GPUS:-idle} / ${WATCH_IMAGE_GPU:-idle}"
    ;;
  restart-watch)
    tmux kill-window -t "$S:watch" 2>/dev/null || true
    tmux new-window -d -t "$S" -n watch "$(watch_cmd)"; echo "watching ${PDFATLAS_SOURCES} every ${WATCH_INTERVAL}s (GPUs: ${WATCH_TEXT_GPUS:-idle} / ${WATCH_IMAGE_GPU:-idle})" ;;
  restart-extract)
    tmux kill-window -t "$S:extract" 2>/dev/null || true
    tmux new-window -d -t "$S" -n extract "$(extract_cmd)"; echo "extract on cores $EXTRACT_CORES, $EXTRACT_WORKERS workers" ;;
  start)
    tmux has-session -t "$S" 2>/dev/null && { echo "already running: ./run.sh attach"; exit 1; }
    [ -f "$PDFATLAS_WORK/catalog.parquet" ] || $PY -m pdfatlas.catalog
    if [ -z "$TEXT_GPUS" ]; then
      read -r -a G <<< "$($PY -m pdfatlas.gpu)"
      (( ${#G[@]} >= 2 )) || { echo "need >=2 idle GPUs (found: ${G[*]:-none}); set TEXT_GPUS / IMAGE_GPU"; exit 1; }
      IMAGE_GPU=${IMAGE_GPU:-${G[0]}}; TEXT_GPUS="${G[*]:1:$MAX_TEXT_GPUS}"
    fi
    echo "text GPUs: $TEXT_GPUS | image GPU: $IMAGE_GPU | cores $CORES (extract $EXTRACT_CORES x$EXTRACT_WORKERS)"
    tmux new-session -d -s "$S" -n progress "$(wrap 1 $PY -m pdfatlas.progress --watch)"
    tmux new-window -t "$S" -n extract "$(extract_cmd)"
    for g in $TEXT_GPUS; do
      tmux new-window -t "$S" -n text$g "$(wrap 2 $PY -m pdfatlas.embed_text --gpu $g) 2>&1 | grep --line-buffered -v -i warn | tee -a $LOG/text$g.log"
    done
    tmux new-window -t "$S" -n img$IMAGE_GPU "$(wrap 1 $PY -m pdfatlas.embed_images --gpu $IMAGE_GPU --workers $IMAGE_WORKERS) 2>&1 | grep --line-buffered -v -i warn | tee -a $LOG/img.log"
    tmux new-window -t "$S" -n ingest "$(wrap 2 $PY -m pdfatlas.ingest) 2>&1 | tee -a $LOG/ingest.log"
    tmux new-window -t "$S" -n topics "$(topics_cmd)"
    tmux new-window -t "$S" -n server "$(server_cmd)"
    [ "$WATCH" = 1 ] && tmux new-window -d -t "$S" -n watch "$(watch_cmd)"
    tmux select-window -t "$S:progress"
    echo "running · ./run.sh attach · http://$(hostname):$PORT · http://$(hostname):$PORT/progress"
    ;;
  *) echo "usage: $0 start|serve|stop|attach|restart-server|restart-topics|restart-extract|restart-watch"; exit 2 ;;
esac
