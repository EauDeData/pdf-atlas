"""Thread caps are set here, before any submodule imports numpy/torch/numba/sklearn.
Shared 128-core box: every library defaults to one thread per core, and a handful of processes
x pools x 128 threads is what took the server down. PDFATLAS_THREADS raises the cap per process."""
import os as _os

_n = _os.environ.get("PDFATLAS_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    _os.environ[_v] = _n
_os.environ["TOKENIZERS_PARALLELISM"] = "false"


def limit_torch():
    import torch
    torch.set_num_threads(int(_n))
    torch.set_num_interop_threads(1)
