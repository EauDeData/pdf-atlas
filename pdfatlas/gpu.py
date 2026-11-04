"""Polite GPU use on a shared box: pick the emptiest GPUs, cap our memory, back off on OOM."""
import os
import subprocess


def free_gpus(min_free_mb=20000, max_util=10):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    g = []
    for line in out.strip().splitlines():
        i, used, tot, util = (int(x) for x in line.split(","))
        if tot - used >= min_free_mb and util <= max_util:
            g.append((used, tot, i))
    # prefer the smaller cards: leave the 96GB ones to people who need them
    return [i for used, tot, i in sorted(g, key=lambda x: (x[1] > 60000, x[0]))]


def setup(gpu, cap_gb=8.0):
    """gpu = physical index. Restricts visibility so we never touch other GPUs; caps our allocator."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 2**30 / tot), 0)
    return "cuda:0"


if __name__ == "__main__":
    print(" ".join(map(str, free_gpus())))
