# Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/cuda_malloc.py


def get_gpu_names():
    from subprocess import check_output

    gpu_names = set()
    out = check_output(["nvidia-smi", "-L"])
    for line in out.split(b"\n"):
        if len(line) > 0:
            gpu_names.add(line.decode("utf-8").split(" (UUID")[0])
    return gpu_names


def cuda_malloc_supported():
    blacklist = {
        "GeForce GTX TITAN X",
        "GeForce GTX 980",
        "GeForce GTX 970",
        "GeForce GTX 960",
        "GeForce GTX 950",
        "GeForce 945M",
        "GeForce 940M",
        "GeForce 930M",
        "GeForce 920M",
        "GeForce 910M",
        "GeForce GTX 750",
        "GeForce GTX 745",
        "Quadro K620",
        "Quadro K1200",
        "Quadro K2200",
        "Quadro M500",
        "Quadro M520",
        "Quadro M600",
        "Quadro M620",
        "Quadro M1000",
        "Quadro M1200",
        "Quadro M2000",
        "Quadro M2200",
        "Quadro M3000",
        "Quadro M4000",
        "Quadro M5000",
        "Quadro M5500",
        "Quadro M6000",
        "GeForce MX110",
        "GeForce MX130",
        "GeForce 830M",
        "GeForce 840M",
        "GeForce GTX 850M",
        "GeForce GTX 860M",
        "GeForce GTX 1650",
        "GeForce GTX 1630",
        "Tesla M4",
        "Tesla M6",
        "Tesla M10",
        "Tesla M40",
        "Tesla M60",
    }

    try:
        names = get_gpu_names()
    except Exception:
        return False

    for x in names:
        if "NVIDIA" in x:
            for b in blacklist:
                if b in x:
                    return False
            return True

    return False


def try_cuda_malloc():
    import os

    if not cuda_malloc_supported():
        print("Failed to use cudaMallocAsync backend...")
        return

    env_var = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
    if env_var is None:
        env_var = "backend:cudaMallocAsync"
    else:
        env_var += ",backend:cudaMallocAsync"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_var
    print("Using cudaMallocAsync backend")
