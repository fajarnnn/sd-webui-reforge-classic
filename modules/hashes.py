import hashlib
import os.path

from modules import shared
import modules.cache

dump_cache = modules.cache.dump_cache
cache = modules.cache.cache


def calculate_sha256(filename: str) -> str:
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    if title not in hashes:
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    if cached_sha256 is None:
        return None

    try:
        ondisk_mtime = os.path.getmtime(filename)
    except FileNotFoundError:
        return None

    cached_mtime = hashes[title].get("mtime", -1)
    if ondisk_mtime != cached_mtime:
        return None

    return cached_sha256


def sha256(filename, title, use_addnet_hash=False):
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    print(f"Calculating sha256 for {filename}: ", end="")
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value


def addnet_hash_safetensors(b: bytes) -> str:
    """
    kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py
    """
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def partial_hash_from_cache(filename):
    hashes = cache("partial-hashes")
    if filename in hashes:
        return hashes[filename]

    try:
        with open(filename, "rb") as file:
            hash_sha256 = hashlib.sha256()
            file.seek(0x100000)
            hash_sha256.update(file.read(0x10000))
            partial_hash = hash_sha256.hexdigest()[0:8]
            hashes[filename] = partial_hash
            return partial_hash
    except Exception:
        return "NOFILE"
