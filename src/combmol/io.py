"""Atomic, provenance-bound receipts. Secrets are never configuration fields."""
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def frozen(path, value):
    destination = Path(path)
    if destination.exists():
        if canonical(read(destination)) != canonical(value):
            raise ValueError(f"Frozen input differs: {destination}")
    else:
        write(destination, value)
    return value


def event(root, stage, **details):
    write(Path(root) / "STATUS.json", dict(stage=stage, updated_at=time.time(), **details))


@contextlib.contextmanager
def exclusive(path, blocking=False):
    import fcntl
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity():
    identity = dict(pid=os.getpid(), time=time.time())
    if Path("/proc/self/stat").exists():
        identity["start_ticks"] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return identity
