#!/usr/bin/env python3
"""Ask Linux to evict file-backed page cache for benchmark inputs.

This is useful when you do not have sudo access for /proc/sys/vm/drop_caches.
It is advisory, so the kernel may keep some pages, but it avoids rewriting or
deleting the files and usually helps prevent hot-cache benchmark results.
"""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path


def iter_files(path: Path):
    if path.is_file():
        yield path
        return

    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and not child.is_symlink():
                yield child


def evict_file(path: Path) -> tuple[bool, str | None]:
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True, None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return False, "missing"
        return False, exc.strerror or repr(exc)
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evict file-backed page cache for one or more files/directories "
            "using posix_fadvise(..., POSIX_FADV_DONTNEED)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to evict from the page cache.",
    )
    args = parser.parse_args()

    total = 0
    evicted = 0
    failed = 0
    bytes_seen = 0

    for raw_path in args.paths:
        root = Path(raw_path).expanduser()
        if not root.exists():
            print(f"missing root: {root}")
            failed += 1
            continue

        for path in iter_files(root):
            total += 1
            try:
                bytes_seen += path.stat().st_size
            except OSError:
                pass

            ok, error = evict_file(path)
            if ok:
                evicted += 1
            else:
                failed += 1
                print(f"failed: {path}: {error}")

    gib = bytes_seen / (1024**3)
    print(
        f"eviction requested for {evicted}/{total} files "
        f"({gib:.2f} GiB scanned, {failed} failures)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
