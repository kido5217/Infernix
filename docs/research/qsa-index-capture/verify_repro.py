#!/usr/bin/env python3
"""Check that every content file of capture root A reappears byte-identically under root B.

Used to show that Run S (the score pass) reproduces Run A (the pre-extension capture) for every
plane, query and selection, so its scores belong to the same steps and columns.

Content files are the raw binaries (pooled/index_q/selected/counts/positions/block_table/
table_rows); meta.txt is skipped (it carries a timestamp and step/capture numbers).

usage: verify_repro.py <rootA> <rootB>
"""
import hashlib
import os
import sys

CONTENT = {"pooled.bin", "index_q.bin", "selected.bin", "counts.bin", "positions.bin",
           "block_table.bin", "table_rows.bin"}


def hashes(root):
    found = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name not in CONTENT:
                continue
            path = os.path.join(dirpath, name)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            found.setdefault(h, []).append(os.path.relpath(path, root))
    return found


def main(a, b):
    ha, hb = hashes(a), hashes(b)
    total = sum(len(v) for v in ha.values())
    missing = [p for h, paths in ha.items() if h not in hb for p in paths]
    print(f"{a}: {total} content files, {len(ha)} distinct hashes")
    print(f"{b}: {sum(len(v) for v in hb.values())} content files, {len(hb)} distinct hashes")
    print(f"reproduced in B: {total - len(missing)} / {total}")
    print(f"missing: {len(missing)}")
    for p in missing[:20]:
        print("  MISSING", p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
