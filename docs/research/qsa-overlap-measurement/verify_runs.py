#!/usr/bin/env python3
"""Check that the primary (eager) prefill selections are byte-identical to the other runs.

Layers of the same prefill chunk were recorded by three serve runs per context; this script hashes the
`selected.bin` and `counts.bin` of every prefill chunk capture (>= 1000 columns) in each run and
groups captures by (pooled_ptr, min_position), so any divergence between runs is visible.

Usage: python3 verify_runs.py [--base out/qsa-index-capture]
"""
import argparse
import glob
import hashlib
import os
import re
import sys


def read_meta(path):
    meta = {}
    for line in open(path):
        line = line.strip()
        m = re.match(r"^([a-z_]+)\s+(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2)
    return meta


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prefill_hashes(run_dir):
    out = {}
    for dp, _, fns in os.walk(os.path.join(run_dir, "selections")):
        if "meta.txt" not in fns:
            continue
        meta = read_meta(os.path.join(dp, "meta.txt"))
        if int(meta.get("columns", "0")) < 1000:
            continue
        key = (meta["pooled_ptr"], meta["min_position"])
        out[key] = (dp, sha256(os.path.join(dp, "selected.bin")),
                    sha256(os.path.join(dp, "counts.bin")))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(here, "..", "..", "..", "out", "qsa-index-capture"))
    args = ap.parse_args()
    base = os.path.abspath(args.base)

    groups = [("16k", ["16k", "16k-eager", "scores-16k"]),
              ("140k", ["140k", "140k-eager", "scores-140k"])]
    for name, runs in groups:
        hashes = {}
        for run in runs:
            path = os.path.join(base, run)
            if not os.path.isdir(path):
                print(f"{name}: {run}: MISSING")
                continue
            hashes[run] = prefill_hashes(path)
        ref_run = runs[1] if len(runs) > 1 else runs[0]  # the eager run is the measurement's primary
        ref = hashes.get(ref_run, {})
        print(f"{name}: reference = {ref_run} ({len(ref)} prefill chunk captures)")
        for run in runs:
            other = hashes.get(run, {})
            same = sum(1 for k in ref if k in other and ref[k][1] == other[k][1]
                       and ref[k][2] == other[k][2])
            missing = len(set(ref) - set(other))
            extra = len(set(other) - set(ref))
            print(f"  {run:12s}: {len(other)} captures; identical selected+counts = {same}; "
                  f"missing-vs-ref = {missing}; extra-vs-ref = {extra}")


if __name__ == "__main__":
    main()
