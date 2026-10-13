#!/usr/bin/env python3
"""Recompute QSA block scores from a capture and check the recorded selection.

Usage: python3 verify_capture.py CAPDIR [n_columns]

Reads step_*/cap_*/{meta.txt,pooled.bin,index_q.bin,selected.bin,counts.bin} and, for each column,
recomputes score_b = 1/sqrt(Di) * sum_h relu(<q_h, pooled_key_b>) exactly as the engine defines it,
then compares the top-(budget/ratio) block set against selected.bin. FP32 accumulation order differs
from the kernel's butterfly, so near-ties are expected to differ; the exact and order-robust
agreement are both reported.
"""
import numpy as np
import os
import re
import struct
import sys

BF16 = np.dtype(np.uint16)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def read_meta(path):
    meta = {}
    for line in open(path):
        line = line.strip()
        m = re.match(r"^([a-z_]+)\s+(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2)
    if "geometry" in meta:
        for key, value in re.findall(r"([a-z_]+)=([^\s]+)", meta["geometry"]):
            meta[key] = value
    return meta


def check(capdir, n_columns=8):
    meta = read_meta(os.path.join(capdir, "meta.txt"))
    columns = int(meta["columns"])
    top = int(meta["top_blocks"])
    ratio = int(meta["ratio"])
    heads = int(meta["index_heads"])
    dim = int(meta["index_head_dim"])
    slots = int(meta["plane_slots"])
    page_bytes = int(meta["plane_page_bytes"])
    pages_used = int(meta["plane_pages_used"])
    max_pos = int(meta["max_position"])
    positions = np.fromfile(os.path.join(capdir, "positions.bin"), dtype=np.int32)
    pooled = np.fromfile(os.path.join(capdir, "pooled.bin"), dtype=np.uint8)
    query = np.fromfile(os.path.join(capdir, "index_q.bin"), dtype=np.uint16)
    selected = np.fromfile(os.path.join(capdir, "selected.bin"), dtype=np.int32)
    counts = np.fromfile(os.path.join(capdir, "counts.bin"), dtype=np.int32)

    assert pooled.size == pages_used * page_bytes, (pooled.size, pages_used * page_bytes)
    assert query.size == dim * heads * columns
    assert selected.size == top * columns
    assert counts.size == columns

    # token t -> the 32-element slice its block slot holds, at page t>>6, row t&63.
    def token_key(t):
        page = t >> 6
        row = t & 63
        off = page * page_bytes + slots * row * 2
        return pooled[off:off + slots * 2].view(BF16)

    checked = 0
    exact = 0
    robust = 0
    worst = 0
    for col in range(min(n_columns, columns)):
        pos = int(positions[col])
        blocks = (pos + 1) // ratio
        if blocks <= top:
            continue
        n = min(blocks, pages_used * 64 // ratio)
        keys = np.empty((n, dim), dtype=np.uint16)
        for b in range(n):
            keys[b, 0:slots] = token_key(ratio * b + 0)
            keys[b, slots:2 * slots] = token_key(ratio * b + 1)
            keys[b, 2 * slots:3 * slots] = token_key(ratio * b + 2)
            keys[b, 3 * slots:4 * slots] = token_key(ratio * b + 3)
        k = bf16_to_f32(keys)  # [n, dim]
        # query element (d, h, col) at d + dim*(h + heads*col)
        q = np.empty((heads, dim), dtype=np.float32)
        for h in range(heads):
            base = dim * (h + heads * col)
            q[h] = bf16_to_f32(query[base:base + dim])
        dots = k @ q.T                      # [n, heads]
        score = np.maximum(dots, 0.0).sum(axis=1) / np.sqrt(dim)

        order = np.lexsort((np.arange(n), -score))[:top]
        engine = selected[col * top:(col + 1) * top]
        if int(counts[col]) != top:
            print(f"  col {col:5d} pos {pos:7d} dense (count={int(counts[col])}), skipped")
            continue
        engine_set = set(int(v) for v in engine)
        recomputed = set(int(v) for v in order)
        inter = len(engine_set & recomputed)
        exact += inter == top
        # order-robust: nothing unselected rises above the selected score floor
        floor = score[list(engine_set)].min()
        outside = [b for b in range(n) if b not in engine_set]
        if not outside or score[outside].max() <= floor + 1e-6:
            robust += 1
        if outside:
            worst = max(worst, float(score[outside].max() - floor))
        checked += 1
        print(f"  col {col:5d} pos {pos:7d} blocks {n:6d} overlap {inter}/{top} "
              f"max_outside_minus_floor {float(score[outside].max() - floor) if outside else 0.0:.3e}")
    print(f"{capdir}: checked {checked} columns; exact-set {exact}/{checked}; "
          f"order-robust {robust}/{checked}; worst outside-floor gap {worst:.3e}")
    return checked, exact, robust


if __name__ == "__main__":
    capdir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    check(capdir, n)
