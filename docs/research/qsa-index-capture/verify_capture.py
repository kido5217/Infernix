#!/usr/bin/env python3
"""Recompute QSA block scores from a capture and check them against the engine's own numbers.

Usage: python3 verify_capture.py CAPDIR [n_columns]

Reads step_*/{s_|p_}*/{meta.txt,pooled.bin,index_q.bin,selected.bin,counts.bin,scores.bin}:

  * scores:  recomputes score_b = 1/sqrt(Di) * sum_h relu(<q_h, pooled_key_b>) exactly as the engine
    defines it, from the captured pooled keys and index query, and compares it against the engine's
    captured FP32 scores (scores.bin, ops::qsa_select_scores) for the selection's last group. Reports
    the maximum absolute and relative difference (float-rounding level is expected: the engine
    accumulates in its own FP32 order, numpy in another).
  * ids:     checks the recorded top-(budget/ratio) block set against the recomputed top-k.

Captures without scores.bin (the pre-extension runs) are checked for ids only.
"""
import numpy as np
import os
import re
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


def read_inputs(capdir):
    meta = read_meta(os.path.join(capdir, "meta.txt"))
    return meta, {
        "positions": np.fromfile(os.path.join(capdir, "positions.bin"), dtype=np.int32),
        "pooled": np.fromfile(os.path.join(capdir, "pooled.bin"), dtype=np.uint8),
        "query": np.fromfile(os.path.join(capdir, "index_q.bin"), dtype=np.uint16)
        if os.path.exists(os.path.join(capdir, "index_q.bin"))
        else None,
        "selected": np.fromfile(os.path.join(capdir, "selected.bin"), dtype=np.int32)
        if os.path.exists(os.path.join(capdir, "selected.bin"))
        else None,
        "counts": np.fromfile(os.path.join(capdir, "counts.bin"), dtype=np.int32)
        if os.path.exists(os.path.join(capdir, "counts.bin"))
        else None,
        "scores": np.fromfile(os.path.join(capdir, "scores.bin"), dtype=np.float32)
        if os.path.exists(os.path.join(capdir, "scores.bin"))
        else None,
    }


def token_slice(pooled, slots, page_bytes, t):
    off = (t >> 6) * page_bytes + slots * (t & 63) * 2
    return pooled[off:off + slots * 2].view(BF16)


def block_keys(pooled, slots, page_bytes, ratio, n):
    dim = 4 * slots
    keys = np.empty((n, dim), dtype=np.uint16)
    for b in range(n):
        for s in range(ratio):
            keys[b, s * slots:(s + 1) * slots] = token_slice(pooled, slots, page_bytes, ratio * b + s)
    return keys


def column_query(query, dim, heads, col):
    q = np.empty((heads, dim), dtype=np.float32)
    for h in range(heads):
        base = dim * (h + heads * col)
        q[h] = bf16_to_f32(query[base:base + dim])
    return q


def recompute(pooled, query, meta, col, n):
    slots = int(meta["plane_slots"])
    page_bytes = int(meta["plane_page_bytes"])
    ratio = int(meta["ratio"])
    dim = int(meta["index_head_dim"])
    heads = int(meta["index_heads"])
    keys = bf16_to_f32(block_keys(pooled, slots, page_bytes, ratio, n))
    q = column_query(query, dim, heads, col)
    dots = keys @ q.T
    return np.maximum(dots, 0.0).sum(axis=1) / np.sqrt(dim)


def check(capdir, n_columns=8, check_scores=True):
    meta, data = read_inputs(capdir)
    columns = int(meta["columns"])
    top = int(meta["top_blocks"])
    ratio = int(meta["ratio"])
    slots = int(meta["plane_slots"])
    page_bytes = int(meta["plane_page_bytes"])
    pages_used = int(meta["plane_pages_used"])
    positions = data["positions"]
    pooled = data["pooled"]
    query = data["query"]
    selected = data["selected"]
    counts = data["counts"]
    scores = data["scores"]
    has_selection = selected is not None and counts is not None and query is not None

    assert pooled.size == pages_used * page_bytes, (pooled.size, pages_used * page_bytes)

    if check_scores and scores is not None and query is not None:
        stride = int(meta["scores_stride"])
        group_begin = int(meta["scores_group_begin"])
        group_columns = int(meta["scores_group_columns"])
        assert scores.size == group_columns * stride, (scores.size, group_columns * stride)
        board = scores.reshape(group_columns, stride)
        worst_abs = 0.0
        worst_rel = 0.0
        worst_scaled = 0.0
        checked = 0
        for c in range(min(group_columns, n_columns)):
            col = group_begin + c
            n = (int(positions[col]) + 1) // ratio
            n = min(n, pages_used * 64 // ratio)
            if n <= 0:
                continue
            mine = recompute(pooled, query, meta, col, n)
            theirs = board[c, :n]
            diff = np.abs(mine - theirs)
            scale = max(float(np.abs(theirs).max()), 1e-30)
            with np.errstate(divide="ignore", invalid="ignore"):
                elem = np.nan_to_num(diff / np.maximum(np.abs(theirs), 1e-30), nan=0.0, posinf=0.0)
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float(elem.max()))
            worst_scaled = max(worst_scaled, float((diff / scale).max()))
            checked += 1
        print(f"{capdir}: scores checked {checked} columns; max abs diff {worst_abs:.3e}; "
              f"max relative diff {worst_rel:.3e} elementwise, {worst_scaled:.3e} column-scaled")
    elif check_scores:
        print(f"{capdir}: no scores.bin (ids only)")

    if not has_selection:
        return
    checked = exact = robust = 0
    for col in range(min(n_columns, columns)):
        pos = int(positions[col])
        blocks = (pos + 1) // ratio
        if blocks <= top:
            continue
        n = min(blocks, pages_used * 64 // ratio)
        score = recompute(pooled, query, meta, col, n)
        order = np.lexsort((np.arange(n), -score))[:top]
        engine = selected[col * top:(col + 1) * top]
        if int(counts[col]) != top:
            continue
        engine_set = set(int(v) for v in engine)
        recomputed = set(int(v) for v in order)
        exact += len(engine_set & recomputed) == top
        floor = score[list(engine_set)].min()
        outside = [b for b in range(n) if b not in engine_set]
        if not outside or score[outside].max() <= floor + 1e-6:
            robust += 1
        checked += 1
    print(f"{capdir}: ids checked {checked} columns; exact-set {exact}/{checked}; order-robust {robust}/{checked}")


if __name__ == "__main__":
    capdir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    check(capdir, n)
