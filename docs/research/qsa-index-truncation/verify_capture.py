#!/usr/bin/env python3
"""Recompute QSA block scores from a capture and check them against the engine's own numbers.

Usage:
  python3 verify_capture.py CAPDIR [n_columns]           one selection capture
  python3 verify_capture.py --sweep BASEDIR [n_columns]  every capture under BASEDIR

Reads step_*/{s_|p_}*/{meta.txt,pooled.bin,index_q.bin,selected.bin,counts.bin,scores.bin}:

  * scores:  recomputes score_b = 1/sqrt(Di) * sum_h relu(<q_h, pooled_key_b>) from the captured
    pooled keys and index query, and compares it against the engine's captured FP32 scores
    (scores.bin, ops::qsa_select_scores) for the selection's last group. Reports the maximum
    absolute and relative difference (float-rounding level is expected: the engine accumulates in
    its own FP32 order, numpy in another).
  * ids:     checks the recorded top-(budget/ratio) block set against the recomputed top-k.
    Dense columns (counts[c] == -1, the engine's short-circuit -- see pre-registration rule 11) are
    excluded from both checks and counted as excluded: the engine never writes their selection or
    scores, so their bytes are unwritten buffer memory.
  * gap:     for every checked column the boundary gap |s(engine's last selected) - s(engine's
    first rejected)| is computed. A mismatch is "explained" when that gap is below the column's
    measured score error (the accumulation-order noise a different summation order produces) and
    "unexplained" otherwise. The robustness count uses the measured error as its threshold, never
    a fixed 1e-6.

Captures without scores.bin (the pre-extension runs) are checked for ids only; a mismatch there is
reported as unclassified, because the measured score error cannot be computed without the engine's
scores. verify_repro.py compares file hashes only and does not share these checks.
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

    def load(name, dtype):
        path = os.path.join(capdir, name)
        return np.fromfile(path, dtype=dtype) if os.path.exists(path) else None

    return meta, {
        "positions": np.fromfile(os.path.join(capdir, "positions.bin"), dtype=np.int32),
        "pooled": np.fromfile(os.path.join(capdir, "pooled.bin"), dtype=np.uint8),
        "query": load("index_q.bin", np.uint16),
        "selected": load("selected.bin", np.int32),
        "counts": load("counts.bin", np.int32),
        "scores": load("scores.bin", np.float32),
    }


def block_keys(pooled, slots, page_bytes, ratio, n):
    """BF16 keys of blocks 0..n-1: `ratio` tokens x `slots` lanes, in the plane's token-page layout."""
    tokens = ratio * np.arange(n, dtype=np.int64)[:, None] + np.arange(ratio, dtype=np.int64)[None, :]
    offs = (tokens >> 6) * page_bytes + slots * (tokens & 63) * 2
    raw = pooled[offs[..., None] + np.arange(slots * 2)[None, None, :]]
    return raw.view(BF16).reshape(n, ratio * slots)


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


def boundary_gap(score, engine_set, n):
    """|s(engine's last selected) - s(engine's first rejected)| on `score`."""
    selected = np.fromiter(engine_set, dtype=np.int64, count=len(engine_set))
    rejected = np.setdiff1d(np.arange(n, dtype=np.int64), selected)
    if rejected.size == 0:
        return 0.0
    return abs(float(score[selected].min()) - float(score[rejected].max()))


def analyze(capdir, n_columns=8):
    meta, data = read_inputs(capdir)
    columns = int(meta["columns"])
    top = int(meta["top_blocks"])
    ratio = int(meta["ratio"])
    pages_used = int(meta["plane_pages_used"])
    positions = data["positions"]
    pooled = data["pooled"]
    query = data["query"]
    selected = data["selected"]
    counts = data["counts"]
    scores = data["scores"]

    has_scores = scores is not None and query is not None
    has_selection = selected is not None and counts is not None and query is not None
    if pooled.size != pages_used * int(meta["plane_page_bytes"]):
        raise ValueError(f"{capdir}: pooled.bin size {pooled.size} != {pages_used} pages")

    res = {
        "capdir": capdir, "has_scores": has_scores, "has_selection": has_selection,
        "score_checked": 0, "score_dense": 0, "max_abs": 0.0, "max_rel": 0.0, "max_scaled": 0.0,
        "ids_checked": 0, "exact": 0, "explained": 0, "unexplained": 0, "unclassified": 0,
        "id_dense": 0, "near_tie": 0, "min_gap": None, "mismatches": [], "columns": [],
    }

    board = None
    group_begin = 0
    group_columns = 0
    err_by_col = {}
    cap_err = None
    if has_scores:
        stride = int(meta["scores_stride"])
        group_begin = int(meta["scores_group_begin"])
        group_columns = int(meta["scores_group_columns"])
        if scores.size != group_columns * stride:
            raise ValueError(f"{capdir}: scores.bin size {scores.size} != {group_columns}*{stride}")
        board = scores.reshape(group_columns, stride)
        for c in range(min(group_columns, n_columns)):
            col = group_begin + c
            if counts is not None and int(counts[col]) == -1:
                res["score_dense"] += 1
                continue
            n = min((int(positions[col]) + 1) // ratio, pages_used * 64 // ratio)
            if n <= 0:
                continue
            mine = recompute(pooled, query, meta, col, n)
            theirs = board[c, :n]
            diff = np.abs(mine - theirs)
            scale = max(float(theirs.max() - theirs.min()), 1e-30)
            with np.errstate(divide="ignore", invalid="ignore"):
                elem = np.nan_to_num(diff / np.maximum(np.abs(theirs), 1e-30), nan=0.0, posinf=0.0)
            res["max_abs"] = max(res["max_abs"], float(diff.max()))
            res["max_rel"] = max(res["max_rel"], float(elem.max()))
            res["max_scaled"] = max(res["max_scaled"], float((diff / scale).max()))
            err_by_col[col] = float(diff.max())
            cap_err = float(diff.max()) if cap_err is None else max(cap_err, float(diff.max()))
            res["score_checked"] += 1
    res["cap_err"] = cap_err

    if not has_selection:
        return res

    for col in range(min(n_columns, columns)):
        if int(counts[col]) == -1:
            res["id_dense"] += 1
            continue
        n = min((int(positions[col]) + 1) // ratio, pages_used * 64 // ratio)
        if n <= 0:
            continue
        score = recompute(pooled, query, meta, col, n)
        order = np.lexsort((np.arange(n), -score))[:top]
        engine = selected[col * top:(col + 1) * top]
        engine_set = set(int(v) for v in engine)
        recomputed = set(int(v) for v in order)
        res["ids_checked"] += 1

        # The engine's scores cover the last group only; elsewhere the recomputed scores stand in.
        if has_scores and group_begin <= col < group_begin + group_columns:
            gap_score = board[col - group_begin, :n]
        else:
            gap_score = score
        gap = boundary_gap(gap_score, engine_set, n)
        err = err_by_col.get(col, cap_err) if has_scores else None
        if res["min_gap"] is None or gap < res["min_gap"]:
            res["min_gap"] = gap
        if err is not None and gap <= err:
            res["near_tie"] += 1

        if engine_set == recomputed:
            res["exact"] += 1
            res["columns"].append({"col": col, "gap": gap, "error": err, "verdict": "exact"})
            continue

        if err is None:
            verdict = "unclassified"
            res["unclassified"] += 1
        elif gap < err:
            verdict = "explained"
            res["explained"] += 1
        else:
            verdict = "unexplained"
            res["unexplained"] += 1
        selected_ids = np.fromiter(engine_set, dtype=np.int64, count=len(engine_set))
        rejected = np.setdiff1d(np.arange(n, dtype=np.int64), selected_ids)
        record = {
            "col": col,
            "engine_last": int(selected_ids[np.argmin(gap_score[selected_ids])]),
            "first_rejected": None if rejected.size == 0 else int(rejected[np.argmax(gap_score[rejected])]),
            "gap": gap, "error": err, "verdict": verdict,
            "symmetric_difference": sorted(engine_set ^ recomputed),
        }
        res["columns"].append(record)
        res["mismatches"].append(record)
    return res


def print_capture(res):
    d = res["capdir"]
    if res["has_scores"]:
        print(f"{d}: scores: checked {res['score_checked']} columns, dense-excluded {res['score_dense']}; "
              f"max abs diff {res['max_abs']:.3e}; max relative diff {res['max_rel']:.3e} elementwise, "
              f"{res['max_scaled']:.3e} column-scaled")
    else:
        print(f"{d}: no scores.bin (ids only)")
    if not res["has_selection"]:
        return
    line = (f"{d}: ids: checked {res['ids_checked']} columns; exact {res['exact']}/{res['ids_checked']}; "
            f"explained {res['explained']}; unexplained {res['unexplained']}; "
            f"unclassified {res['unclassified']}; dense-excluded {res['id_dense']}; "
            f"near-tie {res['near_tie']}")
    if res["min_gap"] is not None:
        line += f"; min boundary gap {res['min_gap']:.3e}"
    print(line)
    for r in res["columns"]:
        err = "n/a" if r["error"] is None else f"{r['error']:.3e}"
        tail = ""
        if r["verdict"] != "exact":
            tail = (f" engine last-selected {r['engine_last']}, first-rejected {r['first_rejected']}, "
                    f"symmetric difference {r['symmetric_difference']}")
        print(f"{d}: col {r['col']}: boundary gap {r['gap']:.3e}; score error {err}; {r['verdict']}{tail}")


def capture_dirs(root):
    selection = os.path.join(root, "selections")
    out = []
    for dp, _, fns in os.walk(selection):
        if "meta.txt" in fns:
            out.append(dp)
    return sorted(out)


def sweep(base, n_columns=8):
    roots = [os.path.join(base, name) for name in sorted(os.listdir(base))
             if os.path.isdir(os.path.join(base, name, "selections"))]
    if not roots:
        roots = [base]
    header = (f"{'manifest':<16} {'captures':>8} {'ids-chk':>7} {'exact':>6} {'expl':>5} {'unexpl':>6} "
              f"{'unclass':>7} {'dense':>6} {'near':>5} {'max|dScore|':>12} {'max col-scaled':>14}")
    print(header)
    print("-" * len(header))
    for root in roots:
        agg = {"ids_checked": 0, "exact": 0, "explained": 0, "unexplained": 0, "unclassified": 0,
               "id_dense": 0, "near_tie": 0, "max_abs": 0.0, "max_scaled": 0.0, "has_scores": False}
        captures = capture_dirs(root)
        for capdir in captures:
            r = analyze(capdir, n_columns)
            for key in ("ids_checked", "exact", "explained", "unexplained", "unclassified",
                        "id_dense", "near_tie"):
                agg[key] += r[key]
            agg["max_abs"] = max(agg["max_abs"], r["max_abs"])
            agg["max_scaled"] = max(agg["max_scaled"], r["max_scaled"])
            agg["has_scores"] = agg["has_scores"] or r["has_scores"]
        err = f"{agg['max_abs']:.3e}" if agg["has_scores"] else "n/a"
        scaled = f"{agg['max_scaled']:.3e}" if agg["has_scores"] else "n/a"
        print(f"{os.path.basename(root):<16} {len(captures):>8} {agg['ids_checked']:>7} {agg['exact']:>6} "
              f"{agg['explained']:>5} {agg['unexplained']:>6} {agg['unclassified']:>7} "
              f"{agg['id_dense']:>6} {agg['near_tie']:>5} {err:>12} {scaled:>14}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--sweep":
        base = argv[1]
        n = int(argv[2]) if len(argv) > 2 else 8
        sweep(base, n)
    else:
        capdir = argv[0]
        n = int(argv[1]) if len(argv) > 1 else 8
        print_capture(analyze(capdir, n))
