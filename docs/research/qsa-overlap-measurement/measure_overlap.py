#!/usr/bin/env python3
"""Measure cross-query block-selection overlap on the captured QSA prefill selections.

Read-only, CPU-only. Parses the captures of `proto/qsa-index-capture` (head 05f74f1a) exactly as its
`verify_capture.py` does and emits raw per-chunk/per-layer tables plus a summary JSON.

Metric (definition used here; see README):
    R = 1 - |union_q S_q| / sum_q |S_q|
over the **non-dense** query columns q of a prefill chunk, with |S_q| = top = 512 block ids.

Inputs, per prefill chunk (one `selected.bin`/`counts.bin`/`positions.bin` capture):
  * selected.bin = I32 [top=512, columns]; column c's ascending block ids at selected[c*top + i]
  * counts.bin   = I32 [columns]; 512, or -1 for a dense column (never written)
  * positions.bin= I32 [columns]; the absolute KV position of each column
  * meta.txt     = run metadata (columns, top_blocks, ratio, geometry, pooled_ptr, min/max_position)

Usage:
  python3 measure_overlap.py [--base out/qsa-index-capture] [--out results]
                             [--contexts 16k-eager,140k-eager]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

# --------------------------------------------------------------------------------------------------
# capture parsing (same conventions as docs/research/qsa-index-capture/verify_capture.py)

DISTANCES = [1, 2, 4, 8, 16, 64, 256]
# Row bytes per token per KV head, from the facts note §2 (qsa_prompt.cu:44-68):
#   INT8-group64: K 256 + scales 8, V 256 + scales 8 = 528 B
#   BF16:         K 512, V 512                     = 1024 B
#   NVFP4:        K 128 + scales 16, V 128 + 16    =  288 B  (the capture's own --kv-dtype)
ROW_BYTES = {"int8": 528, "bf16": 1024, "nvfp4": 288}


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


class Chunk:
    """One captured prefill selection call."""

    def __init__(self, capdir):
        self.dir = capdir
        self.meta = read_meta(os.path.join(capdir, "meta.txt"))
        m = self.meta
        self.columns = int(m["columns"])
        self.top = int(m["top_blocks"])
        self.ratio = int(m["ratio"])
        self.kv_heads = int(m["kv_heads"])
        self.pooled_ptr = int(m["pooled_ptr"])
        self.step = int(m["step"])
        self.capture = int(m["capture"])
        self.min_position = int(m["min_position"])
        self.max_position = int(m["max_position"])
        self.complete_blocks = int(m["complete_blocks"])
        self.positions = np.fromfile(os.path.join(capdir, "positions.bin"), dtype=np.int32)
        self.counts = np.fromfile(os.path.join(capdir, "counts.bin"), dtype=np.int32)
        self.valid = self.counts != -1
        raw = np.fromfile(os.path.join(capdir, "selected.bin"), dtype=np.int32)
        if raw.size != self.top * self.columns:
            raise ValueError(f"{capdir}: selected.bin size {raw.size} != {self.top}*{self.columns}")
        self.ids = raw.reshape(self.columns, self.top)
        self.valid_ids = self.ids[self.valid]  # (n_valid, top), ascending per column
        self.valid_pos = self.positions[self.valid]

    @property
    def n_valid(self):
        return int(self.valid.sum())

    @property
    def n_dense(self):
        return int((~self.valid).sum())


def collect_prefill_chunks(run_dir, min_columns=1000):
    """Chunk calls: selection captures with >= min_columns columns (prefill chunks only; the decode /
    verify / MTP calls in the same run have <= 13 columns)."""
    root = os.path.join(run_dir, "selections")
    chunks = []
    for dp, _, fns in os.walk(root):
        if "meta.txt" not in fns:
            continue
        meta = read_meta(os.path.join(dp, "meta.txt"))
        if int(meta.get("columns", "0")) >= min_columns:
            chunks.append(Chunk(dp))
    layers = {}
    for c in chunks:
        layers.setdefault(c.pooled_ptr, []).append(c)
    for ptr in layers:
        layers[ptr].sort(key=lambda c: c.min_position)
    return layers


# --------------------------------------------------------------------------------------------------
# per-chunk metrics

def chunk_metrics(chunk):
    ids = chunk.valid_ids
    sum_blocks = int(chunk.counts[chunk.valid].sum())
    hist = np.bincount(ids.ravel())
    distinct = int(np.count_nonzero(hist))
    selected_counts = hist[hist > 0]
    # index = selection count k, value = number of blocks selected by exactly k queries
    count_hist = np.bincount(selected_counts)
    saved = sum_blocks - distinct
    row = {
        "layer": chunk.pooled_ptr,
        "step": chunk.step,
        "capture": chunk.capture,
        "pos_begin": chunk.min_position,
        "pos_end": chunk.max_position,
        "columns": chunk.columns,
        "dense_columns": chunk.n_dense,
        "valid_columns": chunk.n_valid,
        "top": chunk.top,
        "ratio": chunk.ratio,
        "kv_heads": chunk.kv_heads,
        "complete_blocks": chunk.complete_blocks,
        "sum_blocks": sum_blocks,
        "distinct_blocks": distinct,
        "dedup_ratio": (1.0 - distinct / sum_blocks) if sum_blocks else float("nan"),
        "blocks_saved": saved,
        "sum_bytes_int8": sum_blocks * chunk.ratio * chunk.kv_heads * ROW_BYTES["int8"],
        "sum_bytes_bf16": sum_blocks * chunk.ratio * chunk.kv_heads * ROW_BYTES["bf16"],
        "bytes_saved_int8": saved * chunk.ratio * chunk.kv_heads * ROW_BYTES["int8"],
        "bytes_saved_bf16": saved * chunk.ratio * chunk.kv_heads * ROW_BYTES["bf16"],
        "bytes_saved_nvfp4": saved * chunk.ratio * chunk.kv_heads * ROW_BYTES["nvfp4"],
        "singleton_blocks": int(np.count_nonzero(selected_counts == 1)),
        "max_selection_count": int(selected_counts.max()) if selected_counts.size else 0,
    }
    row["singleton_fraction"] = row["singleton_blocks"] / distinct if distinct else float("nan")
    row["hist"] = count_hist
    return row


# --------------------------------------------------------------------------------------------------
# overlap-versus-distance

def layer_distance_curve(chunks, distances=DISTANCES):
    """For one layer (its chunks in position order) return per-distance overlap stats.

    Pairs are keyed on absolute position: two columns are `d` apart when their positions differ by d.
    Candidate pairs only where both columns are non-dense. Scopes: all / same_chunk / cross_chunk.
    """
    pos = np.concatenate([c.valid_pos for c in chunks])
    ids = np.concatenate([c.valid_ids for c in chunks], axis=0)
    n_valid, top = ids.shape
    P = int(pos.max()) + 1
    chunk_starts = np.array([c.min_position for c in chunks], dtype=np.int64)
    chunk_index = np.searchsorted(chunk_starts, pos, side="right") - 1

    enc = np.sort((ids.astype(np.int64) * P + pos[:, None]).ravel())
    N = enc.size
    valid_pos_sorted = np.sort(pos)
    V = valid_pos_sorted.size

    rows = []
    for d in distances:
        # origin columns whose position + d is also a valid (non-dense) column
        tgt = pos + d
        idx = np.searchsorted(valid_pos_sorted, tgt)
        ok = (idx < V)
        ok &= valid_pos_sorted[np.minimum(idx, V - 1)] == tgt
        src = np.nonzero(ok)[0]
        if src.size == 0:
            for scope in ("all", "same_chunk", "cross_chunk"):
                rows.append({"distance": d, "scope": scope, "n_pairs": 0,
                             "mean_intersection": float("nan"), "mean_overlap": float("nan"),
                             "mean_jaccard": float("nan")})
            continue
        cand_shift = (ids[src].astype(np.int64) * P + pos[src, None]).ravel() + d
        j = np.searchsorted(enc, cand_shift)
        found = ((j < N) & (enc[np.minimum(j, N - 1)] == cand_shift)).reshape(src.size, top)
        per_src = found.sum(axis=1)  # |S_q ∩ S_{q+d}| for each origin column
        ochunk = chunk_index[src]
        tchunk = np.searchsorted(chunk_starts, pos[src] + d, side="right") - 1
        for scope in ("all", "same_chunk", "cross_chunk"):
            if scope == "all":
                m = np.ones(src.size, dtype=bool)
            elif scope == "same_chunk":
                m = ochunk == tchunk
            else:
                m = ochunk != tchunk
            n_pairs = int(m.sum())
            if n_pairs == 0:
                rows.append({"distance": d, "scope": scope, "n_pairs": 0,
                             "mean_intersection": float("nan"), "mean_overlap": float("nan"),
                             "mean_jaccard": float("nan")})
                continue
            inter = per_src[m].astype(np.float64)
            rows.append({
                "distance": d,
                "scope": scope,
                "n_pairs": n_pairs,
                "mean_intersection": float(inter.mean()),
                "mean_overlap": float(inter.mean() / top),
                "mean_jaccard": float((inter / (2.0 * top - inter)).mean()),
            })
    return rows


# --------------------------------------------------------------------------------------------------

def format_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_tsv(path, rows, columns):
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(format_cell(r.get(c)) for c in columns) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(here, "..", "..", "..", "out", "qsa-index-capture"))
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--contexts", default="16k-eager,140k-eager")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    contexts = [c for c in args.contexts.split(",") if c]
    per_chunk_rows = []
    hist_rows = []
    distance_rows = []
    summary = {"base": base, "contexts": {}, "row_bytes": ROW_BYTES,
               "metric": "R = 1 - |union_q S_q| / sum_q |S_q|  over non-dense columns q, |S_q|=top"}

    for ctx in contexts:
        run_dir = os.path.join(base, ctx)
        layers = collect_prefill_chunks(run_dir)
        ptrs = sorted(layers)
        ctx_sum = {"run": ctx, "layers": len(ptrs), "chunks_per_layer": 0,
                   "dense_columns": 0, "valid_columns": 0, "sum_blocks": 0,
                   "distinct_blocks_summed": 0, "chunks_detail": []}
        t0 = time.time()
        layer_ratios = []
        for li, ptr in enumerate(ptrs):
            chunks = layers[ptr]
            ctx_sum["chunks_per_layer"] = max(ctx_sum["chunks_per_layer"], len(chunks))
            for ci, ch in enumerate(chunks):
                row = chunk_metrics(ch)
                row["context"] = ctx
                row["layer_index"] = li
                row["chunk_index"] = ci
                hist = row.pop("hist")
                per_chunk_rows.append(row)
                for k in range(1, hist.size):
                    if hist[k]:
                        hist_rows.append({"context": ctx, "layer": ptr, "layer_index": li,
                                          "chunk_index": ci, "selection_count": k,
                                          "n_blocks": int(hist[k])})
                ctx_sum["dense_columns"] += row["dense_columns"]
                ctx_sum["valid_columns"] += row["valid_columns"]
                ctx_sum["sum_blocks"] += row["sum_blocks"]
                ctx_sum["distinct_blocks_summed"] += row["distinct_blocks"]
                ctx_sum["chunks_detail"].append({
                    "layer_index": li, "pooled_ptr": ptr, "chunk_index": ci,
                    "pos_begin": row["pos_begin"], "pos_end": row["pos_end"],
                    "valid_columns": row["valid_columns"], "dense_columns": row["dense_columns"],
                    "sum_blocks": row["sum_blocks"], "distinct_blocks": row["distinct_blocks"],
                    "dedup_ratio": row["dedup_ratio"], "blocks_saved": row["blocks_saved"],
                    "bytes_saved_int8": row["bytes_saved_int8"],
                    "bytes_saved_bf16": row["bytes_saved_bf16"],
                    "singleton_fraction": row["singleton_fraction"],
                })
                layer_ratios.append(row["dedup_ratio"])
            for r in layer_distance_curve(chunks):
                r.update({"context": ctx, "layer": ptr, "layer_index": li})
                distance_rows.append(r)
            del layers[ptr]
        ctx_sum["pooled_dedup_ratio"] = (
            1.0 - ctx_sum["distinct_blocks_summed"] / ctx_sum["sum_blocks"]
            if ctx_sum["sum_blocks"] else float("nan"))
        r_arr = np.array(layer_ratios, dtype=float)
        ctx_sum["ratio_mean"] = float(np.nanmean(r_arr))
        ctx_sum["ratio_min"] = float(np.nanmin(r_arr))
        ctx_sum["ratio_max"] = float(np.nanmax(r_arr))
        ctx_sum["ratio_std"] = float(np.nanstd(r_arr))
        ctx_sum["seconds"] = round(time.time() - t0, 1)
        summary["contexts"][ctx] = ctx_sum
        print(f"[{ctx}] layers={len(ptrs)} chunk-rows={ctx_sum['layers'] * ctx_sum['chunks_per_layer']} "
              f"pooled ratio={ctx_sum['pooled_dedup_ratio']:.4f} mean={ctx_sum['ratio_mean']:.4f} "
              f"min={ctx_sum['ratio_min']:.4f} max={ctx_sum['ratio_max']:.4f} "
              f"({ctx_sum['seconds']}s)", file=sys.stderr)

    chunk_cols = ["context", "layer", "layer_index", "chunk_index", "step", "capture",
                  "pos_begin", "pos_end", "columns", "dense_columns", "valid_columns", "top",
                  "ratio", "kv_heads", "complete_blocks", "sum_blocks", "distinct_blocks",
                  "dedup_ratio", "blocks_saved", "sum_bytes_int8", "sum_bytes_bf16",
                  "bytes_saved_int8", "bytes_saved_bf16", "bytes_saved_nvfp4",
                  "singleton_blocks", "singleton_fraction", "max_selection_count"]
    write_tsv(os.path.join(outdir, "per_chunk_layer.tsv"), per_chunk_rows, chunk_cols)
    write_tsv(os.path.join(outdir, "selection_count_histogram.tsv"), hist_rows,
              ["context", "layer", "layer_index", "chunk_index", "selection_count", "n_blocks"])
    write_tsv(os.path.join(outdir, "distance_curve.tsv"), distance_rows,
              ["context", "layer", "layer_index", "scope", "distance", "n_pairs",
               "mean_intersection", "mean_overlap", "mean_jaccard"])
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
