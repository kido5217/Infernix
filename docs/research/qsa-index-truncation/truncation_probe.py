#!/usr/bin/env python3
"""QSA index truncation prototype (issue #39 rev 5 protocol).

Question: does a coarser index plane preserve the selected blocks, and what does it cost in
bytes?  CPU-only, over ticket #40's captured data.  Primary dataset = the eager runs
(``16k-eager`` and ``140k-eager``); the bar applies at >= 128K, 16K is the trend.

The binding spec is the first comment of kido5217/Infernix issue #39 (rev 5).  Nothing here is
chosen after seeing an arm's agreement; open points are recorded in ``deviations.txt`` (written
before results are read) and in README.md.

Usage (inside the dev shell; numpy only, no GPU):

    nix develop . --command python3 docs/research/qsa-index-truncation/truncation_probe.py \
        fidelity --out raw [--contexts 16k-eager,140k-eager] [--limit-captures N] [--max-columns M]
    ... arms --out raw --arms A1,A2 [--limit-captures N]
    ... a3   --out raw            # picks A4 from the fixed TSVs, calibrates B by bits, runs V1/V2
    ... report --out raw          # per-layer tables, budget curve, bar verdict

Format parsing is vendored from ``verify_capture.py`` (branch ``proto/qsa-index-capture``, head
05f74f1a): ``read_meta`` / ``read_inputs`` / ``block_keys`` are reused verbatim; the loops here
are restructured for full-column coverage with bounded memory.

Arms (protocol section 2):
  A0    full BF16                                    256 B/block
  A1(t) uniform symmetric t-bit code + fp16 scale     16*t + 16 B/block   t=1..4
  A2(m) leading m bits of each BF16 word, read back    16*m B/block        m in {8,9,10,12,16}
  A3    water-filling depth (V1 gating / V2 reported)  (sum_j t_j)/8 + 16 B/block
  A4    one fixed A1 budget chosen by the section 2 rule
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import verify_capture as vc  # noqa: E402  (vendored parsing logic)

# --- fixed constants ---------------------------------------------------------------------------
CAPTURE_ROOT = "/home/kido/network/projects/Infernix/out/qsa-index-capture"
CONTEXTS = {"16k-eager": "16k-eager", "140k-eager": "140k-eager"}
SCALE_GROUP_BLOCKS = 16          # paper's 64-token block == 16 captured 4-token blocks
TOP = 512                        # budget 2048 / ratio 4
NUM_CHANNELS = 128               # index_head_dim (Di)
NUM_HEADS = 4                    # index_heads
FRAME_BYTES = 2_764_800          # one routed-expert frame (design doc: 2.64 MiB)
DEVICE_BANDWIDTH = 1.7e12        # bytes/s, the research note's modelled device bandwidth floor
FID_ABS_BOUND = 1e-5
FID_SCALED_BOUND = 1e-6
BAR_MEAN, BAR_LAYER, BAR_SPREAD = 0.95, 0.90, 0.10
A3_GATE_MARGIN, A3_BITS_TOL = 0.02, 0.02

A1_ARMS = [f"A1_{t}" for t in (1, 2, 3, 4)]
A2_ARMS = [f"A2_{m}" for m in (8, 9, 10, 12, 16)]
FIXED_ARMS = ["A0re"] + A1_ARMS + A2_ARMS
A3_ARMS = ["A3_V1", "A3_V2"]

DEVIATIONS = """\
Deviations / interpretations recorded before any arm's agreement was read.

D1  Scale-group alignment.  The paper's 64-token block is 16 consecutive captured 4-token blocks
    starting at block 0, i.e. exactly one pooled-plane token page (page = block>>4).  The capture's
    plane is indexed by absolute token, so group g = page g.  Reason: the protocol fixes one fp16
    scale per channel per 16 captured blocks; the page is the only 64-token unit the build has.
D2  A1 is the paper's single 4-bit quantization shifted.  Per channel j and 16-block group, the
    scale a = max over the group's 16 pooled block keys of |key| (fp16-rounded, as stored); s = a/8;
    c4 = clip(floor(key/s), -8, 7) + 8 (MSB sign, code 8 = zero).  A1(t) reads c_t = c4 >> (4-t)
    and reconstructs (c_t + 1/2 - 2^(t-1)) * a/2^(t-1), the paper's t-bit mid-rise quantizer.  The
    true keys are the pooled block keys, not raw per-token keys (not captured); a is the max of
    those stored keys.
D3  A1(t) stores a t-bit code (the protocol's byte table 16t+16); the paper stores 4 bits and reads
    t planes -- byte-equivalent for the read.
D4  A2 truncates each BF16 word's leading m bits, zeroes the rest, and reads the word back as BF16
    with no rounding; A2(16) equals A0 by construction.
D5  A3 V1 (paper's rule, gating).  Per capture (one layer call) Var(k_j) is the variance of channel
    j over that capture's stored block keys; g_j = sum_h (q_j^(h))^2 Var(k_j).  A depth vector
    t in [0,4]^128 per column is produced by bisecting the water line theta so that sum_j t_j <= B
    (30 bisections on log theta over [log 0.001, log 10], the paper's bracket).
D6  A3 V2 (per-block variant, reported, non-gating).  a_{beta,j}^2 replaces Var(k_j); because a is
    constant over the 16 blocks of a group it is constant per (group, channel), so the depth vector
    is per (group, column) and each block reads its group's depth.
D7  Rank agreement (section 3.1) = |A0_top64 intersect arm_top64| / 64, where A0_top64 is the 64
    highest-scoring blocks under A0's own scores (the engine's captured scores where the column is
    scored, else the recomputed full-BF16 scores).  selected.bin is ascending, so rank comes from
    the scores; "in rank order" is read as "A0's top 64 by rank" (no positional correlation).
D8  A3 budget calibration (equal mean bits, section 4).  A4 = A1(t*) for the smallest t clearing
    the bar; its mean bits/block = 128*(t*+1).  The A3 budget B is bisected (integer grid, mean
    bits monotone in B) so the variant's mean bits/block over the context's non-dense columns is
    within +/-2 % of A4's.  The calibration uses a deterministic sample (first 64 non-dense columns
    of each capture) and bits only, never agreement; the realized full-data mean bits are reported.
D9  Score error (section 3.2) e = sum_b |s_arm,b - s_A0,b| / (n * (max s_A0 - min s_A0)) over the
    scored columns only, against A0's captured scores, reported as each layer's mean.
D10 Bytes/frames.  Plane total = bytes/block * complete blocks of the deepest captured column
    (16k: 5610; 140k: 35426).  Expert frames = plane bytes / 2,764,800 B (the measured routed-expert
    frame).  Reported per layer and for all 13 planes.
D11 Time (section 3.4, non-gating).  No device scan exists here.  Reported: the prototype's wall
    time per stage (numpy score+select, CPU) and the modelled device bandwidth floor (bytes / 1.7
    TB/s) from the research note.  Neither gates.
D12 Layer identity.  No explicit QSA layer index exists; a layer is its plane pointer (pooled_ptr),
    ranked per run after sorting (13 distinct values; index 12 is the MTP plane).
D13 Contexts are scored separately (section 7.12); 16K is the trend, the bar applies at 140K only.
D14 Dense columns (counts == -1, n <= 512) are excluded from every metric and counted as excluded.
D15 Input identity.  The vendored manifests are the hash record; the run checks every file it reads
    against the manifest and records combined digests in input_hashes.txt.
D16 The query is never quantized: arms coarsen the keys only.
"""


# ------------------------------------------------------------------------------------------------
# Capture access
# ------------------------------------------------------------------------------------------------
class Capture:
    """One selection capture, loaded and parsed; keys/query/n/engine data ready for scoring."""

    def __init__(self, capdir, limit_columns=None):
        self.capdir = capdir
        self.meta, data = vc.read_inputs(capdir)
        m = self.meta
        self.columns = int(m["columns"])
        self.cols = self.columns if limit_columns is None else min(self.columns, limit_columns)
        self.positions = data["positions"]
        self.pages_used = int(m["plane_pages_used"])
        self.ratio = int(m["ratio"])
        n_all = np.minimum((self.positions.astype(np.int64) + 1) // self.ratio,
                           self.pages_used * 64 // self.ratio)
        self.n_per_col = n_all[:self.cols]
        self.counts = data["counts"][:self.cols]
        self.dense = self.counts == -1
        self.non_dense = np.nonzero(~self.dense)[0]
        self.keys_bits = _block_keys_bits(data["pooled"], m, int(self.n_per_col.max()))
        self.keys_f32 = vc.bf16_to_f32(self.keys_bits)
        self.q = _all_queries(data["query"], m)[:, :, :self.cols]
        self.engine_ids = data["selected"].reshape(self.columns, int(m["top_blocks"])).T[:, :self.cols]
        stride = int(m["scores_stride"])
        self.group_begin = int(m["scores_group_begin"])
        self.group_columns = int(m["scores_group_columns"])
        self.board = (data["scores"].reshape(self.group_columns, stride)
                      if data["scores"] is not None else None)

    def col_scored(self, col):
        return (self.board is not None and
                self.group_begin <= col < self.group_begin + self.group_columns)

    def captured_scores(self, col, n):
        return self.board[col - self.group_begin, :n].astype(np.float32)


def _block_keys_bits(pooled, meta, n):
    slots = int(meta["plane_slots"])
    page_bytes = int(meta["plane_page_bytes"])
    ratio = int(meta["ratio"])
    return vc.block_keys(pooled, slots, page_bytes, ratio, n).astype(np.uint16)


def _all_queries(query, meta):
    dim = int(meta["index_head_dim"])
    heads = int(meta["index_heads"])
    cols = int(meta["columns"])
    # index_q(d,h,c) at d + dim*(h + heads*c)  ->  reshape (cols,heads,dim), then [heads,dim,cols]
    q = query.reshape(cols, heads, dim)
    return np.ascontiguousarray(np.transpose(vc.bf16_to_f32(q), (1, 2, 0)))


def layer_map(root):
    ptrs = set()
    for capdir in vc.capture_dirs(root):
        ptrs.add(int(vc.read_meta(os.path.join(capdir, "meta.txt"))["pooled_ptr"]))
    ordered = sorted(ptrs)
    return {p: i for i, p in enumerate(ordered)}


# ------------------------------------------------------------------------------------------------
# Arm transforms
# ------------------------------------------------------------------------------------------------
def a1_prepare(keys_f32, group=SCALE_GROUP_BLOCKS):
    n, d = keys_f32.shape
    g = (n + group - 1) // group
    kp = np.zeros((g * group, d), np.float32)
    kp[:n] = keys_f32
    a = np.abs(kp.reshape(g, group, d)).max(axis=1)   # [g,d]
    a16 = a.astype(np.float16).astype(np.float32)     # fp16 as stored
    a_block = np.repeat(a16, group, axis=0)[:n]
    s_block = np.repeat(a16 / 8.0, group, axis=0)[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        code = np.where(s_block > 0, keys_f32 / s_block, 0.0)
    c4 = (np.clip(np.floor(code), -8.0, 7.0) + 8.0).astype(np.int16)
    return c4, a_block


def a1_reconstruct(c4, a_block, t):
    if t <= 0:
        return np.zeros_like(a_block, dtype=np.float32)
    ct = (c4 >> (4 - t)).astype(np.float32)
    scale = float(2 ** (t - 1))
    return (ct + 0.5 - scale) * (a_block / scale)


def a2_reconstruct(keys_bits, m):
    mask = np.uint16(0xFFFF ^ ((1 << (16 - m)) - 1))
    return vc.bf16_to_f32((keys_bits & mask).astype(np.uint16))


# ------------------------------------------------------------------------------------------------
# Scoring / selection
# ------------------------------------------------------------------------------------------------
def score_all(kq, q, chunk=128):
    n, C = kq.shape[0], q.shape[2]
    out = np.empty((n, C), np.float32)
    root = float(np.sqrt(kq.shape[1]))
    for s in range(0, C, chunk):
        e = min(s + chunk, C)
        dots = np.tensordot(kq, q[:, :, s:e], axes=([1], [1]))
        out[:, s:e] = np.maximum(dots, 0.0).sum(axis=1) / root
    return out


def score_one(kq, q, col):
    dots = kq @ q[:, :, col].T
    return np.maximum(dots, 0.0).sum(axis=1) / float(np.sqrt(kq.shape[1]))


def ranked_top(S, n_per_col, top=TOP):
    n, C = S.shape
    rows = np.arange(n)[:, None]
    Sm = np.where(rows < n_per_col[None, :], S, -np.inf)
    part = np.argpartition(-Sm, top - 1, axis=0)[:top]
    ps = np.take_along_axis(Sm, part, axis=0)
    order = np.argsort(-ps, axis=0, kind="stable")
    return np.take_along_axis(part, order, axis=0), Sm


def rows_intersect(a_ids, b_ids):
    C = a_ids.shape[1]
    out = np.empty(C, np.int64)
    for c in range(C):
        out[c] = np.intersect1d(a_ids[:, c], b_ids[:, c], assume_unique=True).size
    return out


def a0_top64(a_score, n_per_col):
    rows = np.arange(a_score.shape[0])[:, None]
    Sm = np.where(rows < n_per_col[None, :], a_score, -np.inf)
    part = np.argpartition(-Sm, 63, axis=0)[:64]
    ps = np.take_along_axis(Sm, part, axis=0)
    order = np.argsort(-ps, axis=0, kind="stable")
    return np.take_along_axis(part, order, axis=0)


# ------------------------------------------------------------------------------------------------
# A3 reconstruction
# ------------------------------------------------------------------------------------------------
def _recon(c4, a_block, t):
    ct = (c4 >> (4 - t)).astype(np.float32)
    scale = np.where(t > 0, np.float32(2.0) ** (t - 1), np.float32(1.0))
    return np.where(t > 0, (ct + 0.5 - scale) * (a_block / scale), 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------------------------
def _write_dev_and_hashes(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "deviations.txt"), "w") as f:
        f.write(DEVIATIONS)
    _write_input_hashes(out)


def _write_input_hashes(out):
    import hashlib
    lines = []
    for name in ("manifest-eager-16k.txt", "manifest-eager-140k.txt"):
        p = os.path.join(_HERE, name)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        n = sum(1 for _ in open(p))
        lines.append(f"manifest {name} sha256={h} entries={n}")
    with open(os.path.join(out, "input_hashes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _verify_inputs(args):
    """Re-hash every primary-run selection file and check it against the vendored manifest (D15)."""
    import hashlib
    os.makedirs(args.out, exist_ok=True)
    _write_dev_and_hashes(args.out)
    lines = []
    for ctx, manifest in (("16k-eager", "manifest-eager-16k.txt"),
                          ("140k-eager", "manifest-eager-140k.txt")):
        root = os.path.join(CAPTURE_ROOT, ctx)
        entries = []
        for ln in open(os.path.join(_HERE, manifest)):
            sha, nbytes, rel = ln.split()
            if rel.startswith("selections/"):
                entries.append((sha, int(nbytes), rel))
        good = bad = missing = 0
        comb = hashlib.sha256()
        for sha, nbytes, rel in sorted(entries, key=lambda e: e[2]):
            p = os.path.join(root, rel)
            if not os.path.exists(p):
                missing += 1
                continue
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if h == sha:
                good += 1
            else:
                bad += 1
            comb.update(f"{sha}  {nbytes}  {rel}\n".encode())
        lines.append(f"{ctx}: selections files={len(entries)} matched={good} mismatched={bad} "
                     f"missing={missing} combined_sha256={comb.hexdigest()}")
        print(lines[-1], flush=True)
    with open(os.path.join(args.out, "input_hashes.txt"), "a") as f:
        f.write("\n--- selection-file verification (re-hash vs manifest) ---\n")
        f.write("\n".join(lines) + "\n")


def new_row():
    return {"n": 0, "inter": 0.0, "rank": 0.0, "se_num": 0.0, "se_n": 0}


def _run_fidelity_and_fixed(args, arms):
    out = args.out
    _write_dev_and_hashes(out)
    armset = ["A0re"] + arms
    for ctx in args.contexts.split(","):
        root = os.path.join(CAPTURE_ROOT, CONTEXTS.get(ctx, ctx))
        lmap = layer_map(root)
        caps = vc.capture_dirs(root)
        if args.only:
            subs = args.only.split(",")
            caps = [c for c in caps if any(s in c for s in subs)]
        if args.limit_captures:
            caps = caps[:args.limit_captures]
        fid = {"score_checked": 0, "max_abs": 0.0, "max_scaled": 0.0, "ids_checked": 0,
               "exact": 0, "explained": 0, "unexplained": 0, "dense": 0, "mismatches": [],
               "captures": 0, "cols_non_dense": 0, "cap_max_err": 0.0}
        percol = {a: {"inter": [], "rank": []} for a in armset}
        se = {a: [] for a in armset}
        colmeta = {"col": [], "layer": [], "step": [], "n": [], "scored": []}
        timing = {a: 0.0 for a in armset}
        rows = {}          # (arm, layer, step) -> row
        fid_caps = []      # per-capture fidelity records
        t0 = time.time()
        for i, capdir in enumerate(caps):
            meta = vc.read_meta(os.path.join(capdir, "meta.txt"))
            layer = lmap[int(meta["pooled_ptr"])]
            step = int(meta["step"])
            cap = Capture(capdir, args.max_columns)
            nd = cap.non_dense
            nd_pos = {int(c): j for j, c in enumerate(nd)}
            for a in armset:
                percol[a]["inter"].append(np.zeros(nd.size, np.int32))
                percol[a]["rank"].append(np.zeros(nd.size, np.int16))
                se[a].append(np.full(nd.size, np.nan, np.float32))
            colmeta["col"].append(nd.astype(np.int64))
            colmeta["layer"].append(np.full(nd.size, layer, np.int16))
            colmeta["step"].append(np.full(nd.size, step, np.int16))
            colmeta["n"].append(cap.n_per_col[nd].astype(np.int64))
            colmeta["scored"].append(np.array([cap.col_scored(int(c)) for c in nd], bool))

            c4 = a_block = None
            if any(a.startswith("A1_") for a in arms):
                c4, a_block = a1_prepare(cap.keys_f32)
            kq = {}
            for a in armset:
                if a == "A0re":
                    kq[a] = cap.keys_f32
                elif a.startswith("A1_"):
                    kq[a] = a1_reconstruct(c4, a_block, int(a.split("_")[1]))
                else:
                    kq[a] = a2_reconstruct(cap.keys_bits, int(a.split("_")[1]))

            tcap = time.perf_counter()
            mismatches_deferred = []
            cap_err_v = 0.0
            cf = {"sc": 0, "ma": 0.0, "ms": 0.0, "ids": 0, "ex": 0, "expl": 0, "unexpl": 0}
            for s in range(0, nd.size, 128):
                e = min(s + 128, nd.size)
                cols_chunk = nd[s:e]
                nsl = cap.n_per_col[cols_chunk]
                sel_engine = cap.engine_ids[:, cols_chunk]
                qc = cap.q[:, :, cols_chunk]
                S0 = score_all(cap.keys_f32, qc)
                a0sc = S0.copy()
                for j, col in enumerate(cols_chunk):
                    if cap.col_scored(col):
                        n = int(nsl[j])
                        a0sc[:n, j] = cap.captured_scores(col, n)
                a0rank = a0_top64(a0sc, nsl)
                ranked0, _ = ranked_top(S0, nsl)
                inter0 = rows_intersect(ranked0, sel_engine)
                rank0 = rows_intersect(ranked0[:64], a0rank)
                for j, col in enumerate(cols_chunk):
                    ii = s + j
                    n = int(nsl[j])
                    percol["A0re"]["inter"][-1][ii] = inter0[j]
                    percol["A0re"]["rank"][-1][ii] = rank0[j]
                    r = rows.setdefault(("A0re", layer, step), new_row())
                    r["n"] += 1
                    r["inter"] += inter0[j]
                    r["rank"] += rank0[j]
                    if cap.col_scored(col):
                        theirs = cap.captured_scores(col, n)
                        diff = np.abs(S0[:n, j] - theirs)
                        scale = max(float(theirs.max() - theirs.min()), 1e-30)
                        err = float(diff.max())
                        fid["max_abs"] = max(fid["max_abs"], err)
                        fid["max_scaled"] = max(fid["max_scaled"], float((diff / scale).max()))
                        fid["score_checked"] += 1
                        fid["cap_max_err"] = max(fid["cap_max_err"], err)
                        cap_err_v = max(cap_err_v, err)
                        cf["sc"] += 1
                        cf["ma"] = max(cf["ma"], err)
                        cf["ms"] = max(cf["ms"], float((diff / scale).max()))
                        last_err = err
                    else:
                        last_err = None
                    fid["ids_checked"] += 1
                    cf["ids"] += 1
                    if inter0[j] == TOP:
                        fid["exact"] += 1
                        cf["ex"] += 1
                    else:
                        eng = set(int(v) for v in sel_engine[:, j])
                        gs = (cap.captured_scores(col, n) if cap.col_scored(col) else S0[:n, j])
                        gap = vc.boundary_gap(gs, eng, n)
                        mismatches_deferred.append({
                            "capture": os.path.relpath(capdir, CAPTURE_ROOT), "col": int(col),
                            "gap": gap, "col_err": last_err,
                            "symmetric_difference": sorted(
                                set(int(v) for v in sel_engine[:, j]) ^
                                set(int(v) for v in ranked0[:TOP, j]))})
                for a in arms:
                    tt = time.perf_counter()
                    S = score_all(kq[a], qc)
                    ranked, _ = ranked_top(S, nsl)
                    inter = rows_intersect(ranked, sel_engine)
                    rank_inter = rows_intersect(ranked[:64], a0rank)
                    for j, col in enumerate(cols_chunk):
                        ii = s + j
                        n = int(nsl[j])
                        percol[a]["inter"][-1][ii] = inter[j]
                        percol[a]["rank"][-1][ii] = rank_inter[j]
                        r = rows.setdefault((a, layer, step), new_row())
                        r["n"] += 1
                        r["inter"] += inter[j]
                        r["rank"] += rank_inter[j]
                        if cap.col_scored(col):
                            theirs = cap.captured_scores(col, n)
                            diff = np.abs(S[:n, j] - theirs)
                            scale = max(float(theirs.max() - theirs.min()), 1e-30)
                            val = float(diff.sum()) / (n * scale)
                            se[a][-1][ii] = val
                            r["se_num"] += val
                            r["se_n"] += 1
                    timing[a] += time.perf_counter() - tt
            # classify deferred mismatches now that the capture's max error is known
            for m in mismatches_deferred:
                ebound = m["col_err"] if m["col_err"] is not None else cap_err_v
            # classify deferred mismatches now that the capture's max error is known
            for m in mismatches_deferred:
                ebound = m["col_err"] if m["col_err"] is not None else cap_err_v
                m["error"] = ebound
                m["verdict"] = "explained" if m["gap"] < ebound else "unexplained"
                fid[m["verdict"]] += 1
                cf["expl" if m["verdict"] == "explained" else "unexpl"] += 1
                fid["mismatches"].append(m)
            cf["dense"] = int(cap.dense.sum())
            cf["capture"] = os.path.relpath(capdir, CAPTURE_ROOT)
            fid_caps.append(cf)
            timing["A0re"] += time.perf_counter() - tcap
            fid["dense"] += int(cap.dense.sum())
            fid["captures"] += 1
            fid["cols_non_dense"] += int(nd.size)
            if (i + 1) % 20 == 0 or i + 1 == len(caps):
                print(f"[{ctx}] {i+1}/{len(caps)}  {time.time()-t0:.0f}s", flush=True)

        _dump_stage(out, ctx, "fixed", armset, percol, se, colmeta, timing)
        with open(os.path.join(out, f"rows_{ctx}_fixed.tsv"), "w") as f:
            f.write("arm\tlayer\tstep\tcolumns\tinter_sum\trank_sum\tse_sum\tse_n\n")
            for (a, layer, step), r in sorted(rows.items()):
                f.write(f"{a}\t{layer}\t{step}\t{r['n']}\t{r['inter']}\t{r['rank']}\t"
                        f"{r['se_num']:.8g}\t{r['se_n']}\n")
        with open(os.path.join(out, f"fidelity_{ctx}.json"), "w") as f:
            json.dump({k: v for k, v in fid.items() if k != "mismatches"}, f, indent=2)
        with open(os.path.join(out, f"fidelity_mismatches_{ctx}.json"), "w") as f:
            json.dump(fid["mismatches"], f, indent=2)
        with open(os.path.join(out, f"fidelity_captures_{ctx}.tsv"), "w") as f:
            f.write("capture\tscored\tmax_abs\tmax_scaled\tids\texact\texplained\tunexplained\tdense\n")
            for cf in fid_caps:
                f.write(f"{cf['capture']}\t{cf['sc']}\t{cf['ma']:.3e}\t{cf['ms']:.3e}\t"
                        f"{cf['ids']}\t{cf['ex']}\t{cf['expl']}\t{cf['unexpl']}\t{cf['dense']}\n")
        print(f"[{ctx}] stage done {time.time()-t0:.0f}s  fid max_abs={fid['max_abs']:.3e} "
              f"scaled={fid['max_scaled']:.3e} ids={fid['ids_checked']} expl={fid['explained']} "
              f"unexpl={fid['unexplained']}", flush=True)


def _dump_stage(out, ctx, stage, arms, percol, se, colmeta, timing):
    np.savez_compressed(
        os.path.join(out, f"per_column_{ctx}_{stage}.npz"),
        col=np.concatenate(colmeta["col"]), layer=np.concatenate(colmeta["layer"]),
        step=np.concatenate(colmeta["step"]), n=np.concatenate(colmeta["n"]),
        scored=np.concatenate(colmeta["scored"]),
        **{f"{a}_inter": np.concatenate(percol[a]["inter"]) for a in arms},
        **{f"{a}_rank": np.concatenate(percol[a]["rank"]) for a in arms},
        **{f"se_{a}": np.concatenate(se[a]) for a in arms})
    with open(os.path.join(out, f"timing_{ctx}_{stage}.json"), "w") as f:
        json.dump(timing, f, indent=2)


# ------------------------------------------------------------------------------------------------
# A3: water-filling depths (D5/D6/D8)
# ------------------------------------------------------------------------------------------------
def depths_from_theta(g, theta):
    lg = np.where(g > 0, np.log2(np.maximum(g, 1e-45)), -40.0)
    return np.clip(np.round((lg - np.log2(theta)) / 2.0), 0, 4).astype(np.int8)


def build_bit_hist(root, arm, sample=64):
    xs, ws = [], []
    total_blocks = 0.0
    for capdir in vc.capture_dirs(root):
        cap = Capture(capdir)
        if cap.non_dense.size == 0:
            continue
        idx = cap.non_dense[:sample]
        if arm == "A3_V1":
            var_j = cap.keys_f32.var(axis=0)
            for col in idx:
                n = int(cap.n_per_col[col])
                if n <= 0:
                    continue
                w = (cap.q[:, :, col] ** 2).sum(axis=0)
                g0 = w * var_j
                x = np.where(g0 > 0, np.log2(np.maximum(g0, 1e-45)), -40.0)
                xs.append(x)
                ws.append(np.full(NUM_CHANNELS, float(n)))
                total_blocks += n
        else:
            c4, a_block = a1_prepare(cap.keys_f32)
            for col in idx:
                n = int(cap.n_per_col[col])
                if n <= 0:
                    continue
                w = (cap.q[:, :, col] ** 2).sum(axis=0)
                gg = np.arange(n) // SCALE_GROUP_BLOCKS
                G = int(gg.max()) + 1
                a_g = np.zeros((G, NUM_CHANNELS), np.float32)
                a_g[gg] = a_block[:n]
                counts = np.bincount(gg, minlength=G).astype(np.float64)
                g0 = w[None, :] * (a_g * a_g)
                x = np.where(g0 > 0, np.log2(np.maximum(g0, 1e-45)), -40.0)
                xs.append(x.ravel())
                ws.append(np.repeat(counts, NUM_CHANNELS))
                total_blocks += n
    X = np.concatenate(xs) if xs else np.array([0.0])
    W = np.concatenate(ws).astype(np.float64) if ws else np.array([1.0])
    edges = np.linspace(-40.0, 40.0, 2001)
    h, _ = np.histogram(X, bins=edges, weights=W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h, total_blocks


def hist_mean_bits(centers, h, lth):
    # mean code bits/block = 128 channels * mean channel depth; total = code bits + 128 scale bits
    t = np.clip(np.round((centers - lth) / 2.0), 0, 4)
    mean_t = float((h * t).sum() / max(h.sum(), 1.0))
    return 128.0 * (1.0 + mean_t)


def solve_theta(centers, h, target_bits):
    lo, hi = -40.0, 40.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if hist_mean_bits(centers, h, m) > target_bits:
            lo = m
        else:
            hi = m
    return 2.0 ** (0.5 * (lo + hi))


def a4_from_fixed(out, ctx):
    per_layer = {}
    with open(os.path.join(out, f"rows_{ctx}_fixed.tsv")) as f:
        next(f)
        for ln in f:
            a, layer, step, n, inter, rank, se, sen = ln.split("\t")
            if a.startswith("A1_"):
                per_layer.setdefault(a, {}).setdefault(int(layer), [0.0, 0.0])
                per_layer[a][int(layer)][0] += float(inter)
                per_layer[a][int(layer)][1] += float(n)
    chosen = None
    curve = {}
    for t in (1, 2, 3, 4):
        a = f"A1_{t}"
        means = [v[0] / (TOP * v[1]) for v in per_layer[a].values() if v[1] > 0]
        overall = (sum(v[0] for v in per_layer[a].values())
                   / (TOP * sum(v[1] for v in per_layer[a].values())))
        worst = min(means)
        spread = max(means) - min(means)
        curve[a] = {"mean": overall, "worst": worst, "spread": spread}
        if chosen is None and overall >= BAR_MEAN and worst >= BAR_LAYER and spread <= BAR_SPREAD:
            chosen = t
    return (chosen if chosen is not None else 4), curve


def _run_a3(args):
    _write_dev_and_hashes(args.out)
    for ctx in args.contexts.split(","):
        root = os.path.join(CAPTURE_ROOT, CONTEXTS.get(ctx, ctx))
        lmap = layer_map(root)
        t_star, a1curve = a4_from_fixed(args.out, ctx)
        target_bits = 128.0 * (t_star + 1)
        caps = vc.capture_dirs(root)
        if args.only:
            subs = args.only.split(",")
            caps = [c for c in caps if any(s in c for s in subs)]
        if args.limit_captures:
            caps = caps[:args.limit_captures]
        print(f"[{ctx}] A4=A1({t_star}) target_bits={target_bits:.0f}", flush=True)
        probes = {}
        for arm in A3_ARMS:
            centers, h, _ = build_bit_hist(root, arm, sample=64)
            theta = solve_theta(centers, h, target_bits)
            probes[arm] = {"theta": theta,
                           "probe_bits": hist_mean_bits(centers, h, np.log2(theta))}
            print(f"[{ctx}] {arm} theta={theta:.6g} probe_bits={probes[arm]['probe_bits']:.2f}",
                  flush=True)
        percol = {a: {"inter": [], "rank": []} for a in A3_ARMS}
        se = {a: [] for a in A3_ARMS}
        colmeta = {"col": [], "layer": [], "step": [], "n": [], "scored": []}
        rows = {}
        bits = {a: {"sum_bits": 0.0, "blocks": 0.0, "maxB": 0} for a in A3_ARMS}
        timing = {a: 0.0 for a in A3_ARMS}
        # A1(4) per-column metrics, to reuse exactly when the water line is all-4 on every
        # channel with a non-zero query component (then A3's scores equal A1(4)'s identically).
        fz = np.load(os.path.join(args.out, f"per_column_{ctx}_fixed.npz"))
        a14 = {"inter": fz["A1_4_inter"], "rank": fz["A1_4_rank"], "se": fz["se_A1_4"]}
        gidx = {a: 0 for a in A3_ARMS}
        reused = {a: 0 for a in A3_ARMS}
        t0 = time.time()
        for i, capdir in enumerate(caps):
            meta = vc.read_meta(os.path.join(capdir, "meta.txt"))
            layer = lmap[int(meta["pooled_ptr"])]
            step = int(meta["step"])
            cap = Capture(capdir, args.max_columns)
            nd = cap.non_dense
            for a in A3_ARMS:
                percol[a]["inter"].append(np.zeros(nd.size, np.int32))
                percol[a]["rank"].append(np.zeros(nd.size, np.int16))
                se[a].append(np.full(nd.size, np.nan, np.float32))
            colmeta["col"].append(nd.astype(np.int64))
            colmeta["layer"].append(np.full(nd.size, layer, np.int16))
            colmeta["step"].append(np.full(nd.size, step, np.int16))
            colmeta["n"].append(cap.n_per_col[nd].astype(np.int64))
            colmeta["scored"].append(np.array([cap.col_scored(int(c)) for c in nd], bool))
            c4, a_block = a1_prepare(cap.keys_f32)
            var_j = cap.keys_f32.var(axis=0)
            # per-depth reconstructed keys, shared by every column (the depth only selects rows)
            Rstack = np.stack([_recon(c4, a_block, v) for v in range(5)], axis=0)  # [5,n,128]
            for ai, arm in enumerate(A3_ARMS):
                tt = time.perf_counter()
                log2theta = np.log2(probes[arm]["theta"])
                for ci, col in enumerate(nd):
                    col = int(col)
                    n = int(cap.n_per_col[col])
                    if n <= 0:
                        gidx[arm] += 1
                        continue
                    w = (cap.q[:, :, col] ** 2).sum(axis=0)
                    wpos = w > 0
                    t4_safe = False
                    if arm == "A3_V1":
                        g = w * var_j
                        lg = np.where(g > 0, np.log2(np.maximum(g, 1e-45)), -40.0)
                        t = np.clip(np.round((lg - log2theta) / 2.0), 0, 4).astype(np.int8)
                        bsum = int(t.sum())
                        block_bits = (bsum + 128.0) * n
                        t4_safe = bool(np.all(t[wpos] == 4)) if wpos.any() else True
                    else:
                        gg = np.arange(n) // SCALE_GROUP_BLOCKS
                        G = int(gg.max()) + 1
                        a_g = np.zeros((G, NUM_CHANNELS), np.float32)
                        a_g[gg] = a_block[:n]
                        g = w[None, :] * (a_g * a_g)
                        t_g = depths_from_theta(g, 2.0 ** log2theta)
                        counts = np.bincount(gg, minlength=G)
                        bsum = int((t_g.sum(axis=1) * counts).sum())
                        block_bits = bsum + 128.0 * n
                        t = t_g[gg]
                        t4_safe = bool(np.all(t[:, wpos] == 4)) if wpos.any() else True
                    if t4_safe:
                        inter = int(a14["inter"][gidx[arm]])
                        rank_inter = int(a14["rank"][gidx[arm]])
                        seval = a14["se"][gidx[arm]]
                        sen = None if np.isnan(seval) else float(seval)
                        reused[arm] += 1
                    else:
                        if arm == "A3_V1":
                            dots = np.zeros((n, NUM_HEADS), np.float32)
                            for v in range(5):
                                mask = t == v
                                if not mask.any():
                                    continue
                                A = Rstack[v][:n] if mask.all() else Rstack[v][:n][:, mask]
                                dots += A @ cap.q[:, mask, col].T
                            S = np.maximum(dots, 0.0).sum(axis=1) / float(np.sqrt(NUM_CHANNELS))
                        else:
                            kq = np.take_along_axis(Rstack[:, :n, :], t[None], axis=0)[0]
                            S = score_one(kq, cap.q, col)
                        ranked = np.argsort(-S, kind="stable")[:TOP]
                        eng = cap.engine_ids[:, col]
                        inter = np.intersect1d(ranked, eng, assume_unique=False).size
                        if cap.col_scored(col):
                            theirs = cap.captured_scores(col, n)
                            diff = np.abs(S - theirs)
                            scale = max(float(theirs.max() - theirs.min()), 1e-30)
                            sen = float(diff.sum()) / (n * scale)
                            a0rank = np.argsort(-theirs, kind="stable")[:64]
                        else:
                            S0 = score_one(cap.keys_f32[:n], cap.q, col)
                            a0rank = np.argsort(-S0, kind="stable")[:64]
                            sen = None
                        rank_inter = np.intersect1d(ranked[:64], a0rank, assume_unique=False).size
                    if sen is not None:
                        se[arm][-1][ci] = sen
                    percol[arm]["inter"][-1][ci] = inter
                    percol[arm]["rank"][-1][ci] = rank_inter
                    r = rows.setdefault((arm, layer, step), new_row())
                    r["n"] += 1
                    r["inter"] += inter
                    r["rank"] += rank_inter
                    if sen is not None:
                        r["se_num"] += sen
                        r["se_n"] += 1
                    bits[arm]["sum_bits"] += block_bits
                    bits[arm]["blocks"] += n
                    bits[arm]["maxB"] = max(bits[arm]["maxB"], bsum)
                    gidx[arm] += 1
                timing[arm] += time.perf_counter() - tt
            if (i + 1) % 10 == 0 or i + 1 == len(caps):
                print(f"[{ctx}] {i+1}/{len(caps)}  {time.time()-t0:.0f}s", flush=True)
        _dump_stage(args.out, ctx, "a3", A3_ARMS, percol, se, colmeta, timing)
        for a in A3_ARMS:
            if gidx[a] != a14["inter"].size:
                raise RuntimeError(f"A3/fixed column misalignment {a}: {gidx[a]} vs {a14['inter'].size}")
        print(f"[{ctx}] A3 reused-A1(4) columns: "
              + ", ".join(f"{a}={reused[a]}/{gidx[a]}" for a in A3_ARMS), flush=True)
        with open(os.path.join(args.out, f"rows_{ctx}_a3.tsv"), "w") as f:
            f.write("arm\tlayer\tstep\tcolumns\tinter_sum\trank_sum\tse_sum\tse_n\n")
            for (a, layer, step), r in sorted(rows.items()):
                f.write(f"{a}\t{layer}\t{step}\t{r['n']}\t{r['inter']}\t{r['rank']}\t"
                        f"{r['se_num']:.8g}\t{r['se_n']}\n")
        with open(os.path.join(args.out, f"a3_budgets_{ctx}.json"), "w") as f:
            json.dump({"A4": f"A1_{t_star}", "target_bits": target_bits, "a1_curve": a1curve,
                       "probes": {a: {"theta": probes[a]["theta"],
                                      "probe_bits": probes[a]["probe_bits"],
                                      "realized_mean_bits": bits[a]["sum_bits"] / bits[a]["blocks"],
                                      "realized_maxB": bits[a]["maxB"]}
                                  for a in A3_ARMS}}, f, indent=2)
        print(f"[{ctx}] a3 done {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
ARM_BYTES = {"A0re": 256, "A1_1": 32, "A1_2": 48, "A1_3": 64, "A1_4": 80,
             "A2_8": 128, "A2_9": 144, "A2_10": 160, "A2_12": 192, "A2_16": 256}


def _read_rows(path):
    per_layer = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        next(f)
        for ln in f:
            a, layer, step, n, inter, rank, se, sen = ln.split("\t")
            d = per_layer.setdefault((a, int(layer)), [0.0, 0.0, 0.0, 0.0, 0.0])
            d[0] += float(inter); d[1] += float(n); d[2] += float(rank)
            d[3] += float(se); d[4] += float(sen)
    return per_layer


def _arm_layer_stats(per_layer, arm):
    rows = {layer: v for (a, layer), v in per_layer.items() if a == arm}
    stats = {}
    for layer, v in rows.items():
        n = v[1]
        stats[layer] = {"n": n, "agree": v[0] / (TOP * n) if n else 0.0,
                        "rank": v[2] / (64 * n) if n else 0.0,
                        "jaccard": v[0] / (1024 * n - v[0]) if v[0] < 1024 * n else 0.0,
                        "se": v[3] / v[4] if v[4] else 0.0}
    tot = sum(v["n"] for v in stats.values())
    mean = sum(v["agree"] * v["n"] for v in stats.values()) / tot if tot else 0.0
    return stats, mean


def _report(args):
    lines = []
    for ctx in args.contexts.split(","):
        per_layer = _read_rows(os.path.join(args.out, f"rows_{ctx}_fixed.tsv"))
        a3_layer = _read_rows(os.path.join(args.out, f"rows_{ctx}_a3.tsv"))
        budget_path = os.path.join(args.out, f"a3_budgets_{ctx}.json")
        ab = json.load(open(budget_path)) if os.path.exists(budget_path) else None
        lines.append(f"\n================ {ctx} ================")
        lines.append(f"{'arm':<7}{'mean':>9}{'worst':>9}{'spread':>9}{'B/blk':>8}{'reduc':>8}")
        for arm in ["A0re"] + A1_ARMS + A2_ARMS + A3_ARMS:
            src = a3_layer if arm in A3_ARMS else per_layer
            if src is None:
                continue
            stats, mean = _arm_layer_stats(src, arm)
            if not stats:
                continue
            worst = min(v["agree"] for v in stats.values())
            spread = max(v["agree"] for v in stats.values()) - worst
            if arm in A3_ARMS:
                b = ab["probes"][arm]["realized_mean_bits"] / 8.0 if ab else float("nan")
            else:
                b = ARM_BYTES.get(arm, float("nan"))
            red = 256.0 / b if b else float("nan")
            lines.append(f"{arm:<7}{mean:>9.4f}{worst:>9.4f}{spread:>9.4f}{b:>8.2f}{red:>7.2f}x")
        if per_layer is not None:
            stats, mean = _arm_layer_stats(per_layer, "A1_4")
            if stats:
                worst = min(v["agree"] for v in stats.values())
                spread = max(v["agree"] for v in stats.values()) - worst
                c1, c2, c3 = mean >= BAR_MEAN, worst >= BAR_LAYER, spread <= BAR_SPREAD
                lines.append(f"BAR A1(4): mean {mean:.4f} {'PASS' if c1 else 'FAIL'} | "
                             f"worst {worst:.4f} {'PASS' if c2 else 'FAIL'} | "
                             f"spread {spread:.4f} {'PASS' if c3 else 'FAIL'} | "
                             f"curve reported PASS | 80 vs 256 = 3.2x")
                # weighting robustness: layer-averaged and step-averaged means
                lc, ls = {}, {}
                with open(os.path.join(args.out, f"rows_{ctx}_fixed.tsv")) as rf:
                    next(rf)
                    for ln in rf:
                        a, layer, step, n, inter, rank, se, sen = ln.split("\t")
                        if a != "A1_4" or float(n) == 0:
                            continue
                        lc.setdefault(int(layer), [0.0, 0.0])
                        lc[int(layer)][0] += float(inter); lc[int(layer)][1] += float(n)
                        ls.setdefault(int(layer), []).append(float(inter) / (512.0 * float(n)))
                laymean = sum(v[0] / (512 * v[1]) for v in lc.values()) / len(lc)
                stepmean = sum(sum(v) / len(v) for v in ls.values()) / len(ls)
                lines.append(f"A1(4) weighting robustness: column-weighted {mean:.4f} | "
                             f"layer-averaged {laymean:.4f} | step-averaged {stepmean:.4f}")
        if ab is not None and a3_layer is not None and per_layer is not None:
            t_star = int(ab["A4"].split("_")[1])
            _, a4_mean = _arm_layer_stats(per_layer, ab["A4"])
            a4_bits = 128.0 * (t_star + 1)
            for v in A3_ARMS:
                st, mn = _arm_layer_stats(a3_layer, v)
                if not st:
                    continue
                vbits = ab["probes"][v]["realized_mean_bits"]
                equal = abs(vbits - a4_bits) <= A3_BITS_TOL * a4_bits
                gates = bool(equal and mn >= a4_mean + A3_GATE_MARGIN)
                lines.append(f"A3 {v}: mean {mn:.4f} vs {ab['A4']} {a4_mean:.4f} "
                             f"(delta {mn-a4_mean:+.4f}); bits {vbits:.1f} vs {a4_bits:.1f} "
                             f"({'equal' if equal else 'NOT equal'}); gate {'PASS' if gates else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(text + "\n")
    # per-layer tables (the rows of section 3), all arms, raw enough to re-derive every mean
    for ctx in args.contexts.split(","):
        per_layer = _read_rows(os.path.join(args.out, f"rows_{ctx}_fixed.tsv")) or {}
        a3_layer = _read_rows(os.path.join(args.out, f"rows_{ctx}_a3.tsv")) or {}
        allrows = dict(per_layer)
        allrows.update(a3_layer)
        with open(os.path.join(args.out, f"per_layer_{ctx}.tsv"), "w") as f:
            f.write("arm\tlayer\tcolumns\tagreement\tjaccard\trank64\tscore_error\n")
            for arm in ["A0re"] + A1_ARMS + A2_ARMS + A3_ARMS:
                stats, _ = _arm_layer_stats(allrows, arm)
                for layer in sorted(stats):
                    s = stats[layer]
                    f.write(f"{arm}\t{layer}\t{int(s['n'])}\t{s['agree']:.6f}\t{s['jaccard']:.6f}\t"
                            f"{s['rank']:.6f}\t{s['se']:.6e}\n")


def main():
    ap = argparse.ArgumentParser(description="QSA index truncation prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fidelity", "arms", "a3", "report", "verify-inputs"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--contexts", default="16k-eager,140k-eager")
        p.add_argument("--limit-captures", type=int, default=0)
        p.add_argument("--max-columns", type=int, default=0)
        p.add_argument("--arms", default="A1,A2")
        p.add_argument("--budgets", default="")
        p.add_argument("--only", default="")
    args = ap.parse_args()
    if args.max_columns == 0:
        args.max_columns = None
    if args.cmd == "fidelity":
        _run_fidelity_and_fixed(args, [])
    elif args.cmd == "arms":
        arms = []
        for grp in args.arms.split(","):
            if grp == "A1":
                arms += A1_ARMS
            elif grp == "A2":
                arms += A2_ARMS
        _run_fidelity_and_fixed(args, arms)
    elif args.cmd == "a3":
        _run_a3(args)
    elif args.cmd == "verify-inputs":
        _verify_inputs(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
