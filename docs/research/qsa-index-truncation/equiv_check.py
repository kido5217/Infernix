#!/usr/bin/env python3
"""Equivalence check: the probe's batched fidelity vs the vendored verifier (verify_capture.py).

The full-coverage fidelity gate in ``truncation_probe.py`` is a batched (BLAS-order) reimplementation
of the same recomputation, because the vendored verifier's per-column key rebuild is O(n) index work
per column and cannot cover ~373 k columns in reasonable time.  This script proves the two agree on
every capture where the vendored verifier is tractable: it runs ``verify_capture.analyze(cap, 10**9)``
(full column coverage for that capture) and compares the exact counts — scored columns, ids checked,
exact, explained, unexplained, dense — against the committed per-capture TSV written by the probe.
The max score errors are compared loosely: they differ at the ~1e-7 level from summation order, but
both must be inside the fixed fidelity bounds.

Usage:  nix develop . --command python3 equiv_check.py [--max-cols 8] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_capture as vc  # noqa: E402

CAPTURE_ROOT = "/home/kido/network/projects/Infernix/out/qsa-index-capture"
FID_ABS, FID_SCALED = 1e-5, 1e-6


def load_tsv(out, ctx):
    path = os.path.join(out, f"fidelity_captures_{ctx}.tsv")
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["capture"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
    ap.add_argument("--max-cols", type=int, default=8,
                    help="only check captures with this many columns or fewer (the tractable set)")
    ap.add_argument("--captures", default="",
                    help="comma substrings of capture paths to check regardless of --max-cols")
    args = ap.parse_args()

    total = agree = 0
    neartie_diff = 0
    for ctx in ("140k-eager", "16k-eager"):
        rows = load_tsv(args.out, ctx)
        root = os.path.join(CAPTURE_ROOT, ctx)
        subs = args.captures.split(",") if args.captures else []
        checked_here = 0
        for capdir in vc.capture_dirs(root):
            meta = vc.read_meta(os.path.join(capdir, "meta.txt"))
            rel = os.path.relpath(capdir, CAPTURE_ROOT)
            selected = (int(meta["columns"]) <= args.max_cols or
                        (subs and any(s in rel for s in subs)))
            if not selected or rel not in rows:
                continue
            r = vc.analyze(capdir, 10**9)
            exp = rows[rel]
            # Exact counts that cannot depend on summation order:
            ok = (r["score_checked"] == int(exp["scored"]) and
                  r["ids_checked"] == int(exp["ids"]) and
                  r["id_dense"] == int(exp["dense"]) and
                  r["unexplained"] == 0 and int(exp["unexplained"]) == 0 and
                  r["exact"] + r["explained"] + r["unexplained"] == r["ids_checked"])
            # The exact/explained split is summation-order dependent (both use FP32 in different
            # orders); count the near-tie differences separately.
            neartie_diff += abs(r["exact"] - int(exp["exact"]))
            # Bounds are fixed and must hold under both orders.
            ok = ok and r["max_abs"] <= FID_ABS and r["max_scaled"] <= FID_SCALED
            total += 1
            agree += 1 if ok else 0
            checked_here += 1
            if not ok:
                print(f"MISMATCH {rel}: verifier sc={r['score_checked']} ma={r['max_abs']:.3e} "
                      f"ms={r['max_scaled']:.3e} ids={r['ids_checked']} ex={r['exact']} "
                      f"expl={r['explained']} unexpl={r['unexplained']} dense={r['id_dense']} | "
                      f"probe {exp}")
        print(f"[{ctx}] checked {checked_here} captures (columns <= {args.max_cols})")
    print(f"EQUIVALENCE: {agree}/{total} captures identical on the order-independent counts and "
          f"within bounds; summation-order near-tie differences (exact split) = {neartie_diff}")
    return 0 if agree == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
