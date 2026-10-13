#!/usr/bin/env python3
"""Deterministic capture prompt generator (ticket #40, branch proto/qsa-index-capture).

Usage: python3 make_prompt.py LINES > prompt.txt

Every line is a numbered record about the QSA index so the tokenizer sees varied,
non-degenerate text. The committed prompt files are regenerated exactly with the line
counts recorded in prompt_lines.txt.
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

TEMPLATE = (
    "Record {k:06d}: at absolute position {p} the indexer pools the four raw keys {p}, {p1}, "
    "{p2} and {p3} into one 128-wide BF16 key, normalizes it with the unit-offset RMSNorm "
    "weight, rotates it by the block's first-token RoPE position, and scores it against the "
    "column query as one over the square root of 128 times the sum of the four ReLU head dot "
    "products. The selection keeps the 512 highest block scores, ties to the lower block id, "
    "and the open block after the last complete one stays attended in full."
)

out = []
for k in range(N):
    p = k * 4
    out.append(TEMPLATE.format(k=k, p=p, p1=p + 1, p2=p + 2, p3=p + 3))
sys.stdout.write("\n".join(out) + "\n")
