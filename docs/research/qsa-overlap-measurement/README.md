# Cross-query block-selection overlap on the captured QSA prefill selections

Measurements for the map ticket "cross-query KV dedup in the QSA prefill" (issue #45) and its task
ticket **#48**. CPU only, read-only over the capture branch `proto/qsa-index-capture` (head
`05f74f1a`, data at `out/qsa-index-capture/`, gitignored). No selection was re-run; no capture file was
written.

The result is what the prototype gate depends on: **the realized dedup ratio is 0.9980 at 16K and
0.9844 at 140K**, i.e. far above the issue's ~30 % bar; the per-chunk spread across the 12 layers
(max − min) is at most 0.0046 at 16K and 0.0173 at 140K.

## Metric (definition used here)

For a prefill chunk with non-dense query columns `q` and per-query selected sets `S_q` (`|S_q| = top =
512` block ids, each block = `ratio` = 4 tokens):

```
R_chunk = 1 - |union_q S_q| / sum_q |S_q|
```

This is the fraction of the chunk's block-gather work that disappears if the chunk gathers each
distinct block once instead of once per query. Summed over a context it is `1 - sum(distinct) /
sum(sum_blocks)`.

Dense columns (`counts = -1`, where the column's complete-block count is `<= top`) carry no selection
and are excluded from both the union and the sum (they are counted in the raw output).

**This is not the paper's metric.** Ticket #47 reads QUILT (arXiv 2610.11134) and will define the
overlap metric from it; nothing here waits on that. The definition above is the redundancy
`1 − distinct/sum` the issue asks for. A second, distance-resolved statistic is reported alongside it:
the mean per-pair overlap `mean_q |S_q ∩ S_{q+d}| / 512` and the mean Jaccard
`mean_q |S_q ∩ S_{q+d}| / |S_q ∪ S_{q+d}|` at position distance `d`. Both are stated so a later
reconciliation with the paper's definition is mechanical.

## Inputs, and which runs were used

| capture | 16K | 140K |
|---|---|---|
| prefill chunks | 6 (`4096 × 5`, `1946`) | 3 (`4096 × 2`, `2432`) |
| layers (distinct `pooled_ptr`) | 12 | 12 |
| chunk × layer captures measured | 72 | 36 |
| positions | 0…22425 | 131072…141695 |
| dense columns excluded | 2051 in chunk 0 (all others 0) | 0 |

Runs used: **`16k-eager` and `140k-eager` as primary** (the ticket's default), because they are the
eager runs and cover all layers at both contexts. They are not a different data set: `verify_runs.py`
shows the prefill `selected.bin` **and** `counts.bin` are byte-identical across the three runs per
context — 72/72 identical among `16k`, `16k-eager`, `scores-16k`, and 36/36 among `140k`,
`140k-eager`, `scores-140k` (`results/run_consistency.txt`). So the measurement is run-independent; using
the eager runs costs nothing and matches the ticket.

A prefill chunk capture is identified as a selection capture with `columns >= 1000` (the decode,
verify and MTP calls in the same runs have `columns <= 13`); layers are keyed by `pooled_ptr` and
chunks ordered by `min_position`. Byte layouts are exactly `verify_capture.py`'s conventions
(`selected.bin` = I32 `[top, columns]`, column `c` at `selected[c*top + i]`; `counts.bin`; `positions.bin`).

## Commands

```
nix develop . --command python3 docs/research/qsa-overlap-measurement/measure_overlap.py
nix develop . --command python3 docs/research/qsa-overlap-measurement/verify_runs.py
```

`measure_overlap.py` writes to `results/`; the raw numbers are re-derivable from the four files:

| file | contents |
|---|---|
| `results/per_chunk_layer.tsv` | one row per chunk × layer: columns, dense/valid columns, sum, distinct, ratio, blocks saved, bytes saved, singleton stats |
| `results/selection_count_histogram.tsv` | long format `(selection_count, n_blocks)` per chunk × layer: how many blocks are selected by exactly `k` queries |
| `results/distance_curve.tsv` | per chunk × layer × `d` × scope (`all`/`same_chunk`/`cross_chunk`): n_pairs, mean intersection, mean overlap, mean Jaccard |
| `results/summary.json` | per-context aggregates, provenance, row-profile constants |

## 1. Realized dedup ratio

Overall: **16K `R = 0.99799`** (per-layer mean 0.9978, range 0.9944–0.9990, std 0.0016);
**140K `R = 0.98444`** (per-layer mean 0.9836, range 0.9725–0.9898, std 0.0045). Both are far above the
issue's ~30 % threshold, and the **layer dependence is small** — the layer-to-layer spread is under
0.5 point at 16K and under 1.8 points at 140K.

Per chunk (mean over the 12 layers; `sum = valid_columns × 512`):

**16K** (positions 0…22425)

| chunk | positions | valid cols | dense | sum | distinct (mean) | ratio (mean) | ratio [min,max] | bytes saved INT8 | bytes saved BF16 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0–4095 | 2045 | 2051 | 1,047,040 | 1,024 | 0.99902 | [0.99902, 0.99902] | 53.0 GB | 102.8 GB |
| 1 | 4096–8191 | 4096 | 0 | 2,097,152 | 2,048 | 0.99902 | [0.99902, 0.99902] | 106.2 GB | 206.0 GB |
| 2 | 8192–12287 | 4096 | 0 | 2,097,152 | 3,072 | 0.99854 | [0.99853, 0.99853] | 106.1 GB | 205.9 GB |
| 3 | 12288–16383 | 4096 | 0 | 2,097,152 | 4,094 | 0.99805 | [0.99805, 0.99805] | 106.1 GB | 205.8 GB |
| 4 | 16384–20479 | 4096 | 0 | 2,097,152 | 5,113 | 0.99756 | [0.99756, 0.99759] | 106.0 GB | 205.7 GB |
| 5 | 20480–22425 | 1946 | 0 | 996,352 | 5,582 | 0.99440 | [0.99437, 0.99457] | 50.2 GB | 97.4 GB |

**140K** (positions 131072…141695)

| chunk | positions | valid cols | dense | sum | distinct (mean) | ratio (mean) | ratio [min,max] | bytes saved INT8 | bytes saved BF16 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 131072–135167 | 4096 | 0 | 2,097,152 | 28,196 | 0.98656 | [0.98410, 0.98982] | 104.9 GB | 203.4 GB |
| 1 | 135168–139263 | 4096 | 0 | 2,097,152 | 28,964 | 0.98619 | [0.98357, 0.98955] | 104.8 GB | 203.3 GB |
| 2 | 139264–141695 | 2432 | 0 | 1,245,184 | 27,493 | 0.97792 | [0.97248, 0.98370] | 61.7 GB | 119.7 GB |

**Absolute terms.** The union is nearly the whole available block pool at every chunk. At 16K the
union equals the chunk's `complete_blocks` (chunk 1: distinct 2,048 = complete 2,048; chunk 4:
5,113 vs 5,120; chunk 5: 5,582 vs 5,606) because 4,096 queries drawing 512 blocks each from a pool of
at most 5,610 blocks saturate it. At 140K the union covers ~83 % of the pool (28,196 vs 33,792;
28,964 vs 34,816; 27,493 vs 35,424).

**Bytes.** The saving is `(sum − distinct) × ratio(4 tokens/block) × kv_heads(2) × row_bytes`, i.e. the
block-gather volume removed per layer per chunk (both KV heads, K and V rows). Row sizes: **INT8-group64
528 B/token/kv-head** (`qsa_prompt.cu:44-68`, facts-note §2) and **BF16 1024 B**. The capture itself ran
`--kv-dtype nvfp4` (288 B/token/kv-head); the column is in the raw TSV. Per full prompt (all chunks ×
12 layers), the gather is 528.8 GB (INT8) / 1025.5 GB (BF16) at 16K and 275.7 GB / 534.7 GB at 140K;
dedup removes 527.7 GB / 1023.5 GB and 271.4 GB / 526.4 GB respectively — i.e. **99.80 % (16K)** and
**98.44 % (140K)** of the gather volume.

## 2. Structure of the overlap

**Per-distance overlap** (mean over the 12 layers; `[min,max]` across layers; `d` = absolute position
difference):

**16K**

| d | mean overlap | per-layer [min,max] | mean Jaccard | cross-chunk overlap |
|---|---|---|---|---|
| 1 | 0.6364 | [0.5260, 0.8041] | 0.4980 | 0.6794 |
| 2 | 0.5922 | [0.4937, 0.7398] | 0.4488 | 0.6446 |
| 4 | 0.5390 | [0.4403, 0.6610] | 0.3960 | 0.6121 |
| 8 | 0.5006 | [0.4346, 0.5665] | 0.3598 | 0.5282 |
| 16 | 0.4291 | [0.3742, 0.4754] | 0.2970 | 0.4316 |
| 64 | 0.3032 | [0.2271, 0.3785] | 0.1982 | 0.2748 |
| 256 | 0.3413 | [0.2859, 0.4125] | 0.2250 | 0.3349 |

**140K**

| d | mean overlap | per-layer [min,max] | mean Jaccard | cross-chunk overlap |
|---|---|---|---|---|
| 1 | 0.5448 | [0.4224, 0.6564] | 0.3969 | 0.6107 |
| 2 | 0.4422 | [0.3475, 0.5328] | 0.3010 | 0.5036 |
| 4 | 0.3280 | [0.1506, 0.4374] | 0.2107 | 0.3837 |
| 8 | 0.2887 | [0.0958, 0.4013] | 0.1814 | 0.2759 |
| 16 | 0.2254 | [0.0611, 0.3570] | 0.1358 | 0.2294 |
| 64 | 0.1610 | [0.0273, 0.3013] | 0.0936 | 0.1566 |
| 256 | 0.1341 | [0.0220, 0.2734] | 0.0767 | 0.1386 |

Reading: **neighbouring queries do not share almost everything.** At 140K the adjacent pair shares 54 %
of its 512 blocks (Jaccard 0.40), falling to 13 % at d=256. At 16K the adjacent pair shares 64 %. The
chunk-boundary-adjacent pair (cross-chunk d=1) shares **more** than the average — 0.6794 vs 0.6364
(16K) and 0.6107 vs 0.5448 (140K) — so boundaries do not break the pattern; at larger d the
cross-chunk pairs are comparable to the within-chunk ones (16K d=256: 0.3349 vs 0.3413; 140K d=256:
0.1386 vs 0.1341).

At 16K the aggregated curve is **not monotone**: it dips at d=64 (0.3032) and recovers at d=256
(0.3413). This holds in 11 of the 12 layers (only layer 0 is the reverse: 0.3729 vs 0.3481); checked
independently by brute-force `intersect1d` on layers 0, 1 and 11 (layer 1: 0.2560 vs 0.2860; layer 11:
0.2760 vs 0.3079). 140K is monotone decreasing.

**Union-per-block distribution** (`selection_count` = how many of the chunk's queries select a block),
pooled over chunk × layer; the long form is `results/selection_count_histogram.tsv`:

| selected by k queries | 16K (251,199 block-instances) | 140K (1,015,828) |
|---|---|---|
| 1 | 304 (0.121 %) | 64,513 (6.351 %) |
| 2 | 267 (0.106 %) | 46,205 (4.549 %) |
| 3–4 | 540 (0.215 %) | 69,162 (6.808 %) |
| 5–8 | 1,033 (0.411 %) | 97,927 (9.640 %) |
| 9–16 | 2,246 (0.894 %) | 132,680 (13.061 %) |
| 17–64 | 18,965 (7.550 %) | 352,354 (34.686 %) |
| 65–256 | 81,883 (32.597 %) | 209,243 (20.598 %) |
| 257–1024 | 113,518 (45.190 %) | 39,328 (3.872 %) |
| 1025+ | 32,443 (12.915 %) | 4,416 (0.435 %) |

**The fraction selected by only one query is 0.12 % at 16K and 6.35 % at 140K** (pooled; per-row means
0.10 % and 6.64 %). Mean multiplicity is 498× (16K) and 64× (140K) — every distinct block is loaded
tens to hundreds of times under the per-query gather. So the ratio is **not** "a few hot blocks": at
140K the mass is spread from k≈2 up to k≈256 (a long tail of mid-count blocks), and the hot blocks
(`1025+`) are only 0.4 %; at 16K the mass is concentrated in high counts (58 % at `257+`), consistent
with a nearly saturated small pool.

## 3. Dependence on context length

- **Ratio decreases slowly with context length**: 0.9980 (16K, ≤5,610 blocks) → 0.9844 (140K, ≤35,424
  blocks).
- **Within a context it decreases as the block pool grows**: 16K chunk 0 (1,024 blocks) 0.99902 →
  chunk 5 (5,606) 0.99440; 140K chunk 0 (33,792) 0.98656 → chunk 2 (35,424) 0.97792.
- The decrease is driven by `distinct` growing toward the pool size while `sum = queries × 512` stays
  fixed per column; the adjacent-query overlap also falls with context (64 % → 54 %), and the fraction
  of singleton blocks rises (0.12 % → 6.35 %).
- It is **flat in the sense that matters for the gate**: both contexts are more than 3× the ~30 % bar
  (0.998 and 0.984 against 0.30).

## 4. Exclusions and coverage

- **Dense columns** (`counts = -1`): 2,051 at 16K per layer — all in chunk 0, positions 0…2050, where
  `complete_blocks <= 512` and the engine short-circuits QSA to dense causal; that is 24,612 chunk ×
  layer rows, excluded and counted (`per_chunk_layer.tsv`, `dense_columns`). **0 at 140K.**
- **Coverage.** The 140K capture arms on `MIN_POS=131072` and holds only positions 131072…141695 —
  the last 10,624 tokens of the 141,703-token prompt. So the 140K ratio describes the ≥128K regime,
  not a from-zero 140K prompt. The 16K capture covers positions 0…22425 (the whole 22,433-token
  prompt).
- The open tail (the `position % 4` tokens after the last complete block) is outside `selected.bin` and
  is not part of this metric; per the facts note a column gathers its 512 blocks **plus** the tail.

## What was not done

- The paper's overlap definition (QUILT, arXiv 2610.11134) was not read here — that is ticket #47. The
  metric used is stated above so the two can be reconciled.
- The same overlap was **not** measured for the decode path (one query per step) — out of scope per the
  map.
- Nothing was timed on the GPU: this is a measurement of the selections, not of a shared-gather
  prototype. The bytes are the gather volume the geometry implies, not a measured kernel time.
