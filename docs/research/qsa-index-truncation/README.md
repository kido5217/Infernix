# QSA index truncation — prototype (issue #39 rev 5, ticket #40 capture)

> Prototype branch `proto/qsa-index-truncation`. **No PR, never merged.** The `main` tree is
> untouched. Branch from `origin/main` (`1e2eda5b`).

This answers the question that survives for this build (issue #39 §0):

> **Does a coarser index plane preserve the selected blocks, and what does it cost in bytes?**

CPU-only analysis over ticket #40's captured data. The binding spec is the **first comment of
issue #39 (rev 5)**; where this file and that comment differ, the comment wins. No threshold,
metric or arm definition was changed after seeing any arm's numbers; every open point is recorded
in `raw/deviations.txt` (written before the results were read) and is summarized under
[Deviations](#deviations).

## Exact command

```bash
cd /home/kido/network/projects/Infernix
nix develop . --command python3 docs/research/qsa-index-truncation/truncation_probe.py \
    verify-inputs --out docs/research/qsa-index-truncation/raw
nix develop . --command python3 docs/research/qsa-index-truncation/truncation_probe.py \
    arms --out docs/research/qsa-index-truncation/raw --contexts 16k-eager,140k-eager --arms A1,A2
nix develop . --command python3 docs/research/qsa-index-truncation/truncation_probe.py \
    a3 --out docs/research/qsa-index-truncation/raw --contexts 16k-eager,140k-eager
nix develop . --command python3 docs/research/qsa-index-truncation/truncation_probe.py \
    report --out docs/research/qsa-index-truncation/raw
```

Python 3.13 + numpy 2.4.4 from `nix develop .`; no GPU. `arms` runs the full-coverage fidelity
check (identity recompute) and the A1/A2 arms in one pass; `a3` calibrates the A3 water line from
A4's mean bits (bits only) and runs V1/V2; `report` derives the per-layer tables, the budget curve
and the bar verdict.

## Dataset (primary, issue #39 §1)

| run | directories used |
|---|---|
| primary | `out/qsa-index-capture/16k-eager/` and `out/qsa-index-capture/140k-eager/` |
| corroboration (not used for the arm metrics here) | `out/qsa-index-capture/16k/`, `140k/`, `scores-16k/`, `scores-140k/` |

Only the `selections/` trees are read (each capture is one QSA layer call). The engine revision per
manifest is `69fcf7d3…` with head `51601ea2…` (ticket #40). A **layer** is identified by its plane
pointer (`pooled_ptr`), ranked per run (13 distinct values; index 12 is the MTP plane). A **step** is
the capture's `step` field. Full column coverage: **245,007** non-dense columns at 16K (24,806 dense
columns excluded) and **128,012** non-dense columns at 140K (0 dense). The bar applies at 140K
(≥128K); 16K is the trend (§7.12).

## Arms and byte costs (§2)

| # | Store | Bytes/block | Formula |
|---|---|---|---|
| A0 | full BF16 | 256 | 128 ch × 16 bit |
| A1(t) | uniform *t*-bit code + fp16 scale | 32 / 48 / 64 / 80 | 16·t + 16 |
| A2(m) | packed leading *m* bits of each BF16 word, read back as BF16 | 128 / 144 / 160 / 192 / 256 | 16·m |
| A3 | water-filling per-channel / per-group depth (V1 gating, V2 reported) | (Σ_j t_j)/8 + 16 | varies |
| A4 | one fixed A1 budget, chosen by the §2 rule | as A1(t*) | — |

A1 follows the paper's quantizer: per channel, one fp16 scale per 16 captured blocks (the paper's
64-token block = one pooled-plane token page), `a = max|key|`, `s = a/8`,
`c4 = clip(floor(key/s), −8, 7) + 8` (MSB = sign, code 8 = zero); a depth-`t` read takes
`c_t = c4 >> (4−t)` and reconstructs `(c_t + ½ − 2^{t−1}) · a/2^{t−1}`, the paper's *t*-bit mid-rise
quantizer. A2 truncates each BF16 word's leading *m* bits with no rounding (A2(16) = A0 by
construction). The query is never quantized.

## Fidelity check (§1, gates the run — §7.2)

Run over **every non-dense column of the primary dataset** (full coverage, not the 8-column sweep),
recomputing `score_b = (1/√Di)·Σ_h ReLU(⟨q_h, key_b⟩)` from the captured pooled keys and query:

| context | scored columns | max abs \|Δ\| | max column-scaled | ids checked | exact | explained | unexplained | dense excluded |
|---|---|---|---|---|---|---|---|---|
| 140K | 5,132 | 4.768e-06 | 4.653e-07 | 128,012 | 127,971 | 41 | **0** | 0 |
| 16K | 8,499 | 5.722e-06 | 5.155e-07 | 245,007 | 244,972 | 35 | **0** | 24,806 |

* Bounds: `1e-5` absolute and `1e-6` column-scaled — both hold.
* Every id mismatch is **explained**: its boundary gap `|s(last selected) − s(first rejected)|` is
  below the applicable bound (the recomputation's own column error where the column is scored, else
  the capture's measured maximum error; bound used reported per mismatch in
  `raw/fidelity_mismatches_*.json`). **No unexplained mismatch.**
* The id rate at full coverage is 99.968 % (140K) and 99.986 % (16K); the unmatched are
  float32 accumulation-order near-ties, exactly the case rev 4 dropped the rate floor for.
* The recomputation is a batched (BLAS-order) implementation of the same math; its max error
  differs from the vendored verifier's per-column matmul at the 1e-7 level but stays inside the
  fixed bound. `equiv_check.py` re-runs the vendored `verify_capture.analyze(cap, full)` on the 444
  captures where the per-column verifier is tractable (all decode captures plus named prefill
  captures) and compares counts: **444/444 agree on every order-independent count** (scored, ids,
  dense, zero unexplained) and both are within bounds, with **one** capture (140K `step_02/s_032`)
  differing by a single exact/explained split — the batched FP32 order flips one near-tie that the
  per-column order resolves exactly, and the flip is *explained*. This is precisely the
  accumulation-order effect the rev-4 rationale accepts; `raw/equivalence_check.txt` records it.

## Results

### Bar verdict on A1(4) (§4)

`A1(4)` = 80 B/block = **3.2×** fewer bytes than A0's 256 B.

| context | mean agreement | cond 1 ≥0.95 | worst layer mean | cond 2 ≥0.90 | spread | cond 3 ≤0.10 | whole curve (cond 4) | verdict |
|---|---|---|---|---|---|---|---|---|
| **140K (bar)** | **0.9213** | **FAIL** | **0.8951** (layer 12, MTP) | **FAIL** | 0.0510 | PASS | PASS | **NOT CLEARED** |
| 16K (trend) | 0.9536 | PASS | 0.9209 (layer 12) | PASS | 0.0427 | PASS | PASS | cleared |

At ≥128K **A1(4) does not clear the bar**: mean (0.9213) is below 0.95 and the worst layer's mean
(0.8951) is below 0.90. The 3.2× byte cut is real; the accuracy bar is not met at the adopted
context.

**Weighting robustness.** The mean above is column-weighted (`Σ|S∩S0| / Σm0` over all (layer, step)
rows). Averaging the 13 per-layer means instead gives **0.9193** (140K) / **0.9511** (16K), and
weighting each step equally gives **0.9296** / **0.9458**. The 140K verdict — fail on conditions 1
(mean < 0.95) and 2 (a layer mean < 0.90) — holds under all three; spread stays ≤ 0.10 under all
three. (At 16K the step-equal variant would put the mean at 0.9458; 16K is the trend, not the bar.
Every weighting is re-derivable from `raw/per_layer_<ctx>.tsv` and `raw/rows_<ctx>_fixed.tsv`.)

### Budget curve (§3.1–3.3, every layer in `raw/per_layer_*.tsv`, every column in `raw/per_column_*.npz`)

Cross-layer mean / worst-layer / spread, rank64 and score error (all means over every non-dense
column; score error over scored columns against A0's captured scores):

**140K (bar context):**

| arm | B/blk | reduc | mean | worst layer | spread | rank64 | score err |
|---|---|---|---|---|---|---|---|
| A1(1) | 32 | 8.0× | 0.4491 | 0.3344 | 0.2574 | 0.3965 | 9.23e-02 |
| A1(2) | 48 | 5.33× | 0.6997 | 0.6088 | 0.1751 | 0.6523 | 3.46e-02 |
| A1(3) | 64 | 4.0× | 0.8421 | 0.7948 | 0.0974 | 0.8141 | 1.74e-02 |
| **A1(4)** | **80** | **3.2×** | **0.9213** | **0.8951** | 0.0510 | 0.9058 | 8.64e-03 |
| A2(8) | 128 | 2.0× | 0.5203 | 0.3736 | 0.2569 | 0.4467 | 2.05e-01 |
| A2(9) | 144 | 1.78× | 0.7471 | 0.6537 | 0.1734 | 0.7041 | 1.21e-01 |
| A2(10) | 160 | 1.6× | 0.8575 | 0.8082 | 0.0985 | 0.8295 | 6.89e-02 |
| A2(12) | 192 | 1.33× | 0.9618 | 0.9485 | 0.0265 | 0.9526 | 1.82e-02 |
| A2(16) | 256 | 1.0× | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 2.66e-08 |
| A3 V1/V2 | 80 (640 bit) | 3.2× | 0.9213 | 0.8951 | 0.0510 | 0.9058 | 8.64e-03 |

**16K (trend):**

| arm | B/blk | reduc | mean | worst layer | spread | rank64 | score err |
|---|---|---|---|---|---|---|---|
| A1(1) | 32 | 8.0× | 0.6723 | 0.5511 | 0.1834 | 0.5033 | 1.14e-01 |
| A1(2) | 48 | 5.33× | 0.8182 | 0.7074 | 0.1522 | 0.7147 | 4.52e-02 |
| A1(3) | 64 | 4.0× | 0.9073 | 0.8440 | 0.0838 | 0.8535 | 2.25e-02 |
| A1(4) | 80 | 3.2× | 0.9536 | 0.9209 | 0.0427 | 0.9256 | 1.12e-02 |
| A2(8) | 128 | 2.0× | 0.7185 | 0.5504 | 0.2157 | 0.5670 | 2.83e-01 |
| A2(9) | 144 | 1.78× | 0.8572 | 0.7741 | 0.1100 | 0.7703 | 1.69e-01 |
| A2(10) | 160 | 1.6× | 0.9215 | 0.8690 | 0.0707 | 0.8698 | 9.59e-02 |
| A2(12) | 192 | 1.33× | 0.9791 | 0.9644 | 0.0192 | 0.9639 | 2.55e-02 |
| A2(16) | 256 | 1.0× | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 3.69e-08 |
| A3 V1/V2 | 80 (640 bit) | 3.2× | 0.9536 | 0.9209 | 0.0427 | 0.9256 | 1.12e-02 |

Per-layer values (every one of the 13 layers) are in `raw/per_layer_<ctx>.tsv`; the raw per
(arm, layer, step) counts are in `raw/rows_<ctx>_{fixed,a3}.tsv`;
every column's agreement/rank/score error is in `raw/per_column_<ctx>_{fixed,a3}.npz`.

At 140K the worst layer for A1(4) is **layer 12 (the MTP plane), 0.8951**, then layer 6 (0.8957);
the best is layer 11 (0.9462). At 16K the worst is also layer 12 (0.9209).

A2's content is its accuracy below `m = 16`: A2(12) (192 B, 1.33×) already reaches 0.9618 (140K),
above A1(4)'s 0.9213 at 80 B, while A2 needs 192 B to beat a 4-bit re-encode at 80 B — the
re-encode is the better byte/accuracy trade on this index.

### A3 vs A4 (§2, §4)

A4 = the smallest A1 `t` clearing the bar; none clears at 140K, and 16K resolves to A1(4) too, so
**A4 = A1(4)** in both contexts (mean bits `128·(t*+1) = 640` bits/block). A3's water line θ is
calibrated by bits only (§D8): θ = 9.09495e-13.

* At that budget the water-filling degenerates: **every channel takes depth 4 on every column**, so
  the realized mean bits are exactly 640 and A3's scores equal A1(4)'s **exactly**. The probe checked
  the all-4 condition on every column and then reused A1(4)'s per-column metrics
  (`A3 reused-A1(4) columns: 245007/245007` at 16K and `128012/128012` at 140K); columns that fail
  the check would be scored in full, and none did.
* Therefore A3 beats A4 by `+0.0000` mean agreement at equal bits (640.0 vs 640.0) — below the 0.02
  gate. **A3 does not gate; A4 carries forward** (`raw/report.txt`).
* A3 V2 (per-group `a²`) is reported and cannot gate; it produced the same numbers as V1 here.

### Bytes and frames (§3.3)

Bytes/block × the deepest captured column's complete blocks (16K 5,610; 140K 35,426), per layer and
for all 13 planes; frames = bytes / 2,764,800 B (the measured routed-expert frame):

| | A0 | A1(3) | **A1(4)** | A2(10) | A2(12) |
|---|---|---|---|---|---|
| 140K plane/layer | 8.65 MiB | 2.16 MiB | **2.70 MiB** | 5.41 MiB | 6.49 MiB |
| 140K 13 planes | 112.44 MiB | 28.11 MiB | **35.14 MiB** | 70.27 MiB | 84.33 MiB |
| 140K frames (13) | 42.64 | 10.66 | **13.33** | 26.65 | 31.98 |
| 16K frames (13) | 6.75 | 1.69 | **2.11** | 4.22 | 5.06 |

### Time (§3.4, reported, non-gating)

No device scan exists here and the scan is latency-dominated (issue #39 §5), so time is reported,
never used to gate. The prototype's own CPU numpy cost (`raw/timing_*.json`): the 140K `arms` stage
(identity/fidelity + A1 + A2 over 260 captures) took **899 s** wall, with each A1/A2 arm's
score+select loop ≈84 s and the identity/fidelity pass ≈880 s; the 16K stage took 293 s. (In the
timing JSON the `A0re` entry is the whole-capture wall, not a separate identity-only figure.) The
modelled device bandwidth floor (bytes / 1.7 TB/s) is ~5.3 µs/layer for A0 and ~1.7 µs/layer for
A1(4) at 140K — a floor, not a measurement, and it does not gate.

## Input identity (§8; D15)

`raw/input_hashes.txt`: the two vendored manifests
(`manifest-eager-16k.txt` / `manifest-eager-140k.txt`, copied from the capture branch) are the hash
record. The run re-hashed **every** selection file it read and checked it against the manifest:
**2700/2700** matched at 16K and **2340/2340** at 140K (0 mismatched, 0 missing), with combined
digests recorded there.

## Deviations

Full text in `raw/deviations.txt`, written before the numbers were read. Headline points:

* **D1** scale group = 16 consecutive captured blocks = one plane token page (the protocol's only
  64-token unit here).
* **D2/D3** A1's `a` is the max of the *stored pooled* block keys (raw per-token keys are not
  captured); A1(t) stores the t-bit prefix.
* **D5/D6** A3 V1 uses the capture's own key variance; V2 uses the per-group `a²`; budget realised
  as a global water line on mean bits.
* **D7** rank agreement = |A0 top-64 ∩ arm top-64| / 64 with A0's top-64 from its captured scores
  (else the recomputed ones).
* **D8** A3 budget chosen from bits only, never agreement, on a deterministic sample.
* **D11** time = prototype CPU loop time + modelled bandwidth floor, non-gating.
* **D12** layer identity = `pooled_ptr` rank (13 values; index 12 = MTP).

## Files

| path | what |
|---|---|
| `truncation_probe.py` | the prototype (stages: `verify-inputs`, `arms`, `a3`, `report`) |
| `verify_capture.py` | vendored parsing logic (`proto/qsa-index-capture` @ 05f74f1a); parsing reused verbatim |
| `equiv_check.py` | runs the vendored verifier at full coverage on the tractable captures and checks it against `raw/fidelity_captures_*.tsv` |
| `manifest-eager-*.txt`, `manifest-eager-*.commit.txt` | vendored input-hash record |
| `raw/deviations.txt` | deviations, written before results |
| `raw/report.txt` | the report stage's stdout (curves, bar, A3-vs-A4) |
| `raw/rows_<ctx>_{fixed,a3}.tsv` | per (arm, layer, step) raw counts (columns, Σinter, Σrank, Σscore-err) |
| `raw/per_layer_<ctx>.tsv` | per (arm, layer) agreement / jaccard / rank64 / score error |
| `raw/per_column_<ctx>_{fixed,a3}.npz` | per-column agreement, rank and score error for every arm |
| `raw/fidelity_<ctx>.json`, `raw/fidelity_captures_<ctx>.tsv`, `raw/fidelity_mismatches_<ctx>.json` | the fidelity gate |
| `raw/equivalence_check.txt` | batched fidelity vs the vendored per-column verifier |
| `raw/a3_budgets_<ctx>.json` | A4 choice, target/realized bits, θ |
| `raw/input_hashes.txt` | input hashes and the manifest re-hash check |
| `raw/timing_<ctx>_fixed.json` | prototype CPU time per arm |

## Limitations

* The **primary** dataset is the eager run; the corroboration runs were not used here (they are a
  different schedule with fewer steps, §1). Per-layer claims therefore rest on the eager run's 260/300
  selection captures and 13 planes.
* A3 V2 is a per-(group,column) variant; its full-coverage values are reported but it cannot gate.
* The scan time reported is the prototype's numpy cost, not the engine's device scan; the engine
  scan is latency-dominated (§5), so no time claim is made.
