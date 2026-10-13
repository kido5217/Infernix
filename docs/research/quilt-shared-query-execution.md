# QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution — reading note

Facts for map **#45** and task **#47** ("QUILT mechanism: the overlap metric, set decomposition and
cascaded sharing"). The paper is **arXiv:2610.11134v1, 2026-10-08, Huawei** (Zhenduo Zhao, Qihui Zhou
et al.).

**What I read.** The **full HTML** at `https://arxiv.org/html/2610.11134v1` (371 lines extracted:
abstract, §1–6, references), not the abstract. All section, equation and figure numbers below are
from that HTML version.

**Our side**, read first: `research/qsa-prompt-overlap-facts` head `30e70f87`
(`docs/research/qsa-prompt-overlap-facts.md`) and `proto/qsa-overlap-measurement` head `b32895e3`
(`docs/research/qsa-overlap-measurement/README.md`, `results/tile_union.tsv`,
`results/per_chunk_layer.tsv`).

Labels: **(paper)** = stated by the paper; **(ours)** = established in this repo (with a path/line);
**(new)** = measured for this note; **(inference)** = my judgement about our build.

---

## Headline

- The paper's headline *overlap* quantity (Figure 1) is the **group-intersection fraction** — the share
  of the Top-K that **all** queries in a group select (`|⋂_{q∈group} S_q| / K`). It is **not** our
  tile-union metric. At group size 2 it reports **>85.91 % (Qasper) / >78.06 % (GovReport)** over all
  layers, >90.03 % in layer 1 (paper §1).
- The same quantity recomputed on our selections **(new)**: **f(w=2) = 0.648 (16K) / 0.551 (140K)**,
  falling to 0.235 / 0.112 at w=8. Our adjacent-query intersection is **materially lower** than the
  paper's. Only our **union** metric is of the paper's magnitude, and it corresponds to the paper's
  *processed-KV reduction*, not its overlap ratio (§1.4).
- The mechanism **does not change the selection**: SCSD (§3.1) and cascaded sharing (§3.3) are exact
  set algebra on the *given* Top-K lists, and §2.2 says QUILT is "agnostic to how sparse indices are
  generated". The **only** selection-changing step is **importance-aware clipping** at the final
  query-specific level (§3.4), which the paper itself calls "the only approximate step".
- The paper's exactness claim is **semantic** (same KV set; padded entries masked; push-down exact),
  argued by construction and measured empirically. It gives **no error bound**, and it is **silent on
  floating-point accumulation order / bit-exactness** — exactly the property our width-invariance
  contract imposes.
- Gains (all on 16× Ascend 910C, GLM-5.3 / DeepSeek-3.2, Top-2048, LongBench, TP+SP): kernel latency
  up to **55.1 %** (avg 53.8 % GLM-TP), processed KV up to **55.9 %** (avg 54.7 % GLM), TTFT up to
  **36.8 %** (avg 35.6 % GLM-TP). TTFT is the only end-to-end number; the other two are kernel-level.

---

## 1. The overlap claim, and its reconciliation with our tile-union metric

### 1.1 What the paper measures **(paper)**

| Aspect | Paper |
|---|---|
| Where | §1 + **Figure 1** ("Top-K KV overlap ratios among neighboring queries across model layers and query-group sizes for GLM-5.3 on Qasper and GovReport") |
| Quantity | fraction of the Top-K **shared by all queries in the group**; report is a function of **group size** and **layer** |
| Granularity | **KV entries (tokens)**, selected **per query**; Top-**2048** per query (§4.1) |
| Head granularity | **not stated** (silent); the indexer is a per-layer Lightning Indexer / DSA selector (§4.1), which selects per query token |
| Model / workload | GLM-5.3, LongBench **Qasper** and **GovReport**, all layers |
| Numbers | group size 2: "consistently exceeds **85.91 %** and **78.06 %** across all layers … and further reaches over **90.03 %** in the first layer"; decreases with group size |

That the quantity is the **all-queries intersection** (not pairwise, not union) is textually forced:
the paper says "As the group size increases, the **fraction of KV entries shared by all queries in the
group** gradually decreases" (§1). A union/redundancy quantity *increases* with group size; this one
decreases. The denominator is the Top-K `K` (§4.1: Top-2048), i.e. the metric is `|⋂ S_q| / K`. The
paper does not print the formula, only Figure 1 and those two percentages.

### 1.2 What we measure **(ours)**

`proto/qsa-overlap-measurement` defines, over non-dense columns (`counts = -1` excluded, `|S_q| = 512`
block ids of `ratio = 4` tokens each):

- **tile-union ratio** `1 − |⋃_{q∈tile} S_q| / (w·512)` over consecutive-column tiles:
  **0.318 / 0.275** at w=2, **0.546 / 0.460** at w=4, **0.704 / 0.587** at w=8, **0.811 / 0.678** at
  w=16 (16K / 140K). Chunk-level union (the `w→∞` asymptote, not bankable): 0.998 / 0.984.
- distance curve: adjacent-pair overlap `|S_q ∩ S_{q+d}| / 512` = **0.636 / 0.545** and Jaccard
  **0.498 / 0.397** at `d = 1` (`results/distance_curve.tsv`).

### 1.3 The paper's metric, recomputed on our selections **(new)**

I recomputed the paper's quantity `f(w) = mean_tiles |⋂_{q∈tile} S_q| / 512` from the same
`selected.bin`/`counts.bin` captures, using the measurement branch's exact conventions (prefill
captures with `columns ≥ 1000`, keyed by `pooled_ptr`, ordered by `min_position`, dense columns
excluded). Mean over the 12 layers per context:

| context | w=2 | w=4 | w=8 | w=16 | w=32 | w=128 | w=512 | w=all |
|---|---|---|---|---|---|---|---|---|
| 16K | **0.648** | 0.404 | 0.235 | 0.137 | 0.080 | 0.029 | 0.014 | 0.005 |
| 140K | **0.551** | 0.256 | 0.112 | 0.058 | 0.029 | 0.006 | 0.003 | 0.002 |

Cross-check: the w=2 values reproduce the branch's `d=1` pairwise overlap (0.648 vs 0.6364 at 16K;
0.551 vs 0.5448 at 140K). The small gap is tile-remainder handling and layer-averaging; the agreement
confirms the reading of the two metrics.

Method (reproducible; CPU-only, read-only over `out/qsa-index-capture/`): for each tile concatenate
the tile's `w` columns' 512 ids, `np.unique` with counts, and take the count of values occurring in
all `w` columns; divide by 512; average over tiles, then over layers.

```python
def group_intersection(cols, w):          # cols: (n_valid, 512) int32, ascending per column
    n, top = cols.shape
    step = n if w is None else w
    sizes, i = [], 0
    while i + step <= n:
        vals, cnt = np.unique(cols[i:i+step].ravel(), return_counts=True)
        sizes.append(int(np.count_nonzero(cnt == step)))
        i += step
    return float(np.mean(sizes)) / top
```

### 1.4 Reconciliation — three different quantities

The paper and the measurement branch are **not** measuring the same thing, and neither is the
"asymptote" reading. Three distinct quantities:

1. **Paper's overlap ratio** (Figure 1): the **group-intersection fraction** `f = |⋂_{q∈group} S_q| / K`.
   Our analogue is the table in §1.3.
2. **Our tile-union ratio** `1 − |⋃_{q∈tile} S_q| / (w·512)` (measurement branch). This is **not** (1).
   For sets of size `K` with common intersection `fK`, `|⋃| ≤ fK + w(1−f)K` gives the bound
   `1 − |⋃|/(wK) ≥ f·(1 − 1/w)`, tight when the query-specific residues are disjoint:
   - w=2: 16K bound `0.648·0.5 = 0.324` vs measured **0.318**; 140K bound `0.276` vs **0.275** → tight.
   - w=8: 16K bound `0.206` vs measured **0.704**; 140K bound `0.098` vs **0.587** → the residues
     overlap heavily, so the union redundancy is far above the intersection bound.
3. **Paper's processed-KV reduction** (Tables 3, 4). A **fully-recursed cascade processes every distinct
   KV entry exactly once** — Table 1: "No sharing 16 … Cascaded **8**", and the text says 8 "equal to
   the number of distinct KV entries". So — **to first order** (the realized Tables 3/4 also fold in
   tile-aware push-down and importance-aware clipping, which cut the count further) — the processed-KV
   reduction equals the **union redundancy** `1 − |⋃ S_q|/(wK)` at the group width, i.e. **our tile-union
   metric**. Their numbers: Table 4
   cascaded rows normalized to baseline (GLM 45.5 %, DeepSeek 70.1 %) ⇒ reductions **54.5 % / 29.9 %**;
   Table 3 averages **54.7 % / 30.4 %**.

So, precisely:

- **To be comparable to the paper's headline overlap claim, our metric must be the group-intersection
  fraction** (§1.3), not the tile-union ratio. At the paper's group size 2 ours is 0.55–0.65 against
  their 0.78–0.86: **our selected sets are less correlated than GLM-5.3's on Qasper/GovReport**
  (different model, indexer and workload — see §5).
- **Our tile-union ratio is comparable, to first order, to the paper's processed-KV reduction**: ours
  0.46–0.70 at w=4–8 (16K) and 0.46–0.59 (140K) against their 0.30 (DeepSeek) / 0.55 (GLM); at w=16
  ours (0.81 / 0.68) exceeds both.
- **Consistency verdict.** Their overlap ratio is *not* consistent with our tile-union numbers read as
  the same quantity (0.86 ≠ 0.32). Nor is it consistent through the union bound: `f = 0.86` would force
  tile-union `≥ 0.43` at w=2, where we measure 0.27–0.32. It **is** consistent once the two are
  recognised as different quantities (intersection vs union redundancy). The practical consequence for
  our build: a **group-wide shared core is not where our saving is** (only 0.24 / 0.11 at w=8); the
  saving lives in the **union** (0.32 from w=2 at 16K, 0.70 at w=8), which a group-intersection-first
  scheme does not bank.

---

## 2. Shift-and-Compare Set Decomposition and cascaded sharing

### 2.1 SCSD — **§3.1, Equations (1)–(7)** (paper)

Given two Top-K index lists `A`, `B` of `N` unique indices each (the paper uses the two-way case "for
clarity, while the same principle naturally generalizes to multi-way"):

```
(1)  (C, T) = Sort(A ‖ B)          C = sorted indices, T_i ∈ {0,1} the source label
(2)  D = (C = C_r)                C_r = right-shifted C
(3)  F = (C = C_l)                C_l = left-shifted C
(4)  E = C[D] = A ∩ B             intersection (one copy of each duplicated index)
(5)  G = ¬(D ∨ F)                 entries matching no neighbour = unique to one list
(6)  A − E = C[G ∧ (T = 0)]
(7)  B − E = C[G ∧ (T = 1)]
```

Because each list is unique, a shared index appears exactly twice in `(C,T)` and, after the sort, the
two copies are adjacent; `D`/`F` capture the two copies. The whole thing is shifts, element-wise
comparisons, Boolean ops and masked selects — "regular data-parallel primitives" that map onto vector
units, avoiding scalar merge/loops.

**Does it change the selection? No.** SCSD is pure set algebra on the lists it is given; it produces
the intersection and the two differences. §2.2 reinforces this: "QUILT operates directly on these
resulting sparse workloads and is therefore **agnostic to how sparse indices are generated**, refined,
or reused."

### 2.2 Pipelined execution — **§3.2** (paper)

SCSD would otherwise sit on the critical path (`T_SCSD + T_Attn`). The paper overlaps it with attention:
QK/PV run on the matrix pipeline, SCSD (sort/shift/compare/Boolean) on the vector pipeline; construct
each sparse segment only just before it is consumed, prime with a few segments so the pipeline reaches
steady state, and split a segment's construction into finer stages that fit the vector-side slack.

### 2.3 Cascaded sharing — **§3.3** (paper)

Start from a relatively **large group (8 by default)**; extract the entries shared by *all* queries and
process them jointly; remove them; **recursively** partition the residual into smaller subgroups and
repeat; when no further sharing is possible, process the rest as query-specific residuals. The running
example (§3.3, Table 1): `S1={A,B,C,E}`, `S2={A,B,C,F}`, `S3={A,B,D,G}`, `S4={A,B,D,H}` → four-way
`{A,B}` shared, then two-way `{C}`, `{D}` in the two subgroups, then residuals `E,F,G,H`;
processed entries 16 (none) / 10 (group 2) / 10 (group 4) / **8 (cascaded = the distinct count)**.
The purpose is to not commit to one fixed group size, and each entry is "processed at the largest query
granularity over which it can be shared".

### 2.4 Tile-aware clipping — **§3.4** (paper) — the only selection-changing step

Let `T` = KV entries per attention tile and `|S| = qT + r` (`0 ≤ r < T`) for a shared segment; `τ = 50 %`
by default:

- `r ≥ τT` → **pad** the tail to a full tile and process it at the current level, "**masking the padded
  entries to preserve exact semantics**".
- `r < τT` → **push the tail down** to the next cascade level, merged with finer workloads. "**This
  push-down is exact**".
- At the final query-specific level, no push-down is possible: for `|R_i| = q_iT + r_i`, keep the
  `q_iT` entries with the highest indexer scores and **prune the `r_i`** least important. "This is the
  **only approximate step** in our tile-aware execution".

**Verdict for us: SCSD + cascaded sharing + push-down only reorganize paging/gather of an unchanged
selection. Importance-aware clipping is the one mechanism that changes what is attended.**

---

## 3. Exactness and fidelity statements (paper)

| Where | Quote / claim |
|---|---|
| §1 | "This push-down is **exact** and trades a small amount of sharing for improved tile utilization." |
| §1 | "We therefore apply *importance-aware clipping*, pruning the lowest-ranked residual entries … This **approximation** further reduces attention-tile execution while introducing **negligible accuracy degradation**." |
| §3.4 | "masking the padded entries to **preserve exact semantics**"; "This push-down is **exact**"; "This is the **only approximate step** in our tile-aware execution and minimizes accuracy degradation by discarding only low-importance residual entries." |
| §4.2.1 | "These results confirm that our **exact execution optimizations preserve the original sparse attention semantics**, while the importance-aware clipping … introduces only negligible accuracy variance." |
| §4.2.1 (accuracy) | Table 2, **21 LongBench datasets**, official per-dataset metric: mean absolute score difference **0.75** (DeepSeek-3.2) / **0.42** (GLM-5.3); mean absolute percentage difference **2.01 %** / **0.90 %**. Largest deltas: musique +2.71, hotpotqa −1.45 (DeepSeek); lcc −0.91, narrativeqa −0.84 (GLM). |
| §4.3.3 (fidelity) | Figure 15, cosine similarity of clipped vs unclipped attention output: head-level mean **1.0000** (random 0.9999), **min 0.406** (random 0.312); hidden-level mean **1.0000** (0.9998), **min 0.719** (0.664). |

**Nature of the claim.** The exact steps are argued exact **by construction** (set algebra + masking),
not by an error bound; the paper gives **no bound** on the clipping error. The accuracy evidence is
empirical (21 LongBench tasks) plus cosine similarity. Two caveats for us:

- The clipping's *worst case* is not negligible: the head-level **minimum** cosine similarity after
  importance-aware clipping is **0.406** (better than random's 0.312, but far from 1). "Negligible"
  describes the mean and the task aggregate, not every head.
- "Exact" is **semantic** (same KV set), **not bit-exact**. Shared entries are processed once and reused
  at a different point in the pipeline than the per-query order; the paper never discusses
  floating-point accumulation order. Our contract (below) is stricter than anything the paper claims.

---

## 4. Reported gains and their conditions (paper)

**Setup (§4.1).** **16× Ascend 910C NPUs**, 64 GB HBM each, 784 GB/s bidirectional D2D, **CANN 9.1.0**.
Baseline: the **OPS-Transformer** sparse-attention kernel, integrated into the same **XYServe** serving
engine (same scheduling, memory management, runtime; the kernel is the only difference). Models:
**GLM-5.3** and **DeepSeek-3.2**, both model-native sparse attention, **Top-2048** per query (Lightning
Indexer / DSA). Workloads: **LongBench** (21 datasets for accuracy; Qasper, GovReport, Musique,
NarrativeQA for performance). **Tensor parallelism (TP)** and **sequence parallelism (SP)**.

| Claim | Value | Level | Conditions |
|---|---|---|---|
| Kernel latency | up to **55.1 %** (GLM-TP, GovReport); GLM-TP avg **53.8 %**, GLM-SP avg **43.4 %**, DS-TP avg **28.7 %**, DS-SP avg **17.1 %** | **kernel microbenchmark** (Fig. 9) | 4 workloads, both models, TP/SP |
| Processed KV | up to **55.9 %** (GLM, GovReport); GLM avg **54.7 %**, DS avg **30.4 %** | **kernel-level count** (Table 3) | same |
| TTFT | up to **36.8 %** (GLM-TP, Musique); GLM-TP avg **35.6 %**, GLM-SP avg **10.4 %**, DS-TP avg **23.0 %**, DS-SP avg **3.4 %** | **end-to-end** (Fig. 11, XYServe) | same |
| Sharing-length sweep | length 0→2048 (all of Top-K shared) → latency −42.74 % / −33.25 % (GLM TP/SP), −41.03 % / −29.53 % (DS) | kernel (Fig. 10) | synthetic controlled-sharing index lists; small overhead at low sharing |
| SCSD vs naive | naive Torch set-intersection **21.84× / 33.65× slower** than baseline (GLM); SCSD removes **96.1 % / 97.3 %**; pipelining a further **45.9 % / 36.9 %** | kernel ablation (Fig. 12) | full design beats baseline by 35.5 % on average |
| Cascaded vs fixed groups | CS reduces processed KV by 30.8 % / 20.6 % / 22.9 % vs groups 2/4/8 (GLM) | kernel (Fig. 13, Table 4) | |
| Tile-aware clipping | −23.71 % / −22.68 % (GLM TP/SP), −24.44 % / −14.13 % (DS) | kernel (Fig. 14) | vs an exact, unclipped variant |

**Which is end-to-end, which is micro.** Only **TTFT** (§4.2.4, Figure 11) is end-to-end. Kernel
latency (Fig. 9) is a sparse-attention *kernel* measurement; processed KV (Table 3) is a *count*, not a
time. The paper has **no end-to-end throughput (tokens/s) number**.

**Prefill shape.** The text gives **no** concrete prefill sequence length or number of query
columns/keys per batch. Figure 2 is a latency breakdown "across different prefill sequence lengths (SP)
and query heads (TP)" but the extracted text carries no numeric lengths. The sharing-length sweep goes
to 2048 tokens (its maximum, = Top-K). **Silent** on the exact batch/columns shape at which the 55.1 %
is realized.

**Quantization.** §4.1 states **no** KV dtype or bit-width. §2.2 says KV "is often stored in a
low-precision format and the selected entries are dequantized on demand", but the actual format used in
the evaluation is **not stated in the readable text** (it may sit only in a figure). **Silent.**

---

## 5. What would transfer to our build and what would not

### 5.1 Our relevant geometry (ours, from `30e70f87`)

- **Selection is exact per query**: top-**512** blocks of **4 tokens** (= 2048 tokens) of the context,
  ids materialized as `selected [512, columns]` i32 + `counts [columns]` (`qsa_select.h:54-57`);
  selection processes columns in groups of `kQsaSelectGroupColumns = 128` (`qsa_select.h:48`), a warp
  reads 8 pooled keys once and scores up to 16 columns (`qsa_select.h:22-24`). Block-sparsity (2048
  tokens/query) equals the paper's **token**-sparsity (Top-2048).
- **Gather is per query**: one CTA per `(column, kv_head)` (`qsa_prompt.cu:895`), `cp.async` token rows
  in 16-byte chunks (`qsa_prompt.cu:367, 412-413`), each warp sweeping 16-token tiles double-buffered
  with its own online softmax (`qsa_prompt.h:14-21`); **no cross-query staging, no per-chunk list**.
- **Chunk**: `--prefill-chunk 4096` columns; 12 QSA layers in a text prefill.
- **Exactness contract**: width-invariance + fixed accumulation — **share loads, never reorder a
  query's accumulation** (ascending block id, then the open tail; per-head butterfly). `qsa_prompt.h:5-10`,
  `qsa.cu:470-473`; facts note §3.

### 5.2 Transfers

- **Set decomposition is unit-agnostic.** SCSD operates on integer index lists, so it applies unchanged
  to our **block ids** (512 per column) rather than tokens; the shared/union primitive is the useful part.
- **The union framing matches our own design intent.** Design §8.10 already specifies, for the prompt
  path, "a tile of queries against the **union of their selected blocks with a per-query mask**. It uses
  the same tile loaders" (`qwen3_8-flash-next-design.md:1199-1200`), currently **unimplemented**. QUILT
  supplies the set-decomposition primitive and the evidence that the overhead is the crux.
- **The processed-KV / tile-union equivalence** (§1.4) lets us use our measured curve to bound a
  prototype's bankable signal.

### 5.3 Does not transfer

- **Importance-aware clipping** (§3.4) changes the selection; our ticket #45 forbids changing the index,
  the pool or the top-k rule. Excluded.
- **Group-as-a-shared-GEMM execution** assumes a matrix unit that processes a *group of queries against
  one KV tile* with a separate vector pipeline for SCSD (§2.3, §3.2). Our prompt kernel is per-`(column,
  kv_head)` with **per-query** online softmax and warp-level 16-token tiles; a shared gather must keep
  each query's own accumulator and order. **(inference)** The paper's joint-tile mapping is a different
  kernel shape, not a drop-in.
- **Magnitude of the memory win.** The paper's saving is HBM load + dequantization on Ascend. Our
  measurement shows the capture's working set already fits the 96 MiB L2 at NVFP4 (2.25–12.32 MiB at
  16K; 74.25–77.84 MiB at 140K), so the redundant per-query loads are **L2-served** at that dtype and
  the DRAM-traffic reduction the paper banks may not exist for us (it would at INT8/BF16 at 140K).
  Prompt attention is 245–248 ms/call and its **gather share is unmeasured**.
- **Overlap magnitude.** Their group-2 intersection (0.78–0.86) exceeds ours (0.55–0.65), and at group
  8 theirs is still "considerable" where ours is 0.24 / 0.11. So the **first cascade level (the
  group-wide shared core) is small for us**; a cascade reaches our overlap only by recursing to small
  subgroups — which is just the union again.
- **Tuned constants.** `T` (KV entries/tile), `τ = 50 %`, group start 8, Top-2048 are tied to Ascend
  tile shapes and the paper's token-level selection; ours are 16-token warp tiles over 4-token blocks.
- **Bit-exactness.** Nothing in the paper supports bit-exact equivalence under shared execution. Adopting
  cascaded sharing's multi-level shared-first ordering would **reorder a query's contributions** unless
  the gather still streams in the query's own ascending block-id order. **(inference)**

---

## 6. The smallest faithful prototype at our geometry

The measurement branch already argued a tile-wide shared gather is contract-legal (union in ascending
block id keeps each query's list as a subsequence); design §8.10 names the same kernel. QUILT's
contribution to that plan is the **set-decomposition primitive**.

**What a tile-wide shared gather must do** (**(inference)** + facts note §6.3):

1. **Consume, don't change.** Read `selected`/`counts` from the Op workspace exactly as the attention
   kernel does (`qsa.cu:926, 942, 1141`); `qsa_select` is untouched. No clipping.
2. **Per (tile of `w` consecutive columns, kv_head)**: build the **union** `U = ⋃_{q∈tile} S_q` in
   **ascending block id**, plus a **per-query membership mask** over `U`. This is the SCSD family applied
   to produce a union + membership instead of a pairwise intersection/difference (Eqs 1–7 generalize:
   `U` is the entries with occurrence count `≥ 1`, membership is the per-query source label; the
   pairwise case is just `count = w`).
3. **Stream `U` in ascending order**: load each block's 4 token rows **once** per `(tile, kv_head)` via
   `cp.async`, shared across the tile's queries; for each query whose mask includes the block, fold
   `(K, V)` into that query's accumulator; queries whose mask excludes it are **masked out** — the
   paper's masking idea (§3.4), applied to *loading*, not to the selection.
4. **Keep everything per query**: its own online softmax state, per-head butterfly, its own 16-token
   warp tiles, and its **open tail** appended after the blocks (`qsa_prompt.cu:168-173`). Since `U` is
   ascending and each query's list is ascending, every query sees its selected blocks as a **subsequence
   in its original order** → the fixed-accumulation/width-invariance contract holds (facts note §6.3).
5. **`w` is the knob**: the tile-union curve says ≥30 % needs `w ≥ 2` (16K) / `w ≥ 4` (140K); `w = 8`
   gives 0.70 / 0.59.
6. **Cost caveat**: `U` cannot be resident — a selected 4-token block of one layer is ~4.2 KB INT8
   (design §19.3.11), and at w=16 the union is 1547 / 2634 blocks (6.5 / 11 MB); it must be streamed
   block by block into one accumulator per query. Prompt attention is 245–248 ms/call with an unmeasured
   gather share, so the prototype needs its **own Op-bench A/B** (an nsys attribution as in design
   §19.3.6 Q0).

**Which of the paper's two mechanisms at our geometry: the set decomposition (single-level union +
membership), not cascaded sharing.** **(inference)** Rationale: (a) the bankable quantity for us is the
**union redundancy**, which is exactly the processed-KV count of a *fully-recursed* cascade (Table 1),
so a single-level union already attains it; (b) our tile width is small (2–16), so the group-wide shared
core that the first cascade level extracts is tiny (0.24 / 0.11 at w=8) and most of the value is in the
residual/union levels anyway; (c) cascaded sharing's shared-first ordering would fight the
fixed-accumulation contract, whereas an ascending union does not. Cascaded sharing is a **later
refinement** (to keep union fragments aligned to our 16-token warp tiles) if the single-level version
underfills tiles — not the first thing to build.

---

## 7. Where the paper is silent / ambiguous (do not fill the gap)

- **Head granularity** of the overlap measurement and the selection (per query token vs per head).
- **Exact prefill sequence length / columns per batch** at which the 55.1 % is realized (Figure 2 only
  shows a breakdown).
- **KV quantization format** (dtype/bit-width) used in the evaluation (§4.1 states none).
- **Floating-point accumulation order / bit-exactness** of the "exact" steps — never discussed; the
  paper's "exact" is set equality.
- The **formula and denominator** behind Figure 1 (only the group-2 percentages and layer-1 value are
  given; I read it as `|⋂| / K` from the "shared by all queries" wording, but the paper does not print
  it).
- The **group definition** ("neighboring queries"): the axis/adjacency is not stated; I assume
  consecutive sequence positions (matching our consecutive query columns).
- The paper does not report **decode**-path behaviour (it targets prefill, §2.1).

---

## Provenance

- Paper: arXiv:2610.11134v1, read in full from the HTML (371 lines).
- Our facts: `research/qsa-prompt-overlap-facts` `30e70f87`; `proto/qsa-overlap-measurement` `b32895e3`.
- New measurement (§1.3): CPU-only, read-only over `out/qsa-index-capture/` (`16k-eager`, `140k-eager`;
  `selected.bin`/`counts.bin`), the measurement branch's parsing conventions; script described inline in
  §1.3. No GPU work; no capture re-run.
