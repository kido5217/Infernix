# The QSA prompt path, its gather, and the exactness contract

Facts for the map "cross-query KV dedup in the QSA prefill (issue #2 execution)". Every claim carries a
`file:line`. Nothing here is measured by this note; quoted timings are from in-tree records.

**Headline.** The prompt path selects **per query column** and gathers **per query column**: one CTA per
`(column, kv_head)` copies only that column's attended rows, with no cross-query staging and no
per-chunk block list. The selection is already materialized in the Op workspace before attention reads
it, so a dedup or shared-gather pass can consume the same arrays without touching selection. The
exactness contract is a **width-invariance and fixed-accumulation rule** — sharing loads is allowed,
reordering a query's accumulation is not. The prompt attention dominates the compute, and no bench
separates its gather from its compute.

## 1. The path, per chunk

- Entry: `ops::qsa_attention` (`src/ops/qsa/qsa.cu:1070`), called once per QSA layer per forward call
  (`src/models/qwen4_exp/execution/forward.cpp:520-522`).
- Selection first: `detail::qsa_select` (`qsa.cu:1141-1142`) → `qsa_select` (`src/ops/qsa/qsa_select.cu:842`),
  kernels `qsa_score_kernel` (`qsa_select.cu:106`) or `qsa_score_wide_kernel` (`:314`), then
  `qsa_select_global_kernel` (`:688`). It iterates every column in groups of
  `kQsaSelectGroupColumns = 128` (`src/ops/qsa/qsa_select.h:48`, `qsa_select.cu:885`).
- Prompt route: with `attention_splits == 1` and `qsa_prompt_supported`, it dispatches to
  `detail::qsa_prompt_attention` (`qsa.cu:1145-1151`; `src/ops/qsa/qsa_prompt.cu:943`) →
  `qsa_prompt_kernel` (declared `qsa_prompt.cu:265`, launched `:896`). Otherwise the FP32
  `attention_kernel` (`qsa.cu:475`) plus `merge_kernel` (`qsa.cu:857`).
- **Per column**: the prompt kernel's grid is `(columns, kv_heads)` (`qsa_prompt.cu:895`), column
  `t = blockIdx.x` (`:283`), and it reads its own list at `selected + t*top_blocks` (`:291`).
- **The selection is materialized** as `selected` `[top=512, columns]` i32 and `counts` `[columns]` i32
  (`qsa_select.h:54-57`), carved into the Op workspace as the first two int32 arrays and sized from
  `budget/ratio` (`qsa.cu:922-943`). Attention reads them back (`qsa_prompt.cu:271` and `:168-173`;
  the FP32 path at `qsa.cu:496-518`).
- **The gather is per query**: each `(column, kv_head)` CTA copies only its column's attended rows
  (`qsa_prompt.cu:339-446`). There is no shared staging tile and no per-chunk block list. The design's
  union-of-selected-blocks alternative with a per-query mask is described
  (`docs/maintainer/qwen3_8-flash-next-design.md:1200`) but is **not built**.

## 2. The gather

- Source: the paged planes, whose page id is resolved across the three spaces by `qsa_space_plane`
  (`src/ops/qsa/page_spaces.cuh:16-29`) — space 0 the device pool, space 1 the pinned GPU-mapped host
  pool (read zero-copy), space 2 a device pool lent from expert frames during a long prefill; the model
  is design §19.3.11 (`qwen3_8-flash-next-design.md:8107-8123`).
- Read: **`cp.async` in 16-byte chunks** (`qsa_prompt.cu:367`, `:412-413`; the same in the FP32 path
  `qsa.cu:586-636`). No TMA.
- Unit: one **token row**, not a page and not a 4 KiB block. `Row<Code>::Bytes` (`qsa_prompt.cu:44-68`)
  is 528 B per token per KV head in INT8 (K 256 + V 256 + scales 8+8) and 1024 B in BF16. Selection's
  unit is a 4-token block; a column selects the top **512 blocks = 2,048 tokens**, plus the open tail.
- **Scoring already shares reads** across columns — a warp reads 8 pooled keys once and scores them for
  up to 16 columns (`qsa_select.cu:102-104`, `:185-205`, `:227-252`; design §19.3.6 `:6581-6583`,
  `:6625-6628`). That sharing is upstream of the gather; the gather itself shares nothing.

## 3. The exactness contract

There is **no "§6.3 order-sensitivity" section** in this repo (design §6.3 is *Residency classes in the
v3 container*, `qwen3_8-flash-next-design.md:748`) and none in the QUILT paper (its §6 is the
conclusion). The rule that governs this path is the QSA **width-invariance / fixed-accumulation**
contract:

- **Selection scoring order** — `src/ops/qsa/qsa_select.h:10-17`: lane partials are summed in
  `warp_sum`'s xor-butterfly order (offsets 16, 8, 4, 2, 1), the heads' ReLUs are added in head order,
  and "Scores, ids and counts are therefore bitwise equal to the original one-CTA-per-column kernel's."
  Implementing code: `qsa_select.cu:193-218`; the wide kernel's bit-reversed leaf tree (`:277-309`).
- **Attention accumulation** — `qsa.cu:470-473`: "an exact maximum, then l summed serially in token
  order and the in-order value fold, under the fixed split partition"; `§19.3.14:8574`: "No
  per-(head, dimension) order changes, so outputs equal the previous kernel's."
- **Width invariance** — `src/ops/qsa/qsa_prompt.h:5-10`: a column's FP32 result must not depend on the
  call's width, "the prefix cache relies on it: a resumed prompt may compute a position in a call of
  another width". Design §19.3.8 `:7730-7734`; a width-dependent split rule was tried and reverted
  (§19.3.15 `:8790-8799`), and `test_qsa` checks a 100-column call reproduces a wider call bit for bit.

**Consequence for any dedup**: sharing the *load* of a block between queries is permitted; changing the
order in which a query accumulates its attended tokens (ascending block id, then the tail —
`qsa_prompt.cu:168-173`) or its per-head butterfly is not.

## 4. Costs (measured in-tree; conditions with each)

| Quantity | Value | Source |
|---|---|---|
| Prompt selection, per prompt token | 0.20 / 0.45 / 1.74 µs at 8K / 32K / 128K | §19.3.6 `:6604-6606` |
| One 4,096-column selection (score+select) | 1.119→1.006 ms at 8K; 14.04→11.44 ms at 250K (after Q3) | §19.3.6 `:6637-6639` |
| QSA `attention_kernel` (FP32 SIMT), per 4,096-token call, late calls ≥100K | 245 ms, 248 ms | §19.3.8 `:7581`, `:7604` |
| QSA selection per 4,096-token call at 128K | ~85 ms (73 ms after Q3) | §19.3.8 `:7586`, `:7805` |
| Prefill at 245,760 tokens (int8, no YaRN) | 8.2K tok/s | §19.3.18 `:9033-9035` |
| pp32k / pp128k prefill wall (after batched copies) | 5.297 s / 17.03 s | §19.3.17 `:8898-8899` |

Modelled, not measured: the per-stage split inside selection (§19.3.6 `:6613-6621`) and the FP32
attention's 30–60 ms target (§19.3.8 `:7759`).

**No in-tree measurement separates the prompt gather from its compute.** `infernix_qsa_attention_bench`
times the whole `qsa_attention` call and `detail::qsa_select` alone, so their difference is
attention+merge (`bench/ops/qsa_attention_bench.cu:228-236`; `bench/README.md:751-763`). Any dedup win
claim therefore needs an Op-bench A/B of the attention stage, ideally with nsys attribution
(§19.3.6 Q0 used nsys for exactly that).

## 5. What the recorded selections already hold

On `proto/qsa-index-capture` (head `05f74f1a`; layouts in its `docs/research/qsa-index-capture/README.md`;
data at `out/qsa-index-capture/`):

- `selected.bin` is the full `[512, columns]` i32 block-id array for **every column of every recorded
  prefill chunk** — 8,388,608 B for a 4,096-column chunk.
- Coverage: **16K** — six chunks of 4,096/4,096/4,096/4,096/4,096/1,946 columns at positions 0…22425,
  12 layers each, 72 selection captures; **140K** — three chunks of 4,096/4,096/2,432 at positions
  131,072…141,695, 12 layers each, 36 captures (the run armed on `MIN_POS=131072`, so the prompt's
  earlier chunks are absent).
- Dense columns (`counts = -1`, where `n ≤ 512`) carry no meaningful ids: 24,806 at 16K, 0 at 140K.
  While `p+1 ≤ 2,051` tokens the selection short-circuits and QSA equals dense causal
  (`include/infernix/ops/qsa.h:33-34`; §8.10 `:1197-1199`).
- Layer identity is the plane's `pooled_ptr` (13 distinct across a run; 12 in a text prefill chunk —
  the MTP plane appears in decode steps).
- **The cross-query overlap has never been computed**: `verify_capture.py` checks a capture's ids
  against a recomputation and `--sweep` aggregates per manifest; neither aggregates selections across
  columns. The truncation prototype's agreement metric (`proto/qsa-index-truncation`,
  `truncation_probe.py:273-281`) is full-vs-truncated **within** one column, not across queries.

## 6. What this means for the evaluation

1. **The overlap ratio is measurable with no new GPU run**: `selected.bin` holds every prefill column's
   ids, so a CPU-only script can compute the distinct-versus-per-query-sum ratio, its decay with query
   distance, and its per-layer spread. The 140K coverage is the last ~10.6K tokens — the ≥128K regime —
   which is what the issue's question is about.
2. **The gather is separable**: selection writes `selected`/`counts` into the workspace before attention
   reads them (`qsa.cu:926`, `:942`, `:1141`), so a shared-gather pass consumes the same arrays and
   leaves `qsa_select` untouched.
3. **A dedup needs a new attention gather and must respect §3's contract** — share loads, never reorder
   a query's accumulation.
4. **Prompt attention (245–248 ms/call) dwarfs selection (73–86 ms/call), but its gather share is
   unmeasured**, so the prototype's timing leg needs its own A/B rather than the existing numbers.

## Flags

- The issue's "§6.3 order sensitivity rule" does not resolve in either document; the contract above is
  the repo's nearest authority and is what a prototype should be held to.
- The design says the prompt route needs "86+ columns" (§19.3.8 `:7728`) while the code's threshold
  gives 43+ (`qsa_prompt.h:5-6`, `attention_splits` `qsa.cu:888-901`) — a doc/code tension unrelated to
  this evaluation.
- The design's union-of-selections prompt kernel (§8.10 `:1200`) is unimplemented; it is the design
  intent a dedup prototype would be reviving, not an existing mechanism.
