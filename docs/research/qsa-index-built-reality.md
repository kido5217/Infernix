# The built QSA index: representation, placement, cost, and the Fathom premise

Findings for the map "bit-plane QSA index scan (issue #5 execution)". Every claim carries a
`file:line`. Nothing here was measured by this note; quoted timings are from in-tree records.

**Headline.** Issue #5's premise — a host-resident QSA index whose scan bounds decoding beyond
`--kv-host-tier` — does not describe this build. The pooled index plane is **device memory in every
space**, the host tier is **not implemented**, and the index is a **runtime activation**, not an
artifact object. Fathom's headline regime (a host scan) therefore has no target here; what remains
testable is whether a coarser index plane preserves selection and returns device bytes.

## 1. What the index is

- **Keys only.** Selection scores every complete 4-token block against a 4×128 query key and returns
  **block ids**; K/V of the selected tokens are read separately. Semantic contract:
  `src/ops/qsa/qsa_select.h:6-17`, `:54-57`.
- **Produced at runtime**, not converted: `ops::qsa_pool_keys` → `pool_kernel`,
  `src/ops/qsa/qsa.cu:1005` (launch `:1021`), writing `layer.pooled_pages`; the op contract is
  `include/infernix/ops/qsa.h:113-122`. Pooling is a **mean over the `R=4` raw keys, then RMSNorm,
  then RoPE** (`src/ops/qsa/qsa.cu:151-160`; the op contract `include/infernix/ops/qsa.h:21`; the
  independent oracle averages the same way, `tools/flash_next/reference.py:282`) — there is no
  squaring of raw keys.
- **Consumed** by `ops::detail::qsa_select` (`src/ops/qsa/qsa_select.h:62`), implemented by
  `qsa_score_kernel` (`src/ops/qsa/qsa_select.cu:106`) and `qsa_select_global_kernel` (`:688`),
  driven once per QSA layer per forward call from `ops::qsa_attention`
  (`src/ops/qsa/qsa.cu:1141`).
- **Representation.** BF16, one 128-wide key per 4-token block per QSA layer: plane layout
  `[Di/R, 64, 1, pages]` = `[32, 64, 1, pages]` at `R=4, Di=128`
  (`include/infernix/ops/qsa.h:85`, comment `:35-36`) — **64 B per token, 256 B per block per layer**.
  The plane is independent of `--kv-dtype` (design §8.10, `qwen3_8-flash-next-design.md:1155-1160`).
- **Score.** `score_b = (1/√Di) · Σ_h ReLU(⟨index_q_h, pooled_key_b⟩)`, ties to the lower block id,
  written ascending (`src/ops/qsa/qsa_select.cu:157,207-218`; contract `src/ops/qsa/qsa_select.h:10`).
- **Width.** Budget 2,048 tokens in 4-token blocks ⇒ **top = 512 blocks** (the relation
  `top = budget/ratio` is `src/ops/qsa/qsa_select.h:4,7,10`; the values are the geometry
  `src/models/qwen4_exp/execution/forward.cpp:72-85` and design §2 `:181`). There is **no
  top-16** anywhere in selection; 16 is not a QSA parameter. Routing width (top-10 of 512 experts per
  layer, 24,576 routed experts total) is a different subsystem (design §2, `:181-182`).
- **Blocks.** QSA's block is **4 tokens**; **64** is the paged-KV page size
  (`src/core/paged_kv_cache.h:20`), not the selection block.

## 2. Placement: the index is device-resident, and the host tier does not exist

- **`--kv-host-tier` is design-only.** The design lists it (`qwen3_8-flash-next-design.md:2709`,
  default `auto` = 65,536) and §9.6 states the enabling rule (`:1562-1564`), but **no field or parse
  case exists** in `apps/cli/options.h` / `options.cpp`, and `docs/cli.md` does not mention it. The
  deployed unit passes no such flag.
- **The index is never a host plane, even in the design.** §19.3.11 (the "as built" record) states:
  "The pooled index-key plane is device memory in every space: space 1's pooled plane is a device
  array, since selection scans every pooled key"
  (`docs/maintainer/qwen3_8-flash-next-design.md:8114-8115`). Only **K/V page data** moves to the
  pinned, GPU-mapped host pool (space 1, `:8110`), read zero-copy. §9.6 says the same
  (`:1549`: "Index-key plane: always fully device-resident").
- **Ops-layer support exists but is never populated.** `QsaPageSpaces`
  (`include/infernix/ops/qsa.h:61-80`) carries `host_k/host_v/lent_*` raw pointers with the space
  translation (`page_spaces.cuh:16-40`), and selection/pooling translate through `layer.spaces`
  (`qsa.cu:1029,1142`). But the Program constructs `QsaKVLayer` **without** setting `.spaces`
  (`src/models/qwen4_exp/program/program_impl.h:712-731`), so it stays at its default (all pages
  device, host pointers null). The only place `QsaPageSpaces` is filled is the op unit test
  (`tests/ops/test_qsa.cpp:1243-1319`). The pooled-plane lookup returns `spaces.pooled`, a **device**
  array, in every space (`src/ops/qsa/page_spaces.cuh:33-39`).
- **Separate machinery, easily confused.** The pinned host KV arena
  (`src/core/host_kv_arena.{h,cpp}`, `plan_host_kv_page_layout`) belongs to the **prefix cache** Host
  tier (`src/models/qwen4_exp/program/prefix/prefix_cache.cpp:29`), reached by `--host-context-mib`
  (default 4096). It is not a QSA index tier.

## 3. Where the scan cost lives

- **Kernel and granularity.** `qsa_score_kernel` + `qsa_select_global_kernel`, once per QSA layer per
  forward call (13 planes per lane: 12 attention layers + the MTP block; design §2 `:178,184`,
  §8.10 `:1169`).
- **Per element read.** One BF16 128-wide pooled key per complete block per column (256 B/block), plus
  the 4×128 BF16 query once per column (`src/ops/qsa/qsa_select.cu:137-144,160-184`). It scores
  **all** complete blocks (`n = (p+1)/R`) with no early exit; the only short-circuit is "if `n ≤ top`,
  select all" (`:130,341,698`). Selection is an exact radix select over all scores (`:498-760`).
- **Measured (device memory, in-tree).**
  - `qsa_score_kernel` + `qsa_select_global_kernel`: **12 µs/call at 8K, 24 µs at 128K**, `merge_kernel`
    3.3 µs (nsys, dense8m, `int8` KV; design §19.3.14, `:8506-8507`).
  - Q2 vs Q0 per layer call: decode select **12.3 / 13.8 / 23.7 µs** at 8K / 32K / 128K; prompt select
    0.20 / 0.45 / 1.74 µs per prompt token (§19.3.6, `:6604-6610`).
  - Aggregated at 256K: "QSA's index scans and host-tier fetches add ~0.16 ms per token"
    (design §4.2, `:529-530`) — a bundled figure, not an index-only measurement.
- **Bandwidth floor — a model, not a measurement.** Pooled keys are 0.5 / 2.1 / 8.4 / 16.8 MB at
  8K / 32K / 128K / 262K, modelled at **1.7 TB/s device** bandwidth ⇒ 0.3 / 1.3 / 4.9 / 9.9 µs; the
  score kernel is modelled at 2.5 / 3 / 6.5 / 11 µs (§19.3.6, `:6613-6617`).
- **The measurement does not match that model, and the scan is not demonstrably bandwidth-bound.**
  Measured per-call time is 12 µs at 8K and 24 µs at 128K, against the modelled floor of 0.3 and
  4.9 µs and the modelled total of roughly 8 and 13 µs — about 1.85× the full model at 128K. Bytes
  grow **16×** with context from 8K to 128K (pooled keys 0.5 MB → 8.4 MB, both rounded) while time
  grows only **2×**, which is the signature
  of a latency-dominated scan with a large fixed component, not of a bandwidth-bound one. Treat
  "fewer index bytes ⇒ proportionally less scan time" as **plausible but unevidenced**: the
  prototype must measure the scan cost, not assume it.
- **No host-memory scan measurement exists** — because the design deliberately keeps the index on the
  device. There is no in-tree evidence that the index scan is PCIe- or host-byte-bound.

## 4. Numeric representation and existing truncation

- **BF16 only**: 2 bytes per element, 256 B per block per layer (`include/infernix/ops/qsa.h:85`).
- **Nothing else exists**: searches for bit-plane / truncation / quantized-index / subsampled /
  adaptive-read-depth / early-exit over `docs src tests bench tools include` find no index-related
  implementation. The packed 4-bit storages (`k8v4`, `nvfp4`, `vq2`, `k4v2`) apply to **K/V**, not the
  index (design §8.10, `:1159-1160`).
- No quantized or subsampled index path, no top-k early exit, no per-query adaptive read depth.

## 5. Constraints on changing it

- **The index is not an artifact object.** The pooled plane is a runtime device page-pool plane
  (`program_impl.h:723`) recomputed by `pool_kernel` each call (`qsa.cu:1005-1030`). No container
  format, layout or tensor record describes it; the artifact stores only the indexer **projection
  weights and norms** (`src/models/qwen4_exp/load.cpp:99-104`; converter
  `tools/convert/qwen4_exp.py:366-374`).
- **Consequence.** A "keys stored as bit-plane pairs" change is **not** a converter change here; it is
  a runtime re-layout of the pooled plane plus every reader: `pool_kernel` (the writer, whose
  accessor and signature are `qsa.cu:126-132`, `:135-142`), the Program's plane allocation, the
  selection kernels (`qsa_select.cu:67-184`), the attention and prompt pooled-plane readers
  (`attention_kernel`, `qsa.cu:476`; `qsa_prompt.cu`), and the op tests. The design's converter statement in issue #5
  ("pooled keys stored as plane pairs") assumes the index is converted — it is not.

## 6. Harnesses

- **Timing**: `infernix_qsa_attention_bench` (`bench/ops/benchmarks.cmake:58`,
  `bench/README.md:751-763`) times the whole call and selection alone per storage/context/width:
  `./build/bench/infernix_qsa_attention_bench --storage bf16,int8,fp8,nvfp4,k8v4,vq2,k4v2 --contexts 8192,131072 --widths 1,5,8,256`.
- **Correctness**: `infernix_qsa_test` (`tests/ops/tests.cmake:238`, `tests/README.md:107-111`) — FP64
  oracles for index queries, pooled keys, tails, selection and attention (page spaces are covered in
  the test body rather than that README text, `tests/ops/test_qsa.cpp:1243-1319`);
  `infernix_qsa_select_test` (`tests/ops/tests.cmake:244`, `tests/README.md:113-121`) — selection.
- **Selection agreement: no harness exists.** The only in-tree comparison is exact equality of two
  `Selection` structs from the same index, inside `tests/ops/test_qsa_select.cu:508-545`. `--kl-reference`
  compares **vocab** top-k, not block selection (`apps/perplexity/main.cpp:387,414`). Measuring "does a
  truncated index preserve the selected blocks" requires new measurement code — but an **independent
  QSA reference already exists** and is the natural starting point: `tools/flash_next/reference.py:266-294`
  implements pooling and selection from the checkpoint (used by `block_check.py`,
  `upstream_check.py`, `compare_logits.py`). It scores upstream against Infernix, persists no selected-block
  sets and no pooled keys, and is therefore not the missing metric.
- The temporary `INFERNIX_TMP_QSA_TRACE` hook referenced by §19.3.11 lives on a branch, not in this
  tree.

## 7. Geometry and size at the deployed regime

- **Layers**: 48 (36 GDN + 12 QSA) + 1 MTP QSA block ⇒ **13 index planes per lane** (design §2
  `:178,184`; §8.10 `:1169`).
- **QSA**: 24 query heads, 2 KV heads, head dim 256; indexer query 4×128, key 1×128; budget 2,048
  tokens in 4-token blocks ⇒ top = 512 blocks (design §2 `:181`).
- **Index entries**: 262,144 blocks per layer per lane at 1M context; **3,407,872 keys per lane**.
- **Index bytes per lane**: 832 B/token × context ⇒ ≈ **104 MiB at 128K**, ≈ 208 MiB at 256K
  (= 0.218 GB decimal, matching design §9.6 `:1549`'s 0.22 GB), ≈ **832 MiB at 1M**. Host memory used
  by the index today: **0**.

## 8. Test configuration (unit `infernix-qwen38-flash-next`, `llm.nix:818-905`)

Artifact `/home/kido/ai/models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix`,
port **8828**, with `--ngram-volume`, `--kv-dtype nvfp4`, `--max-context 1048576`,
`--rope-yarn-factor 4`, `--prefill-chunk 4096`, `--spec mtp`, `--draft-tokens 4`, `--lm-head-draft`,
`--max-concurrency 4`, `--expert-ram-mib 66000`, `--vision`, `--preserve-thinking`,
`--reasoning-loop conclude`, `--default-thinking-budget 32768`, `--request-log-jsonl`,
`--max-pending-requests 64`, `--pending-timeout-ms 600000`, `--media-cache-mib 2048`. No
`--kv-host-tier` (it is not a flag); no `--host-context-mib` (default 4096, the prefix-cache pool).

## What this means for the evaluation

1. **The paper's regime is absent.** No host index, no host-tier flag. Fathom's "no gain when the
   index is device-resident" clause applies to this build as it stands.
2. **What remains testable is still worth testing — but the win is not yet evidenced.** The index
   plane's bytes can be cut by a coarser representation, and a smaller plane frees frames; whether
   the **scan** gets faster is *not* established: the measured curve (2× time over 16× bytes) looks
   latency-dominated, so the prototype must measure the scan rather than assume the byte saving
   converts into time. The accuracy question — does a *t*-bit prefix keep the selected blocks? — is
   independent of where the plane lives and is the primary thing to settle.
3. **The change is runtime-only**, touching the writer, the Program, every QSA reader and the tests —
   not the converter or the container.
4. **Two measurement gaps must be closed before any bar is set**: no selection-agreement harness
   exists, and the captured pooled keys are not recorded anywhere today.
