# Basalt: expert tier and model-file techniques — assessment for Infernix (ticket #58)

Basalt clone: `/home/kido/trash/temp/opencode/basalt` (commit 2ef470c). Authoritative container spec:
`include/basalt/core/model_format.hpp`. Infernix comparison: `/home/kido/network/projects/Infernix` (main).

## Basalt mechanisms

### 1. Second GPU as a peer expert tier (`--gpu 0,1`, `--peer-device N`, `--peer-exclusive`)

- `include/basalt/core/peer_experts.hpp:1-16` — design: the primary card keeps everything it had (dense
  weights, KV, MTP drafter, its own adaptive expert tier, the captured verify graph); the peer adds a second
  tier of routed-expert slots only. From the verify graph's view nothing changes: every non-resident expert
  is a "CPU row" read from mapped host memory; the peer computes the rows whose experts it holds with the
  **same kernels and activations as the primary** (q8_1 + `native_expert_grouped`), so an expert yields
  bit-identical rows on either card (`peer_experts.hpp:11-13`).
- Fill and residency: `peer_experts.hpp:47-56` (`open` takes the profile-ranked pair list minus what the
  primary holds, sized to free memory minus a reserve); `:60-72` residency table / `slot_ptr` / `p2p()`;
  `:74-80` per-layer launch/finish writing rows straight into portable pinned host rows.
- Adaptivity: `:82-103` — swaps hottest experts neither card holds into the peer's least-used slots between
  verify windows; `set_exclusive` (`:91-95`) implements `--peer-exclusive`: the peer's experts have **no RAM
  place**, so an evicted victim is copied to a free RAM slot before its peer slot is reused, and slots are
  kept via `hold`/`copy_to` over P2P; `reassign` (`:101-103`) moves an expert peer→primary when the primary
  evicts one.
- Wiring: `src/program/generate.cpp:792-794` (`--peer-device N` help; `--expert-cache-device1..3` fill
  CUDA1..3 as extra tiers), `:1462,1471` (`--pcie-frac`, `--pcie-mode` — share of experts placed on the
  PCIe/P2P side), `:3776` (`--peer-device` requires an expert profile and the expert cache), `:2580-2620`
  (`--split-skip-if-fits`: if CUDA0 alone holds every profiled pair plus the session, skip the split).
  Layer *splitting across GPUs* (a layer's experts served by both cards) needs the profile
  (`generate.cpp:1738-1739`). Config: `"gpu": "0,1"` in the server section (`README.md:86`),
  `--peer-exclusive` as an engine flag (`README.md:141-142`); the vision sidecar can also live on the peer
  card (`README.md:188,199`). The tier byte accounting warns when >2% of experts are file-only and names a
  peer GPU as a lever (`generate.cpp:4680-4693`).
- Measured effect (`docs/precision.md:116-122`): with tiers pinned (primary N=11582 + peer M=8876 slots on a
  5090 + 5060 Ti), spec identity holds bitwise across both cards — the peer path is output-neutral; the
  value is capacity (more resident experts ⇒ fewer SSD reads), not arithmetic.

### 2. `experts` + `experts.layout` — per-layer expert layout

- `include/basalt/core/model_format.hpp:21-22` — `experts` is the routed blobs; `experts.layout` is a text
  section (`# experts v3`) with offsets relative to `experts`. Necessary because GGUF packs mix quantizers
  per layer (`docs/precision.md:34`: gate/up types vary across layers; down Q2_0 on 30 layers, IQ4_NL on 18).
- `include/basalt/kernels/cpu/expert_layout.hpp:18-31` — the resolved table: per-layer `NativeFmt` (gate/down
  types), `offset`, per-blob `bytes`, `max_blob`, `total`; `blob_bytes/layer_offset/blob_offset` are the only
  addressing any tier (arena, VRAM, CPU pool, prompt path) uses.
- `src/kernels/cpu/expert_layout.cpp:135-204` — parse and checks: unknown `vN` headers are **refused**
  (misread columns), `n_expert` can be overridden from the header (pruned packs ship fewer experts than the
  compiled geometry), each line `layer gate_type down_type offset blob` is validated against the formats'
  computed byte size, layers must be exactly contiguous, and the total must match the section.

### 3. `profile` — the expert profile

- Format (`include/basalt/core/expert_cache.hpp:30-39`): `version, n_layers, n_expert, slots, n_ranked`
  (u32 LE), then `n_ranked` `(layer, expert)` pairs (u16 each) **ranked by descending routing frequency**
  over a representative trace, then an `n_layers×n_expert` i32 rank table (−1 = unranked). What it measures:
  expert routing frequency only; the header records `slots` (the trace's tier size) and is rejected on a
  model-shape mismatch (`src/core/expert_cache.cpp:15-60`). The frequency table itself is deliberately not
  returned — hit rates are measured by running, not re-derived from the file (`expert_cache.hpp:36-37`).
- Production: converter input `--profile FILE` / `--profile-from TRACE` (`README.md:230`,
  `tools/convert/convert.py:4,17`); the engine can *write back* what the adaptive tier learned
  (`--expert-profile-save`, `generate.cpp:424-428,5756-5760`), ranked resident-first then counted heat over
  the prior (`expert_cache.hpp:41-56`; atomic tmp+rename `expert_cache.cpp:99-135`) — "a start from it
  begins where this one ended".
- Runtime consumption (`src/program/generate.cpp`): fill the VRAM tier in profile order
  (`:2713-2745`, incl. moving part of the hot set to the peer); copy profile-hottest experts into the RAM
  copy, stream the rest from the file (`:390`); `--resident-cpu-experts` needs mmap + a profile (`:1712-
  1713`); layer split across GPUs needs it (`:1738-1739`); `--split-skip-if-fits` (`:2580-2620`); graphed
  hit path and serve need a fillable cache, graphed hits need the profile (`:4700`); prefill borrow sizing
  prefers profile order (`:1765`).

### 4. `hc8` — 8-bit hyper-connection copies

- Which tensors (`src/core/verify.cpp:564-577`): per layer, `blk.N.{hc_attn_,hc_ffn_}{down,up,inject}.weight`
  plus `output_hc_down/up.weight`; the drafter's copies are `mtp.`-prefixed (`include/basalt/core/model_file.hpp:17-18`).
- What it is (`model_file.hpp:12-18`, `model_format.hpp:29`): a separate `hc8` section (magic `BASALTH8`)
  holding per matrix the int8 codes plus **one fp16 scale per 32 consecutive inputs** (`scale =
  f16(absmax/127)`, codes clamped ±127; copy size `rows*cols*9/8`, `verify.cpp:578-579`). Produced offline by
  `basalt generate --dump-hc8` (`generate.cpp:287,652-653,1330`), embedded by the converter as `--hc8 FILE`
  (`convert.py:17,238-239,303`); loading the copies is bit-identical to quantizing the BF16 matrices at
  every start, so it removes a per-start quantization pass and lets the BF16 originals stay out of VRAM
  entirely (`--hc8-only` path: `generate.cpp:2179-2187,2263-2312` — "held only as 8-bit copies (X MiB of
  bf16 left out of VRAM, Y MiB of copies)").
- Why (`docs/precision.md:16-17,41,55,191-223`): the one exception to the no-requantization rule. The hc
  down projection (10240→320) has heavy cancellation; llama.cpp's Q8_0 *activation* quantization turns 0.2–
  0.4% per-element input error into ~1% mixed-stream error at layer 0 (numpy check, `precision.md:199-201`),
  inflating every KLD by 0.01–0.02. Basalt's read keeps activations fp32 and quantizes only the **weights**
  to int8 (exact converts, fp32 fmaf per group; `precision.md:55`), matching exact math to 1e-5; measured:
  Basalt Q8_0 pack scores 0.0233 KLD vs an F32-hc base — the two-implementation noise floor (`precision.md:203-212`).
  Prefill still reads BF16 originals (`precision.md:97`). Parity harness: `src/kernels/hc8_parity.cpp:1-13`.

### 5. `.basalt` container framing

- `model_format.hpp:36-60` — magic `BASALTMF`, version 1; a fixed **64 KiB header**: 64-byte `FileHeader`
  (magic, version, n_sections, total file size for a truncation check, reserved) + a table of up to
  **511** ( `(64KiB−64)/128` ) 128-byte `SectionEntry` records: NUL-padded name (64 B), offset, bytes,
  **SHA-256** (32 B), reserved. Every section starts at a multiple of **2 MiB** (`kSectionAlign`) so it can
  be mmap'd, read with O_DIRECT, and backed by huge pages; inter-section bytes are zero; inner tensor
  alignment 4 KiB. Sections are opaque; content list at `model_format.hpp:12-31` (meta JSON, tokenizer/*,
  dense(+index), experts(+layout), gguf view, mtp/*, profile, hc8, vision/*).
- `src/core/model_file.cpp:263-311` — reader checks magic, exact version, section-table length, file
  truncation against `file_bytes`, and per-section alignment/containment. The stored SHA-256 is **not**
  verified at open — it is provenance/integrity data (converter records a cheap first/last-64 MiB+size
  fingerprint per source file, `tools/convert/convert.py:68-70,253`).
- Non-requantization byte-copy rule (`README.md:209-223`): tensors are copied byte-for-byte from the source
  GGUFs; the single exception is `--compat-bf16` (converts small projections the engine reads as BF16,
  recorded in `meta`) — plus the `hc8` copies above. The `gguf` section serves non-expert tensors through a
  virtual GGUF view (`model_file.cpp:371-403`), so GGUF-reading code runs on the container unchanged.

## Infernix today

- **Contract.** One GPU, one resident model, fixed execution lanes at startup
  (`AGENTS.md` "Product and architecture"); no second-device concept anywhere in runtime or model code.
- **Container.** v3 (`docs/maintainer/artifact-container.md`): 32-byte binary header + a complete JSON
  directory (components/objects/bindings/uses/files, §3-§4); unbounded object table; per-object
  `format`/`layout`/`shape`/`offset`/`bytes` (§5.1) from the closed registry (§6.1 — includes
  `q8_g32_fp16`: 8-bit codes, G32, FP16 multiplier); sharded at a default 32 GB with each segment's payload
  start 4096-byte aligned "which suits direct I/O" (§2.3); tensor object alignment 256 B (§5.3). No
  per-object content hash; integrity is exercised by the convert-time verifier, which compares every stored
  tensor/expert/n-gram row against the source (`docs/qwen3_8-flash-next.md:80-85`).
- **Expert tiers.** VRAM cache + pinned-RAM copy + SSD-in-artifact reads
  (`docs/qwen3_8-flash-next.md:227-247`): startup fills slots, warm start loads the top of the saved ranking
  from `<artifact>.expert-state` (`src/models/qwen4_exp/program/expert_cache/expert_state.cpp:13-16` — magic
  `NINFQ4ES`, usage counts + ranked keys; written on stop and every 10 min, runtime-learned only), RAM takes
  the next rank, misses stream from the artifact and feed the CPU expert engine. **Streaming assumes one
  uniform record pitch**: `src/models/qwen4_exp/expert_store.h:20-27` (`record_stride`, segments 4 KiB-aligned
  in offset and size for unbuffered reads), `execution/expert_stream.cpp:12` (kAlign 256, unaligned records
  rejected); `memory_plan.cpp` sizes uniform 2.64 MiB frames (`docs/qwen3_8-flash-next.md:148-151,260-265`).
- **Hyper-connections.** The fused mix route serves `Q8_G32_FP16` RowSplit *and* BF16 weights with FP32
  split-K partials (`src/ops/hyper_connection/hyper_connection_mix_fused.h:2-3,27`); official recipes B/C
  store the hc mixers (down/up/inject, final mixer) in `q8_g32_fp16` at conversion time, explicitly on the
  grounds that a 32-group scale with 8-bit integer codes is finer at 1.06 B/weight
  (`tools/convert/official_recipes.py:320-342,352,373-377,398-399`).
- **Expert layout.** Every artifact is converter-produced with a uniform expert bank per layer
  (`official_recipes.py:303-316`: all `/moe/experts` objects one `nvfp4_mul`/`nvfp4_expert_rg16_v1`), so
  heterogeneous per-layer expert formats are expressible in the JSON directory (one object record per layer,
  own format/layout) but not exercised today.

## Port assessment

1. **Peer-GPU expert tier (`--peer-device`, peer-exclusive).** Cleanest external design Infernix lacks:
   dense/KV/graphs stay on the primary; the peer is pure extra expert capacity, output-neutral by shared
   arithmetic. But it multiplies device buffers, streams, events, P2P portability (`cudaHostAllocPortable`),
   and a third eviction path (peer↔RAM↔file) — all against a stated one-GPU contract.
   *Collision: yes — product contract.*
2. **Shipped expert profile (`profile` section).** Infernix's warm start is runtime-learned only; first run
   and post-recrash/rebuild runs are cold, and the RAM-fill order and SSD-warning all consume a ranking they
   currently derive from the last session. A converter/calibration-produced, frequency-ranked seed slots
   exactly into the existing expert-state ranking pipeline (seed `expert_state`'s ranked list when the
   sidecar is absent, or a resource object in the artifact). Basalt's extra discipline is worth copying:
   validate header shape against the model, refuse unknown versions, and never trust the file's claimed hit
   rate — measure. *Collision: none* (a format/name addition inside existing contracts).
3. **`experts.layout` per-layer layout.** As a *section* it is redundant — v3's per-object records already
   carry format/layout/offset/bytes per layer, plus reference validation. The port-relevant kernel is the
   *validation set* (version refusal, blob-size-vs-format check, exact contiguity) and the real gap it
   exposes: Infernix's streamed tier requires a uniform `record_stride`, so a per-layer heterogeneous
   mixed-format pack cannot stream on the SSD tier today. If such packs become a requirement, the fix is
   per-layer strides/segments in `ExpertStore`/`expert_stream` — model/runtime work, no container change.
   *Collision: none; not needed now.*
4. **`hc8`.** Infernix already gets the whole benefit a different way: recipe B/C store the hc mixers
   directly in `q8_g32_fp16` — the same scheme (int8 codes, fp16 scale per 32) — and the fused Op reads them
   in place with fp32 activations. Basalt needs the `hc8` *copy* only because its byte-copy rule forces the
   BF16 originals into the pack; a section + parity test + per-start adopt path would be redundant
   machinery here. Two transferable nuggets: (a) Basalt's measured rationale for *weight-only* int8 with
   fp32 activations in the hc read (activation quantization inflates mixed-stream error ~1% at layer 0;
   `docs/precision.md:191-223`) independently confirms Infernix's recipe-B choice and argues against any
   future A8-for-hc shortcut; (b) `--dump-hc8`'s "quantize once offline, ship the codes" pattern is what
   Infernix's converter already does at recipe time.
   *Collision: none; mechanism redundant.*
5. **Container framing.**
   - 64 KiB fixed header / ≤511 named sections: Infernix's JSON directory is self-describing and unbounded;
     the binary header buys nothing here. Reject.
   - Per-section SHA-256: cheap provenance/integrity anchor the v3 catalog lacks (its hash-free design leans
     on convert-time verification). Basalt itself only *stores* the hashes (not checked at open), so the
     port would be: optional per-object `sha256` in the object record, checked lazily by the materializer /
     verify tool, not on every open. Low risk, modest value.
   - 2 MiB section alignment for mmap/O_DIRECT/huge pages: v3 aligns segments to 4096 for direct I/O; the
     spec already permits alignment gaps ("gaps between objects … have no load semantics", §5.3). A writer
     policy that 2 MiB-aligns the streamed expert-bank objects lets the SSD tier mmap huge pages and issue
     ≥2 MiB unbuffered reads — directly extends the tier that reads 60+ GiB. Port candidate (writer policy
     + optional mmap read path; reader keeps working since only layout-mandated alignment is normative).
   - Byte-copy, no-requantization rule: this is Basalt's answer to "packs arrive as third-party GGUF in
     exotic i-quant types". Infernix's converter repacks by recipe, and where byte-exactness matters it
     already does it (recipe C imports NVFP4 codes/scales bit-exact,
     `official_recipes.py:349-351`). Adopting the blanket rule would fight the closed-format registry.
     Reject as a rule; keep the practice where the source already matches a registry format.

## Recommended verdicts

| Candidate | Verdict | Reasoning |
|---|---|---|
| Peer-GPU expert tier (`--peer-device`, peer-exclusive, P2P rows) | **needs-product-change** | Directly contradicts the one-GPU contract; everything else about the design (primary untouched, same-kernel bit-identical rows, profile-ordered fill, exclusive RAM/peer hand-off) is a sound blueprint if multi-GPU ever becomes a product goal. Do not port silently. |
| Shipped expert profile (`profile` section semantics) | **port** | Fills a real Infernix gap (cold first runs, missing-ranking RAM fill order); fits existing contracts as a converter-produced ranking that seeds `expert_state`; adopt Basalt's shape validation + "measure, never re-derive" stance. |
| `experts.layout` as a container feature | **reject** (note: per-layer `record_stride` in the streamed tier if heterogeneous expert packs ever ship) | JSON object directory already generalizes it; validation lessons transfer at near-zero cost, uniform-stride limitation is model code, not container. |
| `hc8` section + copies | **reject** | Recipe B/C's `q8_g32_fp16` hc storage achieves the same VRAM/quality outcome through the closed registry; Basalt's copy exists only to dodge its own no-requantization rule. Record Basalt's precision.md hc evidence (fp32 activations, weight-only int8) as external confirmation of the current choice. |
| SHA-256 per section/object | **port (optional, low priority)** | Small catalog addition; useful for shipped-artifact integrity and the verify tool; skip open-time verification cost, verify lazily or in tooling. |
| 2 MiB alignment of large streamed objects | **port** | Writer-policy change inside the v3 spec; strengthens exactly the tier where Infernix streams tens of GiB (huge-page mmap, wide O_DIRECT). |
| Blanket no-requantization byte-copy rule | **reject** | Collides with Infernix's recipe/repack converter model; bit-exact import already happens where source format matches registry. |
