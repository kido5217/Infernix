# Basalt expert-cache-lent prefill chunking → Infernix port assessment (ticket #55)

Basalt (`/home/kido/trash/temp/opencode/basalt`, https://github.com/jesdga95/basalt) is a fork of
Strata stripped to Qwen3.8-Flash-Next on sm_120 (README.md:11-13) — the same model family Infernix
serves as `Qwen4ExpForCausalLM`, with the same VRAM-expert-cache + host-RAM backing. All line refs
below are from this read-only clone; Infernix refs from `main` at `d3414d6f`.

## Basalt mechanism

**Idea.** The batched prompt path's device buffers (activations, GEMM scratch, the streamed-expert
ring) are not reserved VRAM: they are bump-allocated inside a region carved from the **last slots of
the expert cache**, whose experts are marked non-resident for the prompt's duration and copied back
after it. `--prefill auto` then picks the largest chunk whose buffers the cache can lend.

- `--prefill` parsing — `src/program/generate.cpp:1439-1449`: `auto` sets `prefill_auto` +
  `prefill_auto_fit` (ceiling 32768); `auto:N` (N ∈ 8192/16384/32768) sets the operator ceiling;
  a bare number is a fixed chunk. `--no-prefill-borrow` (generate.cpp:1449) reserves buffers
  normally and forces fixed 2048 chunks when there is no expert profile to size a loan (1765-1767).
- **Carve-out API** — `include/basalt/prefill/prefill.hpp`: `init(..., void* borrow, uint64_t
  borrow_bytes)` lays every prompt-path buffer in the borrowed region ("the top slots of the expert
  cache, lent for the prompt and refilled after it"); `relayout(chunk, borrow, ...)` re-lays the
  buffers per request "so a short prompt lends only what it needs"; `bytes_needed(g, ss, chunk)`
  prices a chunk exactly (same allocation sequence as `init`, counted — `src/prefill/prefill.cpp:
  1115-1151`), monotone in T; `ring_slots_for(chunk)` gives the streamed ring's slot count
  (prefill.cpp:90, 1153); the ring lives inside the loan, so a shallower ring lends fewer cache
  slots (`set_ring_depth`, prefill.hpp).
- **Auto sizing** — `plan_lend` (generate.cpp:4481-4496) + `biggest_chunk` bisection on a
  256-token grid (4472-4480): the largest chunk whose `bytes_needed` → slot count `k` satisfies
  `k + 128 ≤ cache.slots()` (128-slot floor) and `k·100 ≤ kAutoLendPct·slots` where the cap is
  **90%** of the cache if ≥90% of expert bytes are pinned in RAM, else **85%** (4464, pinned share
  computed 4420-4428). Ceiling `auto_ceiling = max(8192, min(operator N, max_context))` (4466);
  bare `auto` without `--resident-experts` stays 8192 (4467). A fixed chunk halves until it fits
  its own cap, else uses private buffers. Multi-GPU/layer-split: each stage lends the tail of **its
  own** cache; the chunk is the largest every participant can fund, with a warning naming the
  capping card (4729-4812); a stage with spare VRAM keeps private buffers and lends nothing
  (4829-4844).
- **`--resident-experts` interaction** — the RAM complement is built *knowing the lend region*
  (generate.cpp:4544-4549 passes `lend_from` into `pin_cache_complement`;
  `src/core/expert_source.cpp:728-734, 931, 947-951`): complement occupancy of the lendable tail is
  maximized from its end. Bare `auto`'s ceiling is then re-cut to what the lend can fund:
  `allow = resident_lent_slots + kPromptFileReadBytes/max_blob` — RAM-backed lent slots plus a
  fixed 3.5 GiB of file reading per prompt (`kPromptFileReadBytes`,
  `include/basalt/platform/memory.hpp:43`; generate.cpp:4607-4613). A lent slot whose expert has no
  RAM copy is refilled from the file.
- **Per-request lend/refill cycle (serve)** — `lend` (generate.cpp:7571-7648): for each prompt
  segment, per-participant `first = slots − part_slots(want)` (want = min(segment rounded to 256,
  cap)), `relayout`, then every residency row with `host_res ≥ first` moves to `kNotResident` and
  the table is uploaded; during the chunk those experts stream through the ring from the RAM arena
  or the file. `refill_issue`/`refill_wait` (7519-7546) copy each lent slot's expert back into the
  **same slot** from the host copy (`fill_slot_queued`) and restore the table — done before any
  decode/verify window reads, so windows always see the full cache (comment 7511-7513). With
  `--batch` slots decoding, the loan is capped at `kLendCapDecoding = 1216` slots, restricted to
  RAM-backed slots, ring depth `kRingDecoding = 64`, and the chunk shrinks to the loan
  (4759-4760, 7593-7601); chunks alternate with decode rounds (`decode_share = 0.5`, ~7650).
  Vision "dynamic" mode reuses the identical lend/refill cycle per image (generate.cpp:6262-6320;
  README "Vision" section). The one-shot generate path logs `N lent slots refilled in X ms`
  (~8746).
- **Equal-chunk rule** — bare `auto` reads a prompt in the **fewest equal chunks** of at most
  **12288** tokens (**16384** with `--peer-exclusive`), because every chunk streams the experts and
  a short tail costs a whole pass (generate.cpp:7584-7591, 7603-7608; README.md:147-156). Basalt
  notes chunking changes GEMM tiles and thus a prompt's K/V rounding — answers can differ from
  another chunking at greedy near-ties.

## Infernix today

- `--prefill-chunk` is a fixed startup constant: `src/serve/serve_options.cpp:461-463, 806-807`
  (positive multiple of 128; default 1024, `include/infernix/types.h:337`); INT8/fast-kernel KV
  rounds it down to whole prompt-attention waves, 896 tokens at 24 heads on the 5090
  (`docs/serving.md:1175`). It is baked into `DeviceLayout::plan_device`
  (`src/models/qwen4_exp/program/program_impl.h:414`): `columns = max(chunk, lanes·max_width)`
  (:417) sizes the per-call io arrays (:459-476), the residency table (:560) and the static work
  arena `static_columns = max(lanes·W, min(chunk, 512))` (:487, `kStaticWorkColumns` :2437);
  `wide_work` (:489) is the chunk-sized arena swapped in for wider calls. Startup shortage errors
  name `--prefill-chunk` as a lever (program_impl.h:616;
  `src/runtime/engine/qwen4_exp_instance.cpp:244`).
- Prefill call planning: `prefix/call_plan.h:45-51` — calls at `frontier, frontier + chunk, …`
  with exact taps splitting; the tap ladder's defaults are defined from the chunk
  (`max(4096, 2·chunk)` base, `max(1024, chunk)` gap — `include/infernix/types.h:255-258`,
  `docs/serving.md:1222-1223`). One prefill turn and one compact decode batch per cycle
  (`docs/maintainer/resource-scheduling-and-context-cache.md` §5.1); permits are **typed KV
  coverage**, not expert frames (§3.2); chunk boundaries are the rotation/preemption points
  (§5.2, lines 175-179). Replay splits at every split the ledger recorded, so per-request chunkings
  replay bit-consistently (§5.3).
- **Frame lending already exists and the prefill path already uses it.** `kLoaned` frame state and
  `give_back` (`src/models/qwen4_exp/program/expert_cache/expert_cache.h:156-171`),
  `ExpertResidency::lendable/lend/give_back` (`expert_residency.h:119-127`,
  `expert_residency.cpp:563-623`): `choose_run` takes the lowest-LFRU contiguous run, skips loaned
  frames, avoids in-flight load targets; lend evicts (no write-back — experts are clean) and
  shrinks policy capacity; give_back restores it and the LFRU/deferred-load controller **re-admits
  evicted experts lazily**. The streamed ring is "lent by the expert cache for the prompt and
  returned when no lane is prefilling" (program_impl.h:2913), and `enter_wide`
  (:3004-3033) takes the wide-work arena from the stream lease's tail or a frame lease, falling
  back to a private allocation with a warning; `return_wide_work`/`return_stream_lease`
  (:3036, :3047) give them back.
- The design authority already ranked this mechanism: `docs/maintainer/qwen3_8-flash-next-design.md`
  §9.2 states the deliberate divergence — "Strata re-copies [lent experts] eagerly after every
  prompt, which costs short prompts up to a third of their speed" — and lists prefill-arena loans
  as "not planned yet" (status 2026-10-04); §13 (lines ~1943-1948) *is* the Basalt rule as the
  target design: "max is the largest chunk, searched in 1K steps, whose arena can be lent from
  frames while keeping a floor of resident experts; it is 16K-32K for a solo long prompt. A short
  tail (<1K) … stays separate. architectds/Strata measured +21-59% at 9K-100K from these two
  rules." Infernix's own chunk table (design doc, chunk 1024/2048/4096): pp4096 **413 → 659
  tok/s** while tg512 61.46 → 58.63 and frames 8,791 → 8,300; "lending frames to the prefill arena
  would remove the trade-off." The Strata re-read notes (design doc:260, 300-331) add: ring slot ≈
  0.53 tok/s vs chunk token ≈ 0.05; chunks above 8,192 stayed opt-in in Strata because chunk order
  changes output; a chunk-only change moved teacher-forced KL by 0.023-0.038 at 8.5K-99K.

## Port assessment

**Integration surface.**
1. *Sizing*: a `plan_lend` analog = bisection over `workspace_bytes(c, o, d, lanes, T)` + io +
   residency-table terms (all already parametric in columns) against a lendable-frames budget;
   candidate grid must respect the 128-multiple and INT8-KV wave rounding, where Basalt uses a bare
   256 grid. Small, self-contained.
2. *Arena/ring*: no new lending API needed — extend the existing `enter_wide`/stream-lease
   machinery (`expert_residency.cpp:563-567`: `lendable()` = frames − loaned − 25% floor) to lease a
  **variable, per-prompt** size instead of one startup-sized `wide_work_`,
   and let the ring depth scale with the chunk (today fixed at program_impl.h:790). The layout
   rework is that per-call io and residency tables currently scale with the startup chunk
   (`columns = max(chunk, …)`): an auto ceiling of 16K-32K pays those statically no matter who
   lends the arena, so the port needs chunk-capped io staging (sub-chunk calls already exist for
   MTP: `kMtpChunkColumns`) or accepts the larger fixed tables.
3. *Admission/scheduler*: the chunk becomes a runtime value from `residency_->lendable()`.
   Permits are unaffected (KV-coverage ledger; frames are a clean cache, and §9.2's
   "reserved-but-unwritten holds an expert on loan" keeps the reservation invariant exactly).
   One-prefill-turn-per-cycle and rotation at chunk boundaries are preserved — an auto-chunked
   long prompt simply has more, smaller rotation points.
4. *Concurrent decode*: Basalt's while-decoding caps (1216 slots, RAM-backed-only, ring 64,
   chunk shrinks to the loan) have direct analogs: lend during the decode/prefill rotation with a
   lendable floor ("vision never borrows the last 25%"), minimum-score run, busy-frame avoidance —
   all already built for vision. The refill difference is favorable: Infernix's lazy re-admission
   avoids Strata's eager copy-back (§9.2's stated criticism), at the price of a colder cache after
   every prompt — the recovery cost its VM8 test measured for vision.

**Cost.** The frame-lend primitives, leases and lazy refill are in the product. The new work is:
runtime chunk resolution + its plumbing through `plan_calls`/walk/call-plan; variable-size arena
leases; chunk-independent (or max-chunk-bounded) io/residency layout; a floor/cap policy for
lending while lanes decode; and re-qualification of serve/prefix/replay tests under variable
chunks. Estimate: medium (single-digit dev-days for the mechanism, comparable for contracts/tests).

**Benefit.** Strata's published claim, cited second-hand by Infernix's own design doc
(+21-59% prompt throughput at 9K-100K prompts with `auto:16384`, architectds/Strata numbers on
their cards), and Infernix's measured 413 → 659 tok/s pp4096 for 1024 → 4096 fixed chunks show a
large long-prompt TTFT win is the prize; Basalt publishes no independent end-to-end numbers for
the mechanism (its README documents sizing and its own environment's vision TTFTs only). Per-request
sizing additionally removes the current "the configured chunk is the whole-session cost" behavior —
short prompts lend nothing, which today's fixed route cannot express — and frees frames for KV
capacity at short-context settings.

**Risks / contract collisions.**
- *Determinism*: chunk choice changes GEMM tiles → prompt K/V rounding → greedy outputs at
  near-ties (Basalt's own note; Strata kept >8,192 opt-in for this reason). If the chunk is sized
  from **live** `lendable()`, identical prompts produce different tokens depending on cache/VRAM
  state at admission — a regression against Infernix's reproducible-results posture. Mitigation:
  size the chunk from a static, config-derived budget (floor, ceiling, lanes, model shape), not
  from instantaneous pressure, falling back toward `--prefill-chunk` only when the lease fails.
- *Prefix cache exactness*: tap-ladder defaults are defined from `prefill-chunk`
  (types.h:255-258); a resolved chunk must anchor them deterministically or cross-request reuse
  positions drift. Cached blocks stay self-consistent (KV bytes are copied, Replay honors the
  ledger's splits), so exactness is preserved per-request; the change is that outputs become a
  function of the chosen chunking, which the product currently fixes by flag.
- *In-flight decode slots*: lent frames reduce the resident expert set for the whole prompt, so
  co-resident decoding lanes take more misses/CPU-served experts until lazy re-admission
  repopulates; Basalt answers this with the 1216-slot RAM-backed-only cap and chunk shrink while
  batched — a port needs the same policy, plus interaction with elastic KV growth and the prefix
  cache's Host-block prefetch, which also contend for frames (design doc: "prefetched into free
  and Host-backed Device pages, never growing the pool into the expert frames").
- Single-GPU simplification: Basalt's multi-participant loan logic (per-stage caches, capping-card
  warnings, `relend` under elastic VRAM) does not apply; Infernix has one cache and would only
  port the single-participant path.

## Recommended verdict

**Port the mechanism, not the code — as the already-designed §13 target, gated on a
product decision.** The idea is validated (Strata's claimed +21-59% long-prompt, Infernix's own
1024→4096 chunk table) and Infernix already owns every hard primitive (frame leases with
no-write-back evict, lazy re-admission, lowest-LFRU contiguous runs, chunk-parametric workspace
sizing). What must be decided before implementing, because each is a contract change rather than
an implementation detail: (a) whether the auto chunk is sized from a static deterministic budget
or live lendable state (determinism of outputs), (b) re-anchoring the tap-ladder defaults
(§17/spec contract), (c) the while-decoding lend floor/caps and their effect on lane decode rates.
Do **not** port Basalt's eager refill-back (Infernix deliberately rejects it, §9.2), its
`relayout`-per-request machinery (the lease design covers it), or its multi-GPU participant logic.
Baseline behavior for a fixed `--prefill-chunk` should remain the fallback when no lease fits, so
current routes and their published numbers stay reproducible.
