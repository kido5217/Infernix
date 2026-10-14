# Basalt quant formats and quality tooling — assessment (ticket #59)

Question: (a) do Basalt's GGUF quant formats offer technique-level adoptions for Infernix's NVFP4
route; (b) which of Basalt's quality/verification tooling is portable as practice or tool.
Sources: Basalt clone at `/home/kido/trash/temp/opencode/basalt` (all `file:line` below), Infernix
`main` as checked out here. A ggml runtime import is out of scope per the map; techniques only.

## Basalt mechanisms

**Format storage.** `.basalt` copies GGUF tensor blocks byte for byte; the only conversion is the
opt-in `--compat-bf16`, recorded in `meta` (README.md:222-228). Routed experts keep the source type
mixed per layer and per projection class — IQ3_XXS pack: gate/up IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/
IQ2_S, down Q2_0 on 30 layers and IQ4_NL on 18 (docs/precision.md:34). The mixed allocation is a
pack-creation decision (importance-driven upstream, by llama-quantize with an imatrix); the engine
just reads the stored type.

**Decode dots (src/kernels/cuda/iq_kernels.cu).** Codebook i-quants: 8-bit grid index + sign bits
dequantized through device LUTs (`iq3s_grid`, `iq1s_grid_gpu`), signs unpacked by a popcount-parity
trick (`vecdotq.cuh:55-62`, `iq_kernels.cu:180-227`). Two-term affine for K-quants:
`d*scale·dot − m*min·(act-sum)` with branchless 6-bit scale/min extraction
(`iq_kernels.cu:300-360`). Sub-block scales packed narrower than fp16: IQ3_S 4-bit nibbles
(`iq_kernels.cu:224`), IQ4_XS 6-bit + 2-bit high split (`iq_kernels.cu:281-300`), IQ1_M with the
fp16 super-scale rebuilt from the top nibbles of the four sub-scale words plus 3-bit odd sub-scales
`2q+1` (`iq_kernels.cu:229-260`). Activations are q8_1 (int8 + fp16 scale per 32) with exact int32
per-block accumulation and the fp32 scale applied once at the end (docs/precision.md:73).

**Prompt path (src/prefill/mmq).** A transcribed subset of llama.cpp MMQ: int8-tensor-core dots,
grouped by expert with stream-k work partitioning (mmq.cuh:2-3); a per-(type, J) tile-config table
that "only affects speed, register pressure and shared memory use, not results"
(mmq_config.cuh:4-5, covering Q2_0/Q8_0/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL); a
gather-quantizer producing q8_1 activations through a row-index map (quantize.cu:10-70). IQ1_M and
the K-quants do **not** run MMQ in prefill — dense projections dequantize to fp16 for cuBLAS
(docs/precision.md:96; prefill/gemm.cu:150-183).

**Other arithmetic practices.** hc matrices: int8 shadow copies, one fp16 scale per 32 inputs,
exact int8 + fp32 fmaf (docs/precision.md:41, 55); hc-up and prompt attention use fp16 hi/lo
splits (~22-bit mantissa on tensor cores, docs/precision.md:56, 101). Basalt deliberately does NOT
q8_0 the hc-read activations and shows the cancellation math that justifies it (docs/precision.md:
191-213).

**Quality tooling.**
- `tools/ppl_gate.py` — drives the engine `--serve` session; teacher-forced scoring **through the
  decode path**: prefill half a 512-token wikitext-2 chunk, then verify-window rows of forced width
  `--t` (ppl_gate.py:138-141). The engine computes per-position NLL, base NLL, full-vocabulary KLD
  (log-sum-exp accumulated in double) and same-top-1 against a llama.cpp `--kl-divergence-base`
  `.kld` file (ppl_gate.py:27-34; include/basalt/program/ppl_gate.hpp:112-154). Output is a
  per-row `rows.tsv` with `--report` and `--diff`; forcing T=1/3/7 and diffing rows proves the
  spec-verify path is bit-identical to plain decode (docs/precision.md:129-145).
- `tools/greedy_identity.py` — pass/fail gate: two engine binaries, same prompts at temperature 0,
  every token compared; prints `GREEDY IDENTICAL`/`GREEDY DIFFERS` with exit code 0/1
  (greedy_identity.py:10, 121-126). Placement is pinned (`--adapt-swaps 0 --pcie-frac 0`, manual
  tier sizes) because auto-sized tiers change rounding (lines 5-8, 25-26).
- `tools/serve_soak.py` — hours-long random OpenAI/Anthropic streaming traffic with client cancels,
  near-context-limit prompts, over-limit prompts expecting 400, images; a monitor samples the
  server's `/slots`, GPU memory, process-group RSS and mem/swap, flags streams that stop producing
  tokens, drains, and classifies every finished request against its expectation (serve_soak.py:1-9,
  237-337).
- `docs/precision.md` — one authoritative whole-model precision map: every matmul's weight format,
  activation format and accumulation site with `file:line`, every claim reproducible with the
  commands at the end, plus an explicit account of the reference's own noise (docs/precision.md:
  191-227). Changing arithmetic semantics means updating this table.

## Infernix today

- Closed numeric-format registry: `bf16/fp32/int32`, symmetric integer groups with one fp16 scale
  (`q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16`), `nvfp4`, `nvfp4_mul`,
  `fp8_e4m3fn_row_bf16` (tools/artifact/formats.py:63-88). NVFP4 stores E2M1 codes + one E4M3FN
  scale per 16 + one FP32 matrix divisor; the kernels issue `mma…kind::mxf4nvf4.block_scale…ue4m3`
  (docs/maintainer/nvfp4-quantization-methods.md:12-19).
- The converter's method surface already minimizes scale-fit with an exhaustive MSE search
  (`nvfp4_mse`) and records provenance per artifact (nvfp4-quantization-methods.md:62, 78-80;
  `out/` conversion reports). OSFP4 and DualQuant were evaluated and rejected against a
  pre-registered bar; converter-only calibrated selection and a runtime per-channel activation
  scale are filed as tickets #33 and #34 (nvfp4-quantization-methods.md:82-91).
- Model-level quality: `infernix-perplexity` measures PPL and KL(top-32 + one mass bucket, lower
  bound) + top-1 agreement + target-NLL change against a saved reference, with bucketed context
  positions (docs/perplexity.md §"KL divergence from a reference run"). It scores **prefill only**
  (docs/maintainer/model-quality.md:46-48). `bench/decode_quality` (`infernix_decode_quality_gen`/
  `_judge`) covers the production decode path end to end — DFlash2/MTP/n-gram, CUDA Graph, INT8 KV
  — judged token by token against the a16 (or stored-activations) reference: agreement, mean
  regret, per-pair shared prefix (bench/README.md:149-169). The tree-greedy real test compares
  decode rounds to a fresh-prefill oracle with tolerance gates (tests/README.md:390-404). Noise
  discipline (PPL <1 % is noise; prefer KL to a fixed reference; scatter explanation) is in
  model-quality.md §1. Op-level exactness rests on the FP32/FP64 oracle contract
  (docs/maintainer/op-development.md §6).
- Serving tooling: cancellation-aware protocol client (tools/infernix_serve/client.py:111-195),
  throughput/scaling sweeps (tools/bench/run_serve_concurrency.py), contract smoke tests
  (tools/smoke), `/health` + Prometheus metrics (docs/serving.md:76-105, 1334). No long-duration
  mixed-traffic soak with resource sampling or stuck-stream detection.

## Port assessment

### Techniques (cluster a)

| Basalt technique | Fits Infernix? | Assessment |
|---|---|---|
| Codebook i-quants (IQ1_M…IQ3_S grids) | No — format change | Sub-4-bit non-uniform grids are a different numeric format; hardware block-scale mma cannot consume them, so it is a new registry entry + new kernels, not a technique adopted into NVFP4. Infernix already applies codebook quantization where it owns the kernels — the KV side (`tools/kv_codebook`, `vq2`/`k4v2`). |
| Scale-geometry packing (6-bit IQ4_XS sub-scales, IQ1_M nibble-packed super-scale) | No | NVFP4's scale layout is fixed by the `ue4m3` block-scale mma at 16-element granularity (0.5 scale-bit/weight); GGML's narrower sub-scales trade ALU for bytes because ggml dequantizes in software. Packing differently would need new formats + kernels for a fraction of a bit per weight. |
| Two-term `d·q − m·min` affine (K-quants) | No | Asymmetric zero-points don't exist in the E2M1×E4M3FN representation; the term would need a separate activation-sum FMA and a format change. Basalt itself uses it only because the source GGUF stores it. |
| Mixed-precision allocation per layer/projection class (gate/up vs down, Q5_K head, sensitive layers up) | Yes, as converter policy | Format-per-tensor choice already lives in the converter recipes (AGENTS.md "Converter recipes choose sources, formats, packing"; registry formats per tensor). What is new is the **importance-driven allocation policy** (e.g. first/last layers or down projections at 8-bit, rest NVFP4). Pure converter change; acceptance must be a model-level KL run (model-quality.md). |
| Importance-weighted rounding | Already covered | This is activation-aware scale/code selection — exactly the `nvfp4_mse`-vs-calibrated-selection question measured and deferred to ticket #33 (nvfp4-quantization-methods.md:86-89). Basalt adds no evidence beyond what #33 will re-evaluate with its own bar. |
| Exact-int32 accumulation + single deferred fp32 scale (q8_1 dots) | Already the norm | Infernix's integer and FP4 routes accumulate exactly and apply scale once; no delta. |
| fp16 hi/lo split on tensor cores (hc-up, prompt attention) | Marginal | A technique for ~22-bit products at fp16-mma rate; Infernix's policy is FP32 where it matters (model-quality, op oracle contract) and its prompt kernels are already schedule-tuned (RESEARCH_NOTES entries). No current site where it would change an accepted result. |
| MMQ per-(type,J) config table; stream-k expert grouping; gather-quantizer | Marginal, already present in kind | Config-table instantiation and expert-grouped prefill partitioning are kernel-organization practices Infernix expresses through its own per-format schedule tuning (RESEARCH_NOTES: NVFP4/FP8 schedule sections). Nothing to adopt as a *technique* the repo lacks. |
| Byte-copy / no-requantization container contract | N/A by design | Its value is consuming the public GGUF ecosystem; Infernix's `.infernix` contract (converter records method + provenance; loader validates) already encodes the same discipline for its own sources. |

### Tools (cluster b)

| Basalt tool | Infernix today | What a port adds |
|---|---|---|
| `ppl_gate.py` (PPL/KLD **through the decode path**, forced verify windows, rows.tsv + `--diff`) | KL/PPL gate is prefill-only; the decode gate is token-level (agreement/regret/shared prefix); no per-position distribution record | A decode-path distribution gate: catches decode-kernel drift (graph replay, verify, KV layout changes) at KLD granularity, below top-1 agreement noise. `--diff` of per-position rows between two builds is also a strong refactor gate. The llama.cpp `.kld` interop itself is not wanted — Infernix's a16 reference is the better base. |
| `greedy_identity.py` (build-vs-build exact greedy gate) | `decode_quality_judge` compares agreement vs a reference and a per-pair shared prefix, but there is no pass/fail exact-equality mode between two binaries; the tree-greedy test allows ~3 % disagreement (right for changed arithmetic, too weak for neutral refactors) | A cheap deterministic guard for numerically-neutral changes (scheduler, graph capture, refactors): identical tokens or the PR explains the first difference. Port also the pinning discipline: fix tier/slot sizes, since auto-sizing changes rounding. |
| `serve_soak.py` (random OpenAI/Anthropic traffic, cancels, near/over-limit, expected-error classes, resource + stuck-stream monitor) | Concurrency sweeps measure throughput; contract tests check protocol shape; nothing runs hours of mixed traffic watching RSS/metrics/stuck streams | Long-horizon leak/fragmentation/liveness coverage — highest value on the Qwen4-Exp route, where host/SSD expert offload, preemption and Replay recovery are exactly the kind of state that fails slowly. The Infernix serve client already handles streams and cancels; `/slots` becomes `/health` + the Prometheus metrics. Mostly portable verbatim (~360 lines). |
| `docs/precision.md` discipline (one whole-model precision table, file:line per claim, change it in the same PR) | Facts exist but are spread across tensor-formats, storage-layouts, op contracts and model-quality results; per-matmul activation/accumulation semantics are not in one review-facing table | A documentation practice, not a tool: a compact per-matmul precision map (weight format, activation format, accumulation, state format) that a reviewer can diff. Low cost, fits the existing maintainer docs. |

## Recommended verdicts

**Techniques**
- **Reject:** codebook i-quants, sub-fp16 scale packing, `d·q − m·min` affine, byte-copy GGUF
  ingestion, MMQ runtime/config-table import — all format or runtime changes a closed registry with
  hardware-fixed NVFP4 geometry has no site for; none would clear a materiality bar in the #35
  mold.
- **Needs-product-change (park, don't file):** codebook/≤3-bit weight formats, if a sub-4-bit
  artifact class ever enters the roadmap; it is a new registry format with in-tree kernels.
- **Port as converter-only practice:** importance-driven mixed-format allocation across already
  registered formats (no kernel work; acceptance = model-level KL, same discipline as #33).
- **Covered, do not re-litigate:** activation-aware rounding/calibrated selection — ticket #33
  owns it; a runtime per-channel activation scale — ticket #34 owns it. Basalt contributes no new
  evidence on either.

**Tools**
- **Port (practice, medium effort):** decode-path KLD gate — extend the perplexity/decode-quality
  pair to score a reference distribution through forced decode windows and emit per-position rows
  with a `--diff` mode. Highest unique value in cluster (b).
- **Port (tool, low effort):** build-vs-build exact greedy-identity gate (two `gen` runs + compare,
  or a judge `--pair-exact` mode), with pinned tiers/slots. Fills a real gap between the oracle
  suites and the tolerant agreement checks.
- **Port (tool, low-medium):** a serve soak harness modeled on `serve_soak.py` — random mixed
  protocol traffic with cancels and near/over-limit prompts, monitoring `/health`, metrics, RSS and
  stuck streams over hours. Prioritize the Qwen4-Exp offload route.
- **Port (documentation practice):** a whole-model precision map maintained at change time; fold it
  into the existing maintainer docs rather than a parallel file.

## Provenance

- Basalt clone: `/home/kido/trash/temp/opencode/basalt` (README.md, docs/precision.md,
  tools/ppl_gate.py, tools/greedy_identity.py, tools/serve_soak.py,
  src/kernels/cuda/iq_kernels.cu, src/prefill/mmq/*, src/prefill/gemm.cu,
  include/basalt/program/ppl_gate.hpp).
- Infernix: tools/artifact/formats.py, docs/maintainer/{nvfp4-quantization-methods.md,
  model-quality.md, op-development.md}, docs/{perplexity.md, serving.md, weight-conversion.md},
  bench/README.md:149-169, tests/README.md:370-404, tools/{bench,smoke,infernix_serve}/.
- Related tickets: #33/#34 (from the #3/#35 NVFP4 quantization-method assessment); this note is
  the resolution input for #59 and adds no implementation.
