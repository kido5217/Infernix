# NVFP4 quantization-method adoption

This reference records which converter-side 4-bit quantization methods Infernix has evaluated, what
each would demand of the stored NVFP4 representation, and the measured reason none is adopted. It is
the durable record of issue #3's assessment; the raw artifacts live on the branches named in
[Provenance](#provenance).

Scope: weights and activations quantized to NVFP4 (E2M1 codes, E4M3FN group scales) by the
converter, and the runtime inputs such quantization would need. Whole-model speed, scheduling and
artifact framing are out of scope.

## What the representation allows

The stored `nvfp4` weight format is E2M1 4-bit codes, **one E4M3FN scale per 16-element K-group**,
and **one FP32 divisor for the whole matrix** (`docs/maintainer/tensor-formats.md` §3.3), laid out
and swizzled as `docs/maintainer/storage-layouts.md` §4 fixes. The tensor record rejects unknown fields (`tools/artifact/schema.py`) and the numeric-format
registry is closed (`tools/artifact/formats.py`). The kernels issue
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3` and assert 16/64/128
geometry (`src/ops/common/mma.cuh`, `src/ops/linear/nvfp4/nvfp4_geometry.h`).

Two consequences decide every placement below:

- **A different scale *selection* rule is converter-only.** It rewrites the same words — the shape of
  the existing `nvfp4_mse` method (`tools/convert/quantization/nvfp4.py`,
  `tools/convert/methods.py`). No kernel, layout or format change.
- **A second scale per group, or a per-channel *activation* scale, is not representable.**
  The first needs a new format, layout and kernels; the second has no field at all: the activation
  quantizer takes one scalar divisor (`src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`), the artifact's
  `uses` record admits only that scalar (`tools/convert/recipe.py`, `include/infernix/ops/weight_input.h`),
  and the fused wide route requires the routed bank's contributing inputs to share one divisor
  (`src/ops/offloaded_sparse_moe/cuda/wide_expert.h`; the narrow route computes unequal gate/up
  scales exactly, so the requirement is the wide route's).

## Evaluated methods

| Method | Weight side | Activation side | Tier | Calibration |
|---|---|---|---|---|
| **OSFP4** (arXiv 2610.08231) | diagonal smoothing `W A⁻¹` folds into the stored weights; the calibrated E4M3 block-scale selection rewrites the scales | `A X` — a per-input-channel multiply, applied explicitly where it cannot fold into a preceding normalization | **runtime input** for the transform; **converter-only** for selection alone | corpus + forward passes (a second-moment estimate over a calibration set) |
| **DualQuant** (arXiv 2610.13767) | row scale `α` folded into the dequantization scale | column scale `β` as `X Diag(β)⁻¹`, applied inside the activation quantizer | **runtime input** | none — strictly calibration-free (no activation data, statistics or forward pass) |

Neither needs a second scale inside a 16-element group, so neither is a format change. Both need a
per-channel activation vector on the routed expert inputs: the experts' input is the
hyper-connection collapse of a nonlinearly gated norm output, and the down projection's input is the
product `silu(g)·u`, so a per-channel diagonal cannot ride the preceding normalization
(`tools/flash_next/reference.py`, `hc_mix`).

## Measurement

The protocol was pre-registered before any measurement (ticket #29): 25 real matrices from the local
BF16 Qwen3.8-Flash-Next checkpoint (layers 0, 6, 12, 18, 24, 30, 36, 42 × expert 0 × gate/up/down,
plus the dense shared-expert gate), an exact FP64 oracle, the repo's own quantizer and decoder, and a
bar of **≥ 22/25 wins, median relative GEMM-error reduction ≥ 10 %, no matrix worse than +2 %**
against the better of `nvfp4_absmax` and `nvfp4_mse` per matrix.

| Arm | Wins (of 25) | Median reduction | Worst ratio | Verdict |
|---|---|---|---|---|
| OSFP4 smoothing (transform + the repo's MSE scale rule) | 25/25 | 3.94 % | 0.992 | not cleared |
| OSFP4 calibrated selection (no smoothing) | 25/25 | 2.57 % | 0.997 | not cleared |
| DualQuant | 0/25 | −3.75 % | 1.113 | not cleared |
| exploratory hybrid (non-gating) | — | — | — | reported only |

The controls' medians are 0.1279 (`nvfp4_absmax`) and 0.1182 (`nvfp4_mse`); `nvfp4_mse` is the
better control on every matrix.

**Why the published gains do not transfer.** Both papers report against round-to-nearest or GPTQ
baselines; this converter's control is already an exhaustive MSE scale search. A rule that beats RTN
by a wide margin can still beat an MSE-optimized rule by only a few percent — which is what was
measured, and why the pre-registered bar, not the papers' headlines, governs.

**Two results stand on their own**, independent of the bar:

- DualQuant's row scale `α` **cannot** be absorbed by NVFP4's per-16 E4M3FN scale: folding it changes
  100 % of scale words, and the re-rounding costs a median 2.46e-2 relative L2. That closes the
  question the method's paper leaves open for this format.
- DualQuant loses to the control even with the fold disabled (5/25 wins, median −1.56 %), so the
  method fails for two separable reasons, not one.

**Correctness anchor.** The repo's MSE rule at the *source's* divisor reproduces the stored NVFP4
words exactly (102400/102400 scale words on the 24 informational matrices), which is why that source
set cannot gate: it measures replication of the source's own configuration, not rule quality.

## Disposition

- **Neither method is adopted**, and no converter or runtime change follows from this evaluation.
  The bar was pre-registered to justify buying a runtime cost, and nothing cleared it.
- **Converter-only scale selection** — OSFP4's selection rule without smoothing won 25/25 at 2.57 %.
  Because that number was produced under a bar set for a different question, it is not adopted here;
  it is a separate evaluation with its own pre-registered threshold and a whole-model acceptance
  (ticket #33).
- **The runtime per-channel activation scale** — the capability both methods require is filed for
  design and costing (ticket #34), not built.

## Provenance

- Papers: OSFP4, arXiv 2610.08231; DualQuant, arXiv 2610.13767.
- Representation facts: [#26](https://github.com/kido5217/Infernix/issues/26),
  [#27](https://github.com/kido5217/Infernix/issues/27); registered formats in
  `docs/maintainer/tensor-formats.md` and `docs/maintainer/storage-layouts.md`.
- Pre-registration: [#29](https://github.com/kido5217/Infernix/issues/29). Prototype and raw output:
  `proto/nvfp4-transform-prototype` (`tools/proto/nvfp4_transform_prototype.py` and its
  `…report.json` / `…selfcheck.json`), pushed but not merged; the same branch modifies
  `tools/flash_next/reference.py` to add the activation tap used for the capture.
- The prototype's numbers were recomputed independently from its raw output, its source hashes and
  activation batch re-derived, and its bar arithmetic re-evaluated before publication; the report
  embeds its own reproduction commands and the revision of the code that generated it.

## Related

- [Model-level numerical quality](model-quality.md) — how quality is judged at model scale (KL to a
  16-bit-activation reference, perplexity noise rules); the acceptance any future adoption must meet.
- [Numeric formats](tensor-formats.md) and [Storage layouts](storage-layouts.md) — the represented
  values and byte geometry this reference reasons about.
- [Weight conversion](../weight-conversion.md) — recipes, methods and the `METHODS` surface a
  converter-only selection change would extend.
