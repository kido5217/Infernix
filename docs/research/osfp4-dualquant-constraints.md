# OSFP4 and DualQuant vs the Infernix `nvfp4` weight format

Research note for [issue #26](https://github.com/kido5217/Infernix/issues/26). It classifies what
each method demands in this repo's representation terms, using primary sources only: the papers'
LaTeX sources (arXiv `2610.08231` and `2610.13767`) and the released OSFP4 code
(`github.com/neriahbd/OSFP4`) plus the released DualQuant code (`github.com/MahsaMoonlike/dualquant`,
HEAD `d84b7e9`). Equation numbers are the ones the papers render.

## 0. The fixed target being argued against

The relevant repo facts were established with file:line evidence in the resolution of issue #27 and
are used here without re-derivation:

- The `nvfp4` weight format stores `W[n,k] = e2m1(c) · e4m3fn(s[n, k//16]) / d_w`: E2M1 4-bit codes,
  one E4M3FN scale per 16-element K-group, one FP32 per-matrix divisor `d_w`
  (`docs/maintainer/tensor-formats.md:34,231-271`).
- The tensor record has **no spare field** and rejects extra keys
  (`tools/artifact/schema.py:193-217`, `:54-59`); the numeric-format registry is **closed**
  (`tools/artifact/formats.py:83`); the `uses` record carries only a scalar
  `activation_input_divisor` (`tools/artifact/schema.py:370-375`, `recipe.py:478-499`).
- Kernels issue exactly `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...`
  with 16/64/128 geometry asserts (`src/ops/common/mma.cuh:136`,
  `src/ops/linear/nvfp4/nvfp4_geometry.h:6-17`).
- A scale-**selection** change at that geometry is method-only — a `quantize_*` callable plus one
  `METHODS` entry, precedent `nvfp4_mse` (`tools/convert/methods.py:326,370-383`) — while a second
  per-16 scale would be a new format + layout + route bodies + kernels.

## 1. Constraints table

| Method | What changes | Where it lives | Tier | Calibration at conversion | Reported setting / metric |
|---|---|---|---|---|---|
| **OSFP4** | A block-diagonal **diagonal smoothing matrix `A`** (one factor per input channel, grouped in 16s) applied as `W A⁻¹` and `A X`; plus joint optimization of the existing **per-16 E4M3 weight-block scales** (`Γ_W`) | `W A⁻¹` folds into the stored weights; `A X` is a per-input-channel multiply on the activation side, folded into the preceding norm where possible and applied by the runtime otherwise | **(ii)(c)** — per-channel activation transform. Its block-scale optimization alone is **(i)** method-only; no second per-16 scale, so not (iii) | Corpus + forward passes: 1024 FineWeb-Edu sequences of 2048 tokens; `Σ_X = (1/k₁)Σ x_j x_jᵀ`, and for W4A4 a `k`-row activation subset (`X`) | NVFP4 **W4A4** (and W4A16), Llama-3.1-8B-Instruct: mean accuracy **77.03 %** (W-SIC/X-RTN, absmax act scaling) vs BF16 **79.22 %** and FP-Quant GPTQ **76.34 %**; WikiText-2 PPL **7.88 → 7.67**. Maps onto the repo's A4 recipe (4-bit W+A, NV per-16 E4M3 block scale, FP32 tensor scale) |
| **DualQuant** | A **row scale `α`** (per output channel) and a **column scale `β`** (per input channel) around the format's own block quantizer: `W ≈ Diag(α)⁻¹ Q(Diag(α) W Diag(β)) Diag(β)⁻¹` | `α` is absorbed into the stored weight's dequant scale; `X·Diag(β)⁻¹` is a **per-channel activation pre-scale applied inside the quantization kernel** | **(ii)(c)** — per-channel activation transform. `α` folds into the existing per-16 weight scale (method-only); no second per-16 scale, so not (iii) | **None** — calibration-free; uses only the static weight matrix (no activation data, no statistics, no forward pass) | INT4(G=64)/MXINT4/MXFP4(G=32)/NVFP4(**G=16**) W4A4. Llama-3.1-8B NVFP4 WikiText-2 PPL **6.83** vs RTN 9.04 and SmoothQuant 7.32; downstream MMLU 57.76, HellaSwag 56.65, PIQA 78.24, ARC-Easy 77.90, WinoGrande 71.67. NVFP4 numbers are **fake-quant**; its only real packed W4A4 deployment is **INT4 G=64** on an RTX 4090 (Sec. 5.2). The repo does carry an INT4 G=64 **weight** format (`q4_g64_fp16`, `tools/artifact/formats.py:67`) but runs it with A16 activations: its only **4-bit** activation route is NVFP4 (A4), and FP8's is A8 — INT4 has none (`include/infernix/ops/linear.h:100-106`) |

**Decisive constraint for both:** the method needs a **per-channel multiply on the activation side**,
which this repo does not support — it has only the scalar `activation_input_divisor`. Neither method
needs a second scale inside a 16-element group, and both keep the NVFP4 per-16 E4M3 block geometry,
so neither is (iii).

## 2. OSFP4 (arXiv 2610.08231)

### 2.1 The mathematical object

A diagonal smoothing matrix `A ∈ R^{n×n}`, block-diagonal with 16×16 blocks
`A_ℓ = diag(α_ℓ^{(1)}, …, α_ℓ^{(16)})`, so there is one factor `α_ℓ^{(r)}` per **input channel**
`t = 16(ℓ−1)+r` (Sec. 2.2; App. I, "Detailed Loss Derivations and SIC Procedure"). The matrices actually quantized are

> Eq (6): `W'_ℓ = diag(Γ_{W_ℓ})⁻¹ W_ℓ A_ℓ⁻¹`,  `X'_ℓ = A_ℓ X_ℓ diag(Γ_{X_ℓ})⁻¹`,

with the first-order MatMul error

> Eq (7): `e = Ŵ X̂ − W X = e_W A X + W A⁻¹ e_X + e_W e_X ≈ e_W A X + W A⁻¹ e_X`.

The paper is explicit that this is a SmoothQuant-style pair: "we actually quantize `W A⁻¹` and
(possibly) `A X`" (App. I). `Γ_W` is the existing single E4M3 scale per 16-element group per row
(`Γ_{W_ℓ} ∈ C_scale^m`, App. I), and `Γ_X` the activation side's; there is **no second scale per
group**. The optimization uses the dithered quantizer of Eq (3) and the loss `φ` of Eq (4), with the
RTN/SIC losses Eq (11)/(12), Eq (16), Eq (21) and the four total losses Eq (22)–(25).

### 2.2 Where the object lives

- **Weight side is foldable.** `W A⁻¹` is quantized with the existing per-16 E4M3 scales and stored;
  no new weight geometry. The paper even sets the saved FP32 global weight scale to one and puts the
  compensating factor "already incorporated into `A`" (App. B, "FP32 tensor scales"). The remaining
  weight component — optimized E4M3 block scales `Γ_W` and the fine E4M3 search — is a pure
  selection change at the existing 16/64 geometry.
- **Activation side is per-channel and needs the runtime.** `A X` multiplies the input channels of the
  activation. The paper folds it into a preceding normalization layer "where possible" and applies it
  explicitly "before the attention output and MLP down projections otherwise" (App. D.2,
  "OSFP4 and activation-scale ablations"); "the smoothing transformation `A` is applied at runtime
  only on `o_proj` and `down_proj`" (App. G, "Inference Efficiency"). `A` is rounded to the weight dtype when folded into a
  norm, or to BF16 when "explicit runtime smoothing is required" (App. B).
- The released code confirms the mechanism and its shape: `deploy_mapping_smooth_quant_scale` either
  divides the balance layers' weight columns `weight[:, :input_width] /= scale.view(1, -1)` and
  multiplies the preceding layer (`_deploy_smooth_layer_smoothing`), or, when there is no preceding
  layer, registers a per-input-channel `smooth_quant_scale` parameter and a forward-pre-hook that
  computes `args[0] * scale` (`smoothing.py:15-69`; `base.py:56,420-437`). The README states the same:
  "OSFP4 stores that scale in the checkpoint as `smooth_quant_scale`, and the plugin multiplies it
  into the layer's input before running vLLM's stock NVFP4 kernels."

### 2.3 Tier

**(ii)(c).** `A X` is a per-channel rescale of the activations, applied at run time wherever it
cannot be fused into a preceding norm. This repo supports a scalar `activation_input_divisor`, not a
per-channel vector, so the method is not representable as method-only. It does **not** need a second
per-16 scale, so it is not (iii). Stripped of smoothing (`A = I`), OSFP4's remaining contribution is
a weight-block-scale selection at the existing geometry — that part alone would be tier (i), the
`nvfp4_mse`-style change.

### 2.4 Calibration

Corpus + forward passes. All four configurations first collect `k₁` activation vectors and form
`Σ_X = (1/k₁)Σ x_j x_jᵀ`; W4A4 additionally keeps a random subset of `k ≤ k₁` full activation vectors
as the `n×k` matrix `X` (k activation vectors, i.e. k columns). The released modifier sets `requires_calibration_data = True` and uses 1024 FineWeb-Edu
sequences of 2048 tokens (`base.py`; App. D.1). It is **not** calibration-free.

## 3. DualQuant (arXiv 2610.13767)

### 3.1 The mathematical object

A dual-scale parameterization (Sec. 4.2, Eq (2)):

> Eq (2): `W ≈ Diag(α)⁻¹ Q(Diag(α) W Diag(β)) Diag(β)⁻¹  ≜ Diag(α)⁻¹ Q̂_{α,β} Diag(β)⁻¹`,

with `α` a per-**row** (output-channel) scale and `β` a per-**column** (input-channel) scale; the
inverse scales are constrained to the format's admissible set — for NVFP4, FP8 UE4M3. Substituting
into `X Wᵀ` gives the activation factorization (Sec. 4.2, Eq (3)):

> Eq (3): `X Wᵀ ≈ (X Diag(β)⁻¹) Q̂_{α,β}ᵀ Diag(α)⁻¹ ≜ X̃ W̃ᵀ`,  `X̃ = X Diag(β)⁻¹`.

`α, β` are found by BCD on the weight-reconstruction MSE (Eq (4)), via the variable-split problem
(Eq (5)) with closed-form updates `α_i* = ‖R_{i,:}‖²/(W_{i,:}R_{i,:}ᵀ)`, `β_j* = ‖S_{:,j}‖²/(S_{:,j}ᵀW_{:,j})`
(Sec. 4.3; Algorithm 1). The black-box `Q` is the *native* format quantizer: for NVFP4 it is
`NVFP4[E2M1]{16}` with a per-16 E4M3 scale (`formats/nvfp4.py`), so `α`/`β` sit **around** the existing
per-16 block scale, not as a second scale inside a group.

### 3.2 Where the object lives

The paper states the inference-time arrangement directly (Sec. 5.1): the layer computes
`Q(X̃) Q̂ᵀ Diag(α)⁻¹`; "the row scale `α` is absorbed into the per-row dequantization scale of `Q̂` at
quantization time and incurs no runtime cost", while "the column scale `β` introduces a per-channel
rescaling of the activations **before** quantization" and "does not replace the format-native
activation scale", applied "inside the activation-quantization kernel without a separate kernel
launch". It is deliberately not folded into the shared LayerNorm: `β` is optimized independently
per layer, unlike SmoothQuant. The released code matches:

- `_reconstruct_block_alpha_beta` stores `alpha_inv_q.unsqueeze(1) * BlockQuant(α·W·β)` — `α` is
  multiplied into the stored weight (foldable), never applied at run time.
- When activation quantization is enabled, the wrapper returns the per-column vector `column_scales`
  (`beta_inv`, one value per input channel) "for runtime activation-side pre-scaling"
  (`layer_wrapper_dualquant.py:405-410,516-533`); otherwise it folds `β` into the weights.
- `methods/dualquant.py:25` sets `needs_calib_acts = False`.

### 3.3 Tier

**(ii)(c).** `β` is a per-input-channel vector applied to activations before quantization — a runtime
per-channel transform the repo does not support (`activation_input_divisor` is a scalar). `α` is
folded into the existing weight scale and is not a new stored object. No second per-16 scale, so not
(iii).

### 3.4 Calibration

**None.** The abstract and Sec. 4.1 claim calibration-free operation "without any access to
calibration data or activation statistics"; Sec. 5.2 states the offline cost has "no calibration
forward passes, no Hessian, and no inner optimization loop"; `β` is a deterministic function of the
static weight matrix alone (`_init_beta` uses only `W`; `needs_calib_acts = False`). What
"calibration-free" excludes is confirmed to be: any activation tensor, any activation statistic, and
any forward pass. It does require the model's static weights and the target format.

## 4. Does each paper's evaluated setting map onto this repo's A4 recipe?

This repo's A4 recipe is 4-bit weights **and** activations in NVFP4 (G=16, per-16 E4M3 block scale,
per-matrix FP32 `d_w`, scalar `activation_input_divisor`).

- **OSFP4**: format-for-format, yes — E2M1 with one E4M3 scale per 16 input channels and an FP32
  tensor scale (App. B), evaluated natively through vLLM on an RTX 5090 / B200 (Sec. 4). The
  mismatch is exclusively the missing per-channel activation transform.
- **DualQuant**: the **format** for its NVFP4 row matches (G=16, UE4M3 scale, 4-bit W+A), but its
  NVFP4 results are **fake quantization** on GH200 with weights dequantized to FP16/BF16 at inference
  (Sec. 5, evaluation setup). Its only real packed deployment is **INT4 with G=64** on an RTX 4090
  (Sec. 5.2, "Real packed W4A4 deployment") — a **W4A4** route the repo does not run (it carries
  `q4_g64_fp16` weights but only A16 activations; its only 4-bit activation route is NVFP4). The metric that maps onto the A4 recipe
  is therefore the NVFP4 fake-quant perplexity/accuracy, not the deployed number.

## 5. Equations relied on

OSFP4 (arXiv 2610.08231):
- Eq (3) dithered quantizer `Q̃_FP4(x) = 2^U Q_FP4(2^{-U} x)`; Eq (4) `φ(x)`; Eq (5) Lemma 1.
- **Eq (6)** `W'_ℓ = diag(Γ_W)⁻¹ W_ℓ A_ℓ⁻¹`, `X'_ℓ = A_ℓ X_ℓ diag(Γ_X)⁻¹`.
- **Eq (7)** `e ≈ e_W A X + W A⁻¹ e_X`.
- Eq (11)/(12) `L^RTN_W`, Eq (16) `L^RTN_X`, Eq (21) `L^SIC_W`; Eq (22)–(25) the four total losses
  (`W-RTN`, `W-SIC`, `W-RTN/X-RTN`, `W-SIC/X-RTN`).
- App. B: `a_{ℓr}=log α_ℓ^{(r)}`, `b_{ℓi}=log γ_{W_ℓ}^{(i)}`, `c_{ℓj}=log γ_{X_ℓ}^{(j)}`; the
  canonicalization and activation global scale `g_X = 448·6 / M_X`.

DualQuant (arXiv 2610.13767):
- **Eq (1)** joint error decomposition `X Wᵀ − Q(X)Q(W)ᵀ = (X−Q(X))Wᵀ + Q(X)(W−Q(W))ᵀ`.
- **Eq (2)** `W ≈ Diag(α)⁻¹ Q(Diag(α) W Diag(β)) Diag(β)⁻¹`.
- **Eq (3)** `X Wᵀ ≈ (X Diag(β)⁻¹) Q̂ᵀ Diag(α)⁻¹`.
- Eq (4) primal MSE; Eq (5) variable splitting; closed-form `α_i*, β_j*` (Sec. 4.3, unnumbered).
- Sec. 5.1: inference `Q(X̃) Q̂ᵀ Diag(α)⁻¹`, `X̃ = X Diag(β)⁻¹`.

## 6. What would settle the open points

1. **Folding `α` into the NVFP4 per-16 E4M3 scale (DualQuant).** The paper says `α` is "absorbed into
   the per-row dequantization scale", but NVFP4's native scale is per-16-group (a finer index than
   per-row). Whether `α_i` times each block scale is stored as a re-rounded E4M3 value (exact in the
   format) or needs an extra per-row field is not specified. Settle it by packing `Q̂` with a real
   NVFP4 encoder and comparing reconstruction with and without `α` folded — the released code proves
   this only for the INT4/G=64 path, not NVFP4.
2. **OSFP4 `A` fusion into this repo's norms.** The paper assumes a preceding RMSNorm/LayerNorm can
   absorb a per-channel factor for `q/k/v/gate/up`. Whether this repo's norm path can host that
   without a separate multiply is a repo-side question, not a paper ambiguity; settle it by inspecting
   the norm kernels' epilogue.
3. **Carrying a per-channel activation vector.** Both methods need a per-input-channel vector on the
   activation side. Whether it must be a new stored field or can be recomputed at load time
   (DualQuant's `β` is a deterministic function of the stored weights; OSFP4's `A` is not) is a design
   choice, not resolved by either paper. Settle it by deciding the container-side contract first.

No genuine ambiguity was found about *where* each transform lives: both papers (and both released
codebases) place the activation-side factor as a runtime per-channel multiply.

## 7. Sources

- OSFP4 paper, LaTeX source: `arXiv:2610.08231` (`OSFP4_Arxiv_main.tex`, `accuracy-summary.tex`,
  `ablation-summary.tex`).
- OSFP4 code: `github.com/neriahbd/OSFP4` — `src/llmcompressor_osfp4/modifiers/smoothing.py`,
  `base.py`, `runtime_contract.py`, `observers/scale_selection.py`, `modifiers/README.md`, `README.md`.
- DualQuant paper, LaTeX source: `arXiv:2610.13767` (`DualQuant_arxiv.tex`, `tables/main_table_llama_family.tex`,
  `tables/downstream_main.tex`, `appendix/packed_w4a4.tex`, `appendix/causal_effect_of_beta.tex`,
  `appendix/baseline_methods.tex`).
- DualQuant code: `github.com/MahsaMoonlike/dualquant` (HEAD `d84b7e9`) — `layer_wrapper_dualquant.py`,
  `methods/dualquant.py`, `formats/nvfp4.py`.
- Repo facts: resolution comment of `kido5217/Infernix#27`.
