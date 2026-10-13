# Fathom (arXiv 2609.17652v3) as written, and what of it can touch a device-resident QSA index

Research note for the map executing issue #5. Throwaway artifact: no PR. Companion to
`docs/research/qsa-index-built-reality.md` (branch `research/qsa-index-built-reality`), which is
taken here as the authority on the built QSA index and is not re-derived.

**Source read.** Full HTML of *Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV
Caches*, Vivek Kalyanarangan, arXiv 2609.17652v3 (submitted v1 2026-09-15, v3 2026-09-18), fetched
from `https://arxiv.org/html/2609.17652v3`. The HTML rendering carries no page numbers, so every
citation below is by section / equation / table / figure. Where the LaTeX-to-HTML conversion is
ambiguous it is flagged. Abstract page checked at `https://arxiv.org/abs/2609.17652`; code link
`https://github.com/vivekkalyanarangan30/fathom` (not inspected — out of scope).

**One-line frame.** Fathom is a *host-resident* key-scan method: it stores the K cache as
channel-major 4-bit bit planes and lets each query read only a prefix of planes per channel,
so the scan reads fewer PCIe bytes. Its quantizer and its bit-allocation rule are statements about
*bytes vs. score error* and are placement-agnostic; its speedup is a statement about *PCIe bytes vs.
time* and is not.

---

## 1. The bit-plane encoding (§3.2, Eq. 2; Appendix A)

### 1.1 The code, concretely

Keys are quantized **once** to 4 bits with a **uniform symmetric** code, per channel, with **one
fp16 scale per 64-token block** (§3.2, first paragraph). For block `β`, channel `j`, with block
maximum `a_{β,j} = max_{i∈β} |k_{i,j}|` and cell width `s_{β,j} = a_{β,j} / 8`, the code is
(Eq. 2):

```
c_{i,j} = clip( floor( k_{i,j} / s_{β,j} ), -8, 7 ) + 8  ∈ [0, 16)
```

"so that code 8 is zero and the most significant bit is the sign" (§3.2). Note the asymmetry is
only in the clip range (`-8` vs `7`), i.e. the code is offset-binary around 8; the reconstruction
is a **mid-rise** quantizer, not mid-tread.

**Plane packing.** Bit `p` (`p = 0` is the most significant) of the 64 codes of block `β`,
channel `j`, is packed into **one 64-bit word** `P_{β,j,p}`, called a plane. Planes are stored
**channel-major**, so for a channel the words `P_{·,j,0…t-1}` are contiguous over the sequence
(§3.2). (Figure 1, left, shows one channel of one 64-token block as four planes; a depth-`t` read
is "the first `t` words".)

### 1.2 The prefix property — precise statement and assumptions

Reading the first `t` planes of a channel gives `c^(t) = c >> (4 - t)` (§3.2), and the paper's exact
claim is:

> "the dequantized value `(c^(t) + 1/2 - 2^{t-1}) · a_{β,j} / 2^{t-1}` is exactly the `t`-bit
> mid-rise quantizer of the range `[-a_{β,j}, a_{β,j})` with cells of width `a_{β,j} / 2^{t-1}`; at
> `t = 4` it reduces to `(c - 8 + 1/2) · s_{β,j}`" (§3.2).

Equivalently (the same statement, §3.3): a depth-`t` read "splits that range into `2^t` cells of
width `Δ = 2 a_{β,j} / 2^t`", with per-cell error variance `Δ²/12 = a_{β,j}² / (3 · 4^t)`.

**What the claim assumes** (all stated in the paper, §3.2 "Compatibility with other KV
quantizers"):

- **Uniformity.** The code must be a *uniform* scalar quantizer. "Uniformity is what makes a prefix
  of bit planes an exact coarser quantizer; a non-uniform code would not have this property." A
  prefix of a uniform code *is* a coarser uniform code (so KIVI per-channel codes with a zero point
  and INT8 are "compatible directly").
- **Bit order relative to magnitude.** The MSB is the **sign** and the next bits are the decreasing
  significance bits of the offset magnitude (`floor(|k|/s)`); the prefix `c >> (4-t)` therefore
  keeps the t most significant bits of the *offset-binary magnitude*, so each added plane halves
  the cell width. This is the "bit order relative to the value's magnitude" that makes a prefix a
  *value* prefix, not a byte prefix.
- **Rotation is allowed only if it concentrates variance.** A fixed pre-quantization rotation is
  compatible ("A fixed rotation before quantization is compatible") *provided it concentrates
  variance (a KLT)*; a variance-*spreading* rotation (TurboQuant's random rotation) makes
  water-filling vacuous, and TurboQuant's Lloyd–Max quantizer is in addition non-uniform (§3.2).
- **FP8 / float prefixes do not have the property.** The paper says FP8 "whose prefixes are sign
  and exponent bits, is monotone in magnitude and would likewise need measured rather than `4^{-t}`
  marginal gains" (§3.2). This is the load-bearing exclusion for a BF16 store (see §5 below): a
  BF16 prefix is sign + exponent + mantissa, monotone in magnitude but **not** an exact uniform
  `t`-bit quantizer, and its marginal-error law is not the `4^{-t}` of Eq. 4.
- **Scope of "exactly".** The exactness is a property of the *code prefix as a scalar quantizer*.
  It does not claim the code `c` equals the true key — only that the `t`-plane prefix is the exact
  best (round-to-nearest-cell) uniform `t`-bit quantizer of the same range and scale. The block
  scale `a_{β,j}` must be stored and read for the dequantization to be that quantizer.

### 1.3 Index dtype and size in their setup

The store is "a 4-bit copy of K with fp16 block scales, **68 bytes per token per KV head**"
(§3.2), i.e. 512 code bits (4 bits × 128 channels) plus 32 scale bits (128 channels × 16 bits / 64
tokens) = 544 bits = 68 bytes. The baselines' indexes: Double Sparsity label cache **17 B/token**
(32 channels at 4 bits), a 32-byte-per-token ShadowKV-style landmark index (fp16 mean key per
8-token block, 256 bits), and a 2-bit all-channel thumbnail (288 bits) (§2, §4 byte accounting). The
paper charges the full
68 B/token in its memory comparison and says the store "becomes free only in a stack whose K cache
is already 4-bit and channel-major", a path it "ha[s] not built or measured" (§7 "Memory"). So in
their setup the index is a **separate copy**, made from the K cache keys, keyed by channel and by
64-token block, with no per-token reconstruction except for winners.

---

## 2. The water-filling read-depth rule (§3.3 Eq. 3–4; §3.4; Appendix A)

### 2.1 Inputs — the importance measure

For the group `{q^(h)}_h` of query heads sharing one KV head, the importance of channel `j` is
(Eq. 3):

```
g_j = Σ_h ( q_j^(h) )² · Var( k_j )
```

"with `Var(k_j)` measured once on calibration keys" (§3.3). So the inputs are:

- **per-query** `q_j^(h)` — the query's own component (squared), summed over the `G` query heads of
  the group;
- **per-model, calibrated once** `Var(k_j)` — the channel variance of the keys, measured on
  calibration text (Wikitext-103 train split; §4 "Models and hardware"; §3.3).

It is therefore **not per-channel-magnitude-only** and **not purely calibrated**: the query is
live, the key variance is offline. Appendix A ("Importance weights") works this through: with
`(q_j^(h))²` summed over two heads and `Var = (1.222, 0.755, 0.127, 0.001)`, the importances are
`g = (12.22, 3.77, 0.63, ≈0)`.

### 2.2 Objective and constraint

Objective: minimize the expected squared score error of the read set, `Σ_j g_j 4^{-t_j}` (§3.3) —
i.e. a **score-error-variance** objective, not a latency objective. Constraint: `Σ_j t_j ≤ B`, a
**bit budget** (bit count, not bytes): "Minimizing it under `Σ_j t_j ≤ B` is reverse water-filling"
(§3.3). `B` is per query *group* (shared by the `G` heads that share one KV head) and per layer;
§3.4 calls the same-`B`-for-every-layer case the "flat budget" and a greedy layer-varying case
`B_ℓ ∈ {24,…,128}` the "per-layer plan". The paper does state the derivation's approximation: "with
`a_{β,j}²` standing in for `Var(k_j)`, channel `j` contributes an expected squared score error
proportional to `g_j 4^{-t_j}`" (§3.3) — see the ambiguity note in §7.

### 2.3 How the depth is chosen — closed form plus one scalar search

Integer depths are (Eq. 4):

```
t_j = clip( round( log4( g_j / θ ) ), 0, 4 ),      Σ_j t_j ≤ B
```

"bits go first to the channels with the largest `g_j`, and each channel's depth is set by how far
its importance sits above a common water line `θ`" (§3.3). Since each `t_j` is a non-increasing
step function of `θ`, so is their sum, and:

> "the water line is the smallest `θ` at which the sum fits the budget, and **30 bisection steps on
> `log θ` per query group** locate it." (§3.3)

So the rule is a **closed-form depth map** (Eq. 4) plus a **1-D monotone search for `θ`** (30
bisections), not a table and not a general search. Channels with `t_j = 0` are skipped entirely
("inactive"); the rest are the active channels, and "a read at mean 48 bits touches **30–34 of 128
channels** at 1–4 planes each on the models we test" (§3.3; Table 21 gives 30–31 at mean 48 on Qwen3-8B).

**Cost per query.** The paper gives the *operation count* (30 bisections × 128 channels per query
group) but **does not report a measured latency for the depth-selection step itself** — it is
folded into the scan. This is a real gap for a prototype (see §7). Appendix A confirms the
procedure on a six-key example: the bracket starts at `[0.001, 10]` on `θ`, each trial is the
geometric mean of the bracket ends, and after 30 halvings `θ = 0.079` with depths `t = (4,3,1,0)`
summing to `B = 8`; at `B = 6` it stops at `θ = 0.118`, `t = (3,2,1,0)`.

### 2.4 Basis rule (§3.5)

Water-filling over **raw** channels assumes score variance is concentrated in few channels. On
models with **QK-norm** (Qwen3) it is, and rotating keys with the calibration KLT *raises* error; on
models without QK-norm (Llama-3.1, Qwen2.5) the KLT `V` (score-variance-ordered eigenbasis of the
calibration keys, per KV head and layer) lowers error at equal bits. The rotated variant stores
planes of `(k − μ) V` and uses `g_j = Σ_h ( (q^(h) V)_j )² λ_j`; "nothing else changes", and unlike
Loki nothing is truncated — all `D` coordinates are stored (§3.5). The rule is chosen once per model
by evaluating both stores on calibration text. Ablation A2 (Table 11) quantifies it: on Qwen3-8B at
32k the KLT *raises* mean-48 error from 0.0081 to 0.2908; on Qwen2.5-7B it lowers it from 0.0075 to
0.0044.

### 2.5 §3.6 — the host read, for completeness

With the store in host memory, a word-at-a-time kernel is limited by scattered small transactions,
so "the first `t_j` planes of channel `j` are one contiguous run of `t_j · n/8` bytes" gathered
into a staging buffer and scanned in HBM; "the same transfer is given to every baseline" (§3.6).
This section is *entirely* about host placement and has no device-resident target.

---

## 3. Results and their conditions

### 3.1 The headline: 1.67× at 1M tokens

- **Baseline.** "the 136-bit scans, which is the SparQ r=32, Double Sparsity and Loki byte count"
  (§5.1, Table 2/3 caption). Also reported against a block-landmark index (2.50×) and a 2-bit
  thumbnail (3.12×) at 1M (§5.1; Table 1).
- **What is counted.** **GPU time per decode step** = "the profiler's kernel plus memcpy time of a
  decode step, because it measures the work the method changes" (§4 "Offload harness"). Wall-clock
  is reported separately (median of timed steps) and is *worse* for Fathom: 1.38× vs 1.67× at 1M,
  because the Python harness adds host-side overhead that "differs by method and would not exist in
  a fused implementation" (§4, §5.1, A6). "the primary timing metric is GPU time" (§4).
- **Hardware / setup.** One **NVIDIA A100-SXM4-80GB** pod, **PCIe 4.0** host link, 2 TB host RAM
  (RunPod secure cloud; Appendix E also lists 16 vCPU, PyTorch 2.8, Triton 3.4, Transformers
  4.56.1, CUDA 12.8). Qwen3-8B, `k = 512`, **batch 1**, **synthetic KV** (calibration keys tiled to
  `n`, random values) at every point of Table 2 and Figure 2 because "no model here prefills a
  million tokens" (§4). Rows *and* the scan index in **pinned host memory**.
- **Numbers (Table 2).** Fathom 48 (56 measured bits) GPU 177 ms/step at 1M vs 296 ms for the
  136-bit scan (ratio 1.67), 186 ms for SparQ r=16 (68 bits, ratio 1.05), 444 ms landmark (2.50),
  554 ms thumbnail (3.12). PCIe traffic per step: 2.32 GB (Fathom 48) vs 5.33 GB (136-bit) vs
  2.76 GB (SparQ r=16). At 256k the ratio over the 136-bit scan is 1.37×; at 128k with a **real
  prefill** it is 1.26× (Table 3, left), and 1.00× vs SparQ r=16. Component split at 1M (Table 4):
  Fathom 48 scan transfer 78 ms + scan kernel 21 ms = 99 ms "scan total", vs 197+20 = 216 ms for
  the 136-bit scan; every scan "moves its bytes at the link rate, about 26 GB/s" (§5.1).
- **The 18% figure.** "in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer
  bytes with lower attention error on six of seven model and context settings" (Abstract; §5.2:
  "At equal time Fathom reads 18% fewer scan bytes"; 56 vs 68 bits at matched time).

### 3.2 The "matches exact top-k" claim (§5.5, Table 7; §5.6)

- **Metric.** RULER-style tasks: "Score is the fraction of gold strings present in the
  generation" (§4 "RULER-style tasks") over RULER needle/tracking templates on a Wikitext-103
  haystack; prompts are plain completions **without the chat template** (§4, Appendix B).
- **Conditions/tasks.** Qwen3-8B at 32k, `k=128`, **40 samples/task**, six tasks (single, multi-key,
  multi-value, multi-query NIAH, variable tracking, frequent-word extraction); Qwen2.5-7B-Instruct-1M
  at 128k, `k=128`, **20 samples/task**, four tasks (Appendix B, Tables 12–13).
- **What is compared.** Mean task score, method vs "exact top-k oracle" (and dense). Table 7:
  Qwen3-8B 32k — oracle 0.874 ± 0.014, dense 0.884, Fathom 48 (56 bits) 0.881 ± 0.014, Fathom 64
  (74 bits) 0.878 ± 0.014; Qwen2.5-7B-1M 128k — oracle 0.660 ± 0.033, dense 0.677, Fathom KLT 48
  0.671 ± 0.032, KLT 64 0.676 ± 0.035. The text: "Every per-token scan lies within **0.008** of the
  exact top-k oracle at 32k and within **0.025** at 128k, inside that error" (standard error ≈0.014
  and 0.034), and — the caveat that matters most — "**No per-token scan separates from another at
  this `k`; differences between them appear only in attention error and in bytes.**" The block
  landmark is the only one that separates (loses 0.043 at 32k, 0.130 at 128k).
- **The stronger, separating evidence is §5.6** (real OpenHands/SWE-rebench sessions, Qwen2.5-7B-1M,
  ~95k tokens): metric = **word-level sequence-match ratio between the method's decoded step and the
  exact top-k step** (step agreement). At `k=512` (Table 8): Fathom 92 bits 0.60 ± 0.05, equal to
  SparQ r=32's 0.60 ± 0.04 at 136 bits (paired −0.00 ± 0.04), 32% fewer bytes; Fathom 56 bits 0.53,
  74 bits 0.54; Double Sparsity 136 bits 0.55; SparQ r=16 68 bits 0.49; landmark 0.42. At `k=2048`
  (Table 9): Fathom 56 bits 0.67 ± 0.06 vs SparQ r=16 0.49 ± 0.05, landmark 0.47. So the abstract's
  "matches exact top-k decoding" is (i) a saturated synthetic-task observation and (ii) a
  *fidelity*-not-*identity* result on real sessions — agreement 0.60–0.67, not 1.0. It compares
  **whole decoded steps**, not block ids, and not recall of selected blocks.

### 3.3 Equal-error bits and the plane-budget ablation

- Table 6 (§5.3): to reach Double Sparsity c=32's *error* (136 bits), Fathom reads **46–74 bits** on
  the 7 settings (a 1.8–2.9× saving); to reach SparQ r=32's error, **38–92 bits**. Loki r=32 (136
  bits) is matched at 56–75 bits on the 4 settings where it is not rank-limited.
- The plane-budget ablation is **A4** (§6): "The 2-bit thumbnail reads every channel at two planes,
  288 bits per token. Its error … is reached by the water-filled read at **28–148 bits** on all 7
  settings … Uniform depth spends bits on channels that carry no score for the query." Fixed-budget
  sweeps (mean 40/48/64/80/128) appear in Tables 14–15; A1 (§6, Table 10) is flat vs per-layer
  budget; B1 (Appendix D, Table 19) shows the *store* precision (uniform 4-bit vs 8-bit vs
  rate-allocated 128-bit) barely moves error at a fixed read budget (0.0052 at 56 bits for both
  4- and 8-bit stores).

---

## 4. The caveats, quoted

### 4.1 The "no gain when the index is device-resident" clause, in full and in context

Abstract, last sentence:

> "The store is the 4-bit K copy a quantized serving stack already holds, and **the method is not
> faster when the index is resident in GPU memory**."

§5.1, "Control: index in HBM" (Table 3, right; Figure 8):

> "If the same indices are kept in HBM and only the rows are offloaded (Table 3, right; Fig. 8),
> the per-token scans land at **45–48 ms per step at 128k**, the landmark index at 44 ms and the
> thumbnail at 55 ms, and **ours is not faster**. The time advantage therefore **requires the index
> itself to live in the slower tier**, which is the case of long contexts with many concurrent
> sessions, where a 17-byte-per-token label cache (5.1 GB per million-token sequence for Qwen3-8B)
> or a 32-byte-per-token landmark index (9.7 GB) does not fit next to the weights."

§7, "HBM-resident scan kernel" (Table 17, Figure 9) — *why*:

> "With everything in HBM the byte saving does not become time (Table 17, Fig. 9). The planes
> kernel extracts one bit per shift-and-mask and applies `G` multiply-adds per extracted bit, where
> a 4-bit nibble carries four times the information per extracted element, so it reaches a fraction
> of the bandwidth the nibble scans reach. The 32-channel scan's kernel time is **0.73× ours at 32k
> and 0.70× at 128k; ours is slower in the kernel despite reading 38% fewer bytes** on these
> layers."

§7 "Arithmetic per byte" adds the per-bit cost the excerpt omits: "Bit extraction adds one to two
integer operations per bit read, four times the per-bit cost of a nibble scan." §5.1 states the
kernel cost in the same terms: "its scan kernel costs about twice SparQ's per byte because it
extracts bits."

Table 17 (HBM stores, A100, layers 0/1/2/18/30, batch 4×8 KV heads, CUDA graphs, median of 30):
Fathom 48 (84 bits) scan 0.140 ms at 32k / 0.415 ms at 128k, reaching **75/102 GB/s**; SparQ r=16
(68 bits) 0.060/0.164 ms, 148/218 GB/s; 32-channel (136 bits) 0.102/0.291 ms, 175/245 GB/s; full
4-bit (544 bits) 0.330/1.010 ms, 217/282 GB/s. Figure 9: "The planes kernel reaches a third to a
half of the channel scans' bandwidth; none of them approaches the 2 TB/s peak."

§7, "Arithmetic per byte" — the reversal condition:

> "… an A100 offers about 5 integer operations per byte … so a scan that is arithmetic-bound on the
> A100 is arithmetic-bound on any current data-centre GPU; the HBM-resident behaviour above is a
> property of the operation, not of the card. Amdahl's law compounds it … **Over PCIe the ordering
> reverses.** Over PCIe every method runs at the link's rate, the arithmetic is idle, and the byte
> ratio approaches the time ratio (Table 2)."

§8 (Limitations):

> "**With the scan index resident in HBM the method is not faster than the 4-bit channel scans; its
> time advantage requires the index to live in a slower tier.**"

§9 (Conclusion):

> "**Where everything sits in HBM the method is not faster, because the scan is arithmetic-bound
> there and reading fewer bits does not remove multiply-adds; a fused kernel would narrow the gap
> but not remove the arithmetic bound.** The method fits serving stacks that already keep a 4-bit K
> cache and hold it in a slower tier."

Table 1 rows make the same point: for "rows in host memory, index in HBM" the binding term is "the
shared row fetch" and the result is "not faster; all per-token scans land at 45–48 ms"; for
"everything in HBM" the binding term is "arithmetic per scanned bit" and "the scan kernel is 1.4×
slower than a nibble scan".

### 4.2 Other caveats the note must carry

- **Index dtype / how it was produced.** Uniform 4-bit symmetric per-channel code, one fp16 scale
  per 64-token block, made from the K cache keys (§3.2). The *store* is placement-independent; the
  paper's own baselines (Double Sparsity label cache, landmark, thumbnail) are 4-bit / fp16 codes
  with `k∪` row fetches (§2, §4 "Baselines", §4 "Byte accounting": "A method's bits per token are
  the code bits it reads plus 16 bits per 64 tokens for the fp16 block scale of every channel it
  reads").
- **Key distribution dependence.** The method's premise is concentrated score variance. It holds on
  QK-norm models (Qwen3) with raw channels and needs a KLT rotation on Llama-3.1 / Qwen2.5 (§3.5,
  A2). If variance is not concentrated, water-filling has nothing to allocate (§3.2). No key
  distribution beyond per-channel variances is reported.
- **Per-layer budget is deployment-specific.** "a plan calibrated at 16k evaluated at 32k is worse
  than the flat budget"; recommendation is a flat budget by default (§3.4, A1).
- **Baseline caveats.** Landmark is a ShadowKV reimplementation "in a setting … that ShadowKV and
  Quest do not use"; CPU-side retrieval (RetroInfer/MagicPIG) is not compared; timing beyond 128k
  uses synthetic KV; Triton kernels are prototypes with per-method Python issue cost; fidelity is
  **one held-out window per setting** and RULER scores have standard errors of a few points
  (§8). Offload tables are batch 1 with one batch-2 check; the many-session motivation "is argued
  from index sizes rather than measured" (§8). The 128k instruct runs used no chat template (§8).
- **The paper's own recommended target.** "The method fits serving stacks that already keep a 4-bit
  K cache and hold it in a slower tier" (§9).

---

## 5. Applicability to this build, mechanism by mechanism

Built reality (from `docs/research/qsa-index-built-reality.md`, its `file:line` citations not
re-derived here): the QSA pooled index plane is **device memory in every space**; `--kv-host-tier`
does not exist (grep of `apps/ include/ src/` this session finds it only in
`docs/maintainer/qwen3_8-flash-next-design.md:2709`); the index is a **runtime activation**, not an
artifact object; it is **BF16**, one 128-wide key per **4-token block** per QSA layer, plane
`[Di/R, 64, 1, pages]` (`include/infernix/ops/qsa.h:35-36, :85`), produced by
`mean over R=4 raw keys → RMSNorm → RoPE` (`src/ops/qsa/qsa.cu:151-160`; contract
`include/infernix/ops/qsa.h:21, :30-33`), scored by
`score_b = (1/√Di) Σ_h ReLU(⟨index_q_h, pooled key_b⟩)` (`src/ops/qsa/qsa_select.h:10-11`),
top-512 = budget 2048 / ratio 4 blocks, 13 planes/lane, ≈104 MiB/lane at 128K.

| Fathom mechanism | § | Transfers to a device-resident BF16 QSA index? |
|---|---|---|
| **Channel-major bit-plane layout + exact prefix quantizer** | §3.2, Eq. 2 | **Only as a re-encoding, and only for a uniform integer code.** Infernix's plane is BF16; a BF16 prefix is sign+exponent, which the paper explicitly excludes from the exactness and `4^{-t}` law (§3.2). To get the property you must quantize the pooled key to a uniform 4-bit per-channel code with a per-block scale — a **representation change to the runtime plane**, not a prefix of the existing BF16. The *accuracy* question ("does a `t`-bit prefix keep the selected blocks?") then transfers and is the primary thing to measure. |
| **Per-query water-filling depth rule** | §3.3, Eq. 3–4 | **The rule transfers and is bandwidth-agnostic**; its objective is `Σ_j g_j 4^{-t_j}` under a bit budget, with no bandwidth term. The inputs map: `g_j = Σ_h (q_j^(h))² Var(k_j)` becomes the QSA indexer's `Σ_h (index_q_h,j)² Var(pooled_k_j)` over the indexer's heads. **But the ReLU changes the error propagation**: the paper's score is a plain `q·k` with no gate (Eq. 3), so (our reading) every head contributes to the error it models; Infernix's score sums `ReLU(·)`, so heads with a negative pre-activation contribute no signal and the effective `g_j` is query- and block-dependent (see §6). Its *reason to exist* — converting byte savings into time — does not transfer: §7 says the planes kernel is arithmetic-bound device-resident ("1.4× slower than a nibble scan"; 0.73×/0.70× at 32k/128k) and "reading fewer bits does not remove multiply-adds". |
| **Per-layer budget (§3.4)** | §3.4 | Transfers as a tuning choice; needs calibration at the deployment context length (§3.4, A1). |
| **Basis rule (§3.5)** | §3.5 | The QSA pooled key is **RMSNorm'd** and then RoPE'd, i.e. approximately normalized per block — the same regime in which the paper chooses **raw channels** (Qwen3/QK-norm). So the expected branch is raw, not KLT; but RoPE mixes channel pairs and the pooling already averages 4 tokens, so the channel-variance structure of a *pooled* key is not the same as the raw-key structure the calibration assumes — **unverified from the paper**. |
| **Host-resident read / gather (§3.6)** | §3.6 | **No target.** The index is device-resident; there are no contiguous host runs to gather. |
| **Headline 1.67× GPU-time speedup** | §5.1 | **Does not transfer.** It is conditioned on `the index itself … live in the slower tier` (§5.1) and the paper states the HBM-resident case is not faster (§7, §8, §9). |

**Which parts of the framing rest on the host/PCIe assumption.** The *cost model* (§3.1 Eq. 1:
"Reducing `b` matters exactly when `b n/8` is the largest of the three terms, which is the
long-context, many-session regime this paper targets"), the *offload harness* (§4), *all of §5.1*
(Table 2 converts bits to time via "the link rate, about 26 GB/s"), §3.6, §7's "over PCIe the
ordering reverses", and §9 rest on the PCIe/host assumption. The *encodings and rules*
(§3.2–3.5, Eqs. 1–4) and the *accuracy results* (§5.3–5.6, §6) do **not** — they are about bytes,
score error, and selection quality wherever the index lives. The 1.67× number, the 26 GB/s link
arithmetic, Table 2's GB/step column, and Table 4's transfer-vs-kernel split are the host-specific
artifacts.

**A distinct saving that is *not* Fathom's.** Fathom's store is a 4-bit copy; its HBM control
compares 4-bit planes vs a 4-bit nibble scan, so §7's "no gain" is the statement that *per-query
read depth* is not a device win. This build's index is **16-bit BF16**, so a move from BF16 to a
uniform 4-bit per-block code is a **4× compression of bytes and frames at full depth** — a
magnitude difference the paper never measures: its store-precision ablation varies only
4-/8-/128-bit stores at a fixed read budget and finds error barely moving (0.0052 at 56 bits for
both the 4- and 8-bit stores; Table 19), and its fp16 baselines are block/landmark-mean or low-rate
stores, not an all-channel per-token channel-major 16-bit store, so they do **not** bound a
BF16→uniform-4-bit change at equal bytes. Whether that compression returns time is a separate
question, and the built-reality note already shows the
current scan is latency-dominated (2× time over 16× bytes from 8K→128K) rather than
bandwidth-bound. So: the paper cannot license a device-resident *time* win, and the byte win of
going BF16→4-bit is a different claim that must be measured, not inferred from Fathom.

---

## 6. What a faithful prototype would have to implement

Scope it as **accuracy-first**, since that is the property that transfers. Concretely:

1. **Capture the pooled index plane.** Instrument the existing path so that, per QSA layer, the
   FP32/BF16 pooled keys (`layer.pooled_pages`, written by `pool_kernel`, `src/ops/qsa/qsa.cu:1005`)
   and the index queries (`qsa_index_query`) are recorded for a held-out workload. This is the
   missing data the built-reality note names: "the captured pooled keys are not recorded anywhere
   today."
2. **Profile channels.** For each QSA layer and index head, compute `Var(k_j)` over a calibration
   slice of the captured pooled keys ("calibrated once on calibration keys", §3.3). Also record the
   per-block `a_{β,j}` distribution, since it is what Eq. 2 uses.
3. **Build the plane store.** Re-encode each 128-wide pooled key **per block** with the uniform
   symmetric code of Eq. 2 (`s = a_{β,j}/8`, `c = clip(floor(k/s), −8, 7) + 8`, MSB = sign),
   packing bit `p` of the channel's codes into a 64-bit-per-block word, channel-major (block ⟷ the
   paper's "token"; the paper's 64-token plane word becomes a 64-*block* word here). Store the fp16
   per-block scale per channel. Then `c^(t) = c >> (4−t)` must reproduce the `t`-bit mid-rise
   quantizer of §3.2 — **verify this identity directly** (it is the whole of the encoding claim and
   is cheap to check exactly).
4. **Prefix scoring.** For each query, for each channel `j`, form the `t`-bit reconstruction and the
   dot product with `index_q_h,j` summed over the index heads, exactly as `qsa_score_kernel` does
   today (`src/ops/qsa/qsa_select.cu`), including the `ReLU` and the `1/√Di` factor and the
   ascending-block-id tie rule (`src/ops/qsa/qsa_select.h:10-13`).
5. **Depth selection.** Implement Eq. 4: importance `g_j = Σ_h (index_q_h,j)² Var(k_j)`; find the
   smallest `θ` with `Σ_j clip(round(log4(g_j/θ)), 0, 4) ≤ B` by bisection on `log θ`; `B` a flat
   per-layer bit budget. Measure the cost of this step — **the paper does not report it**. Decide
   explicitly how to treat the `ReLU`: either (a) the paper's smooth form (sum `g_j` over all heads,
   ignoring the gate), or (b) a gated form that only sums heads whose `ReLU` pre-activation is
   positive. (b) is faithful to Infernix's score and is an open design question the paper does not
   settle.
6. **Agreement metric.** Score against an FP32/FP64 oracle that selects the exact top-512 blocks
   from the full-precision (BF16, then FP64-decoded) pooled keys, on the same queries. Report, per
   depth `t` (or per budget `B`) and per layer: (i) **selected-block overlap / recall** — fraction
   of the oracle's top-512 block ids present in the truncated selection, and Jaccard; (ii) rank
   fidelity of the truncated scores (e.g. Spearman on the union of candidates); (iii) downstream
   task score if available. The built-reality note is explicit that no selection-agreement harness
   exists and that `--kl-reference` compares vocab top-k, not block selection; the natural oracle
   substrate is `tools/flash_next/reference.py:266-294`.
7. **Sweep the budget.** Reproduce the paper's plane-budget ablation form (A4 / Table 14): agreement
   vs mean bits, flat budget, and a per-layer plan as a second point. Include the degenerate uniform
   `t=2` ("thumbnail") and `t=4` (full) as bracket points, and the BF16 full-precision index as the
   zero-error end.
8. **Only then measure time**, and expect the paper's §7 result: fewer index bytes, an
   arithmetic-heavier kernel, and no time win device-resident. A byte/frame win at full depth
   (BF16→4-bit) is a separate, plausible-but-unevidenced claim.

**Not in a faithful prototype:** any host-read/gather path (§3.6), any PCIe bandwidth model, any
reliance on `--kv-host-tier` (it does not exist), and any converter change — the plane is a runtime
activation, so the writer `pool_kernel`, the Program's plane allocation, every QSA reader
(`attention_kernel`, `qsa_prompt.cu`), and the op tests all change, not the container.

---

## 7. What I could not establish from the paper

- **No measured cost for the depth-selection step.** The 30-bisection-on-`log θ` search per query
  group (§3.3) has no latency figure; the paper never says where it runs — our inference is that it
  is absorbed into "scan kernel" (Table 17). Not derivable.
- **The `Var(k_j)` vs `a_{β,j}²` conflation.** §3.3 derives `Σ_j g_j 4^{-t_j}` "with `a_{β,j}²`
  standing in for `Var(k_j)`", but `g_j` as defined (Eq. 3) *contains* a calibration-time
  `Var(k_j)` that is per-channel and shared across blocks, while `a_{β,j}` is per-block. The paper
  does not state whether the prototype should use the calibration `Var`, the per-block `a`, or a
  blend. **Ambiguous as written.**
- **Per-query cost of the whole scan vs per-model.** The bit budgets are labelled by a "mean" (40
  /48/64/80/128) and the *measured* bits (Table 2: mean 48 → 56 bits) include scales; the paper does
  not give a closed form for the label and it is not needed, but the two are visually easy to
  confuse.
- **Key distribution.** Only per-channel variances and the active-channel counts (Table 21) are
  reported; the actual pooled/ raw key distribution and the residual-error shape are not, so whether
  the `4^{-t}` law holds on a mean-pooled, RMSNorm'd, RoPE'd key is **untested** here.
- **How `ReLU` interacts with the error model.** The derivation (§3.3) has no ReLU gate; the score
  definition does (Infernix) or does not (Fathom's plain `q·k`). The paper is silent on gated scores.
- **Pre- vs post-RoPE quantization.** Fathom quantizes the K *cache* keys; it contrasts KVQuant's
  pre-RoPE quantization (§3.2) but does not state whether its own keys are pre- or post-RoPE. The
  QSA index key is **post-RoPE**, which mixes channel pairs and could change which channels are
  "active". Not addressed.
- **Page/table numbers absent** in the HTML rendering; citations are section/equation/table only.

---

## 8. Load-bearing facts for the map

1. **Quantization rule (Eq. 2, §3.2):** `c = clip(floor(k/s), −8, 7) + 8`, `s = a/8`,
   `a = max|k|` per 64-token block; MSB = sign; prefix `c >> (4−t)` is **exactly** the `t`-bit
   mid-rise uniform quantizer of `[−a, a)`, cell width `a/2^{t−1}`, error variance `a²/(3·4^t)` —
   **for a uniform code only**; FP8/float prefixes are explicitly excluded.
2. **Depth rule (Eq. 3–4, §3.3):** inputs `g_j = Σ_h (q_j^(h))² Var(k_j)` (live query, calibrated
   `Var`); objective `min Σ_j g_j 4^{-t_j}` subject to a **bit** budget `Σ_j t_j ≤ B`; closed form
   `t_j = clip(round(log4(g_j/θ)), 0, 4)` plus a **30-step bisection on `log θ`** per query group.
   **Bandwidth-agnostic as stated**; the paper reports no latency for it.
3. **Reported numbers:** 1.67× **GPU time** per decode step at 1M tokens vs the 136-bit scans
   (SparQ r=32 / Double Sparsity / Loki byte count), A100-SXM4-80GB, PCIe 4.0, Qwen3-8B, `k=512`,
   batch 1, **synthetic KV**, rows **and index** in pinned host memory (Table 2); 2.50× vs landmark,
   3.12× vs thumbnail; 1.37× at 256k; 1.26× at 128k real prefill; 18% fewer bytes than SparQ r=16
   at matched GPU time. RULER-style: every per-token scan within 0.008 (32k) / 0.025 (128k) of the
   exact top-k oracle and **unseparated from each other** at `k=128`. Real sessions: 0.60 step
   agreement at 92 bits = SparQ r=32's 0.60 at 136 bits (`k=512`); 0.67 at 56 bits vs SparQ r=16's
   0.49 (`k=2048`). Equal-error bits: 46–74 vs Double Sparsity's 136, 38–92 vs SparQ r=32.
4. **Device-resident caveat (verbatim, §5.1):** "The time advantage therefore requires the index
   itself to live in the slower tier…"; (§7) "ours is slower in the kernel despite reading 38% fewer
   bytes"; (§8) "With the scan index resident in HBM the method is not faster than the 4-bit channel
   scans"; (§9) "reading fewer bits does not remove multiply-adds." HBM scan kernels: 75–106 GB/s
   vs 148–282 GB/s for nibble scans (Table 17).
5. **Applicability verdict:** the *encoding accuracy question* ("does a `t`-bit uniform prefix keep
   the selected blocks?") transfers and should be measured; the *depth rule* transfers as a
   bandwidth-agnostic bytes/accuracy tradeoff but not as a device speed mechanism; the *1.67×
   speedup* and all of §5.1/§3.6 rest on the host/PCIe assumption and **do not** transfer. Because
   this build's index is BF16 16-bit while the paper never varies an all-channel per-token 16-bit
   store against a 4-bit one (its store ablation over 4-/8-/128-bit stores holds the read budget
   fixed and barely moves error, Table 19; its fp16 baselines are block/landmark-mean or low-rate
   stores), Fathom's HBM control does **not** bound the BF16→4-bit compression — that is a separate,
   unevidenced claim.

**Where the paper's claims depend on the host-resident assumption the map must account for:** the
cost model (Eq. 1), the entire offload harness (§4), the bits→time conversion (§5.1, Table 2's
GB/step and Table 4's transfer/kernel split), §3.6, the "ordering reverses over PCIe" argument
(§7), and the 1.67×/1.37×/1.26× ratios and §9 conclusion. The map should cite the paper's
*encoding and accuracy* results as applicable and its *speed* results as explicitly inapplicable to
a device-resident index, per the paper's own §7–§9.
