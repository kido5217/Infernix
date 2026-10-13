# Per-layer residency skew vs global LFRU — replay prototype (#12)

Research note and prototype for wayfinder ticket [#12](https://github.com/kido5217/Infernix/issues/12),
run against the frozen protocol (issue #10, first comment) on the eight version-2 `NRTRACE1`
coding-agent traces under `out/route-traces/`.

**Question.** At the deployed frame capacity, does any per-layer residency skew beat the engine's
global LFRU on *live routed VRAM misses per generated token*, and by how much per 1 % of the pool?

**Answer.** No per-layer skew beats global LFRU. The paper's capacity allocator (arm a) is a wash
(+0.05 %); the entropy score-weight skew (arm b) is neutral at best (−0.21 % at the gentlest
amplitude) and a loss above it; the sequential-replay control alone costs +8.9 %, larger than any
skew's effect beyond noise. The one arm that clears the protocol's ≥ 5 % bar is the warm-start
*prior* (arm c, −19.2 %), and none of its gain is skew *shape*: adding the entropy shape to it makes
it 6 points worse.

## Prototype and calibration gate

`tools/expert_cache_replay/per_layer_skew.py` is a VRAM-only replay of the frame pool that **mirrors**
`host_tier.py`'s policy rather than forking the file: it imports `host_tier.read_trace` and reproduces
the engine's LFRU score `f / (age + 1)`, the recorded per-round promotion budget, the one-round copy
visibility, and the event-atomic grouping. It runs in full-RAM mode (`--ram-slots all`, prior none) —
the deployment's `--expert-ram-mib 66000` (66000 MiB > 48 × 512 × 2,764,800 B) holds the whole key
space, so the RAM tier never binds and this is exactly `host_tier.py`'s own `--validate` regime
(which reproduces the engine's logged "since start" hit rate, 86.17 % vs 86.8 % on long-1). A miss is
a live routed expert absent from a frame; the metric is misses / generated tokens, generated tokens =
`Σ tokens` over Decode + Verify + Forced rounds.

**Gate (§1).** The uniform-LFRU event-atomic arm reproduces
`python3 tools/expert_cache_replay/host_tier.py <trace> --vram-budget recorded --vram <V>
--ram-slots all --priors none` for each trace's own header frames, pooled and per session:

| session | V | prototype | host_tier | rel |
|---|---:|---:|---:|---:|
| long-1 | 8178 | 47.433615 | 47.433615 | 0.0000 % |
| long-2 | 8143 | 58.497373 | 58.497373 | 0.0000 % |
| long-3 | 8142 | 63.634991 | 63.634991 | 0.0000 % |
| long-4 | 8142 | 58.142310 | 58.142310 | 0.0000 % |
| short-1 | 8142 | 122.526502 | 122.526502 | 0.0000 % |
| short-2 | 8142 | 115.207709 | 115.207709 | 0.0000 % |
| short-3 | 8197 | 109.416139 | 109.416139 | 0.0000 % |
| short-4 | 8197 | 115.881107 | 115.881107 | 0.0000 % |
| **pooled** | | **66.142014** | **66.142014** | **0.0000 %** |

Gate passes (bar 0.1 % relative); decisions are step-for-step identical.

## Arms

- **(iv) uniform** — the calibrated global LFRU, the control.
- **(v) sequential** — the same traces under per-access (not event-atomic) semantics: every routed
  access is one policy tick with no in-group protection. The replay-semantics control.
- **(a) quota / allocator** — MoE-CORE §3.1 incremental marginal-loss transfer on per-layer hit-rate
  response curves `Ĥ_l(K)` (replayed per layer, same budget rule), `K_l` summing to `V`.
  `D⁻_l = Ĥ_l(K_l) − Ĥ_l(K_l−ΔK)`, `G⁺_l = Ĥ_l(K_l+ΔK) − Ĥ_l(K_l)`; donors `D⁻_l < ε_drop`,
  receivers `G⁺_l > τ_low` (τ_low = median `G⁺`), transfer ΔK from donor to receiver while
  `G⁺_r > D⁻_d`. **`quota_even`** is the even split of the same total capacity (hard partition,
  no allocator) — the shape-free quota control.
- **(b) score weights** — shared pool with LFRU score `w_l · f/(age+1)`, `w_l = exp(A·z_l)` where
  `z_l` is the pooled per-layer routing entropy standardised across layers. Reported at `A` ∈
  {0.25, 0.5, 1.0}; **`w_permuted`** is the same weight multiset randomly permuted across layers
  (seed 17), separating "some non-uniform weight" from "this entropy shape".
- **(c) warm-start seed** — the admission-time domain key: the VRAM starts with the top `0.5·V`
  experts ranked by usage over the *other three* sessions of the same long/short category
  (leave-one-out profile), each seeded with LFRU count 16 and clock 0 (the engine's `kSeedCountCap`
  / `seed` semantics). `seed+w_entropy` adds the amp-1 entropy weights.

The allocator's response curves are computed on the **pooled eight sessions** (in-sample, i.e.
optimistic for arm (a)); all other arms are per-session.

## Results — pooled misses / generated token

| arm | deployed | 4008 | 6347 | 8000 | 8351 | 10200 | 12000 | Δ deployed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **uniform (iv)** | 66.142 | 143.878 | 91.345 | 67.916 | 64.171 | 50.500 | 44.302 | 0.00 % |
| sequential (v) | 72.032 | 156.466 | 99.713 | 73.984 | 69.855 | 54.687 | 47.603 | **+8.91 %** |
| quota_even | 69.152 | 144.652 | 93.805 | 70.976 | 67.212 | 52.829 | 45.613 | +4.55 % |
| quota (a) | 66.175 | 143.966 | 91.415 | 68.021 | 64.251 | 50.594 | 44.393 | **+0.05 %** |
| w_ent 0.25 | 66.006 | 144.343 | 91.327 | 67.791 | 64.056 | 50.460 | 44.278 | −0.21 % |
| w_ent 0.5 | 66.693 | 147.000 | 92.621 | 68.513 | 64.680 | 50.777 | 44.426 | +0.83 % |
| w_ent 1.0 (b) | 70.034 | 157.522 | 98.589 | 72.038 | 67.801 | 52.204 | 45.039 | **+5.88 %** |
| w_permuted | 73.650 | 157.032 | 101.929 | 75.724 | 71.320 | 54.310 | 45.864 | +11.35 % |
| seed (c) | 53.471 | 137.859 | 81.460 | 55.537 | 51.173 | 33.322 | 22.486 | **−19.16 %** |
| seed+w_entropy | 57.439 | 150.154 | 88.181 | 59.670 | 54.943 | 35.283 | 23.405 | −13.16 % |

*(all cells are pooled over the eight sessions at the stated global capacity; `deployed` uses each
 trace's own header frames.)*

Per-session, deployed (misses/token):

| arm | long-1 | long-2 | long-3 | long-4 | short-1 | short-2 | short-3 | short-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 47.43 | 58.50 | 63.63 | 58.14 | 122.53 | 115.21 | 109.42 | 115.88 |
| sequential | 50.96 | 66.88 | 70.97 | 65.24 | 128.89 | 123.20 | 117.24 | 122.40 |
| quota (a) | 47.52 | 58.46 | 63.74 | 58.23 | 122.37 | 115.45 | 109.05 | 115.88 |
| w_ent 0.25 | 47.31 | 58.33 | 63.55 | 58.18 | 122.25 | 114.94 | 108.93 | 115.82 |
| w_ent 1.0 | 52.36 | 63.20 | 67.87 | 62.33 | 123.52 | 115.84 | 110.10 | 117.02 |
| seed (c) | 43.20 | 48.34 | 49.17 | 47.54 | 90.75 | 77.73 | 79.64 | 86.43 |

**Per-layer entropy statistic.** It varies: pooled `H_l` spans **6.981 .. 8.482 bits** (range
1.50 bits, std 0.332) across the 48 layers — the low-entropy (concentrated) layers 15, 31, 40, 32,
39 and the high-entropy (spread) layers 0, 2, 1, 22. This reproduces the per-layer-stats note. So the
skew *input* is real; the skew *mechanism* is not a win.

## Skew shape versus prior

- **Shape.** Any non-uniform weight sets a layer's admission ranking against every other layer in
  the shared pool. At `A = 0.25` (weights 0.52–1.62:1) the effect is −0.21 % — inside the replay
  noise measured by arm (v) — and it turns into a loss as `A` grows (`+0.83 %` at 0.5, `+5.88 %` at
  1.0, weights 0.075–6.91:1). The entropy **ordering carries real signal** — at the same amplitude
  1.0, entropy weights beat the same weights *permuted across layers* by 3.6 points (70.0 vs
  73.7) — but the signal points the wrong way against the unpartitioned pool: the correct shape is
  no shape.
- **Prior.** The seed is worth −19.16 % pooled, and the gain is a warm-start head start under the
  throttled recorded budget: it is large where the cache has little runtime to fill (short-1
  −25.9 %, short-2 −32.5 %) and small where it does (long-1 −8.9 %, long-3 −22.7 %).
- **Separated.** `seed` (flat weights) is −19.16 %; `seed + entropy weights` is −13.16 %. The skew
  shape *subtracts* 6 points from the prior. Every point of the winning arm's gain is the prior.
- **Capacity shape specifically.** Arm (a) moves `K_l` between 106 and 254 at V = 8142 while
  `sum(K_l) = V`. Measured as the capacity actually handed from one layer to another —
  `sum(max(even_l - K_l, 0))` = **525 slots = 6.45 % of the pool**, not the 1,050 that
  `sum|K_l - even_l|` double-counts — the pooled metric moves +0.05 %. **Per 1 % of pool
  reallocated, the marginal effect is ~0.0078 %** — indistinguishable from zero at this capacity.
  (Equal partition, which moves the same capacity without fitting it, costs +4.55 %; the allocator
  recovers that and lands back at parity.)

## Arm (a) capacity vector and threshold sensitivity

`K_l` at V = 8142 (sum 8142, even split 170):

```
254 212 212 149 170 191 212 212 170 170 170 170 212 191 170 107
128 170 191 170 170 191 212 149 170 170 170 149 191 170 169 106
127 169 190 148 148 190 211 127 127 148 169 127 148 148 148 169
```

The allocator gives slots to the **spread (high-entropy)** layers 0, 6, 7, 22 (254/212/212/212) and
takes them from the **concentrated (low-entropy)** layers 15, 31, 40 (107/106/127) — the same
direction as the entropy weights, which also weight high-entropy layers up. The concentrated layers'
response curves have already saturated at the even-split capacity, so they are the donors; the
spread layers are still steep and are the receivers.

Allocator defaults at V = 8142: `ΔK = 21`, `ε_drop = 0.03734`, `τ_low = 0.02830` (the medians of
`D⁻`/`G⁺` at the even split; even-split `G⁺` 0.00898–0.04730, `D⁻` 0.01293–0.05465 hit-rate points).
Pooled deployed misses/token under each variant (uniform = 66.142, even quotas = 69.152):

| ΔK = K₀/16 | ΔK = K₀/8 | ΔK = K₀/4 | ε_drop = 0 | ε_drop = median | ε_drop = 0.01 | τ_low = 0 | τ_low = median | τ_low = 0.01 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66.118 (−0.04 %) | 66.175 (+0.05 %) | 66.411 (+0.41 %) | 69.152 (+4.55 %) | 66.175 (+0.05 %) | 69.152 (+4.55 %) | 66.175 (+0.05 %) | 66.175 (+0.05 %) | 66.175 (+0.05 %) |

`ΔK` barely matters (the curves are nearly linear over ±ΔK). `ε_drop` matters and the median is the
right choice: with no donors below the threshold the transfer loop never runs and the arm degenerates
to the even split. `τ_low` is inert — the `G⁺_r > D⁻_d` pair test, not the receiver threshold,
decides which transfers happen. No setting beats global LFRU.

## Verdict and caveats

- **The protocol's ≥ 5 % offline bar is not met by any per-layer skew.** Arm (a) is a wash, arm (b)
  is neutral-to-negative, arm (v)'s replay-semantics penalty (+8.9 %) is larger than any skew
  effect. MoE-CORE's own +1.1 % capacity ablation is consistent with this.
- **Hard equal per-layer quotas lose 4.55 %** — the 3–34 % partition loss the design record already
  reports. The MoE-CORE transfer recovers almost all of it (+0.05 % vs global), but only back to
  parity.
- **Only the warm-start prior wins**, and that is arm (c) — not a per-layer skew. It clears the bar
  pooled (−19.2 %) and for every session (worst −8.9 %), and it exceeds arm (v)'s artifact.
- **Limits.** Full-RAM mode (the deployment's own host tier); the arm-(a) allocator is fitted on the
  pooled eight sessions, so it is optimistic for (a) and does not change the verdict; the seed is a
  leave-one-out same-category profile, an upper bound on a real cross-run saved state; the traces
  are one checkpoint and eight sessions; and a replay gain is not a wall-clock gain (the live A/B is
  the protocol's job). The absolute deployed miss rate is sensitive to the recorded budget — the
  cache fills only as fast as the trace's own promotion budget allows — which is itself what makes
  the warm start valuable.

## Capacity point (b) — the per-round profile

The protocol scores the offline bar at **both** the trace header's `frames` and the time-weighted
mean capacity actually held. The `profile` column does the second: the sim follows the trace's own
per-round frame count instead of one fixed `V` (`per_round_vram`, `per_layer_skew.py`). Pooled
misses/generated token:

| arm | deployed (header frames) | profile (per-round frames) | sweep 8351 |
|---|---:|---:|---:|
| uniform (iv) | 66.142 | 66.806 | 64.171 |
| sequential (v) | 72.032 | 72.688 (+8.80 %) | 69.855 |
| quota_even | 69.152 | 69.437 (+3.94 %) | 67.212 |
| quota (a) | 66.175 | 66.801 (−0.01 %) | 64.251 |
| w_ent 0.25 | 66.006 | 66.685 (−0.18 %) | 64.056 |
| w_ent 1.0 (b) | 70.034 | 70.788 (+5.96 %) | 67.801 |
| w_permuted | 73.650 | 74.460 (+11.46 %) | 71.320 |
| seed (c) | 53.471 | 54.170 (−18.91 %) | 51.173 |
| seed+w_entropy | 57.439 | 58.182 (−12.91 %) | 54.943 |

The verdict is unchanged at point (b): the largest relative move between the two capacity points is
`quota_even`'s 0.61 points (+4.55 % to +3.94 %), and the winner's gain is 18.91 % there against
19.16 % at the header frames.

## Reproduce

Prerequisites: the eight traces at `out/route-traces/*.bin` (git-ignored — record them with
`bench/route_traces/record.py`, see that directory's README) and a Python with numpy (the repo's
`nix develop .` shell, Python 3.13; the module itself needs no other dependency).

```
python3 tools/expert_cache_replay/per_layer_skew.py calibrate
python3 tools/expert_cache_replay/per_layer_skew.py arms --jobs 16 --json out/per-layer-skew.json
```

The `arms` run prints the arm table across the deployed capacity, the **per-round profile** and the
sweep, and writes the capacity vectors, the allocator thresholds and the sensitivity to JSON.
Deserialize with the repo's `read_trace`. A full run is CPU-bound: about five minutes at `--jobs 16`,
around an hour with the default single worker.

Branch `proto/per-layer-residency-skew`. The prototype is deliberately unpolished and unmerged.

One correction the prototype's own history cannot show: the sequential control's first version
omitted the layer offset in its access keys and was therefore capacity-independent — every access
missed, a ~97 % miss rate. The reported arm (v) is the corrected version; the branch is a single
squashed commit, so the wrong intermediate state is not reconstructable from it.
