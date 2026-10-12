# MoE-CORE's layer-wise expert-cache allocator vs the global LFRU prior

Research note, ticket T1 (issue #8). Scope: what MoE-CORE actually specifies, ablated, and
measured, reconciled against this repo's expert-cache design (§9.3–§9.4, §8.7, S4b of
[`qwen3_8-flash-next-design.md`](../maintainer/qwen3_8-flash-next-design.md)).

Primary sources read in full (HTML render of the arXiv v-pages cited below):

- **MoE-CORE** — *MoE-CORE: Coordinated Expert Offloading and Residency for Memory-Constrained MoE
  Inference*, arXiv:2610.01950v1 (cs.DC, 1 Oct 2026).
- **Trace** — Yu Zhang, *Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload
  Contamination, and Operating Regimes*, arXiv:2608.07911v4 (cs.LG/cs.PF, 8 Aug 2026, rev. 25 Aug 2026).
- **Negative** — Shriniwas Ramesh Suram, *Cacheable by Design? Training Mixture-of-Experts Routers
  for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a
  Systems Measurement Study*, arXiv:2608.18261v1 (cs.AI/cs.LG, 18 Aug 2026).

A correction to the ticket's framing up front: MoE-CORE's allocator is **not** seeded by the
workload-domain hint. The domain hint selects *which* experts fill each layer's slots
(initialization, §3.2); the allocator selects *how many* slots each layer gets (§3.1) and reads no
domain signal. The two are separate mechanisms in the paper's own words (§3.2, quoted below).

---

## 1. The allocator

**Source:** MoE-CORE §3.1 "Dynamic Layer-Wise Expert-Cache Allocation"; motivation §2.4;
evaluated configuration §4.2.

**Objective.** A fixed decode-cache memory `M_dec` is divided across the `L` MoE layers as per-layer
slot counts `K_l`, subject to `Σ_l K_l · S_l ≤ M_dec` (`S_l` = one expert's size at layer `l`). The
goal is not a formula but a greedy hill-climb: shift slots "from locally insensitive high-hit layers
to cache-sensitive low-hit layers" (§3.1). The motivation is empirical (§2.4): under a uniform
84-experts-per-layer quota, DeepSeek-V4-Flash-W4A8's per-layer hit-rate range spans 7.64–12.07
percentage points across five workloads, with layers 4 and 7 in the bottom ten of all five and
layers 7, 9, 19, 25, 27, 31, 37 strict local troughs in every trace.

**Inputs.** The allocator's only measured input is the **layer-wise hit-rate response curve**
`Ĥ_l(k)`, obtained by replaying "representative routing traces" at "a small set of candidate cache
sizes around the uniform allocation", with the replacement and initialization policies **held
constant** so the measured difference reflects only slot count (§3.1). It does **not** consume
routing entropy, an observed miss rate, or the domain hint. Alongside the curve it uses the
layer-wise median hit rate and two thresholds `ε_drop` (permitted removal loss) and `τ_low`
(receiver shortfall below median), plus a slot increment `ΔK`.

**Formula.** At each round, for the current allocation,

```
D_l⁻ = Ĥ_l(K_l) − Ĥ_l(K_l − ΔK)        (eq. 6a, marginal loss of removing ΔK)
G_l⁺ = Ĥ_l(K_l + ΔK) − Ĥ_l(K_l)        (eq. 6b, marginal gain of adding ΔK)
```

Donor set = layers whose hit rate is above the layer-wise median **and** whose removal loss
`D_l⁻ ≤ ε_drop`. Receiver set = layers whose hit rate falls at least `τ_low` below the median. Pick
the donor `d` with the smallest `D_d⁻` and the receiver `r` with the largest `G_r⁺`; if
`G_r⁺ > D_d⁻` and the memory constraint still holds, transfer:

```
K_d ← K_d − ΔK,   K_r ← K_r + ΔK        (eq. 7)
```

Donor and receiver sets are recomputed after every transfer. Reallocation stops when no receiver
remains below the low-hit threshold, no donor can release slots within `ε_drop`, or no feasible
transfer offers positive estimated gain (§3.1).

**Granularity.** **Per layer**, not per expert. `K_l` is a slot count for layer `l`; the experts
that occupy those slots are chosen by the domain profile (§3.2), and after that by RHAC (§3.3).

**Static per workload, not per-session.** "Profiling and allocation run outside the per-token
inference path and are repeated when the model, deployment workload, or available decode-cache
capacity changes" (§3.1). The word *dynamic* in the section title contrasts with a fixed uniform
quota; it does not mean per-session or online adaptation.

**Evaluated configuration** (§4.2): uniform default 84 slots/layer; the evaluated nonuniform
allocation uses 78 slots for most layers and 120 slots for layers 7, 9, 19, 31, 37. (Note: §2.4's
trough list also names layers 25 and 27; they do not receive the 120-slot bump.)

---

## 2. Initialization vs capacity

**Source:** MoE-CORE §3.2 "Offline Domain Hot-Expert Profiles"; boundary stated in §3.2.

They are **separate mechanisms, not one seeded by the other** — the paper states the separation
explicitly: "This construction separates cache capacity from cache contents: the layer-wise
allocator determines how many experts layer `l` can retain, while the selected domain profile
determines which experts initially occupy those slots" (§3.2).

- **Capacity (§3.1)** is decided offline, per deployment workload, from measured hit-rate-response
  curves. It is domain-independent and ends before any request is served. It changes neither the
  router decisions nor the model computation (§3.1).
- **Initialization (§3.2)** happens per request, keyed by a domain identifier `d` supplied by the
  serving scheduler at admission. For each domain the profiler records each expert's **mean router
  selection score** over the domain's dataset,

  ```
  μ_{d,l,e} = (1/|T_d|) Σ_{t∈T_d} r_{l,t,e}        (eq. 8)
  ```

  and stores, per (domain, layer), the descending permutation `π_{d,l}` (eq. 9). When the allocator
  has assigned `K_l` slots, the resident set is initialized to the top-`K_l` experts:
  `R_init_l(d) = {π_{d,l}(1), …, π_{d,l}(K_l)}` (eq. 10).
- **Where one ends and the other begins.** Capacity is decided before decoding, from hit-rate
  curves; initialization is decided at admission, from mean selection scores; RHAC then adapts
  contents per token and is what makes the initialization transient. They are computed from
  different offline data and are not derived from each other; the only coupling is that `K_l`
  bounds the size of the initial top-`K_l` set. Profiles are built for "a fixed set of commonly
  served domains using mathematical-reasoning, code-generation, and QA datasets" (§3.2); the
  `π_{d,l}` permutation is precomputed, so the domain choice is a lookup at admission.

---

## 3. Ablation

**Source:** MoE-CORE §4.5.2 (Figure 8), GSM8K + DeepSeek-V4-Flash-W4A8, one component changed per
experiment, same expert-memory budget, exact execution, AES disabled.

| Configuration | TPOT (ms) | Δ vs full |
|---|---:|---:|
| Full system | 36.0 | — |
| **w/o nonuniform allocation** (uniform per-layer capacity) | **36.4** | **+1.1%** |
| w/o domain hot-expert profile (initialization) | 36.5 | +1.4% |
| w/o RHAC (replacement) | 46.8 | +30.0% |

The nonuniform-capacity row is exactly the requested isolation — replacement, initialization and
prefetch held fixed, only the uniform-vs-nonuniform allocation changed. Its attributable gain is
**1.1% TPOT (0.4 ms)**, and the ablations' layer-wise hit-rate curves "remain close to that of the
full system, but exhibit localized reductions in hit rate" (§4.5.2). RHAC is by far the dominant
decode-cache mechanism (30.0%, hit rate "above 90% in most layers" → "approximately 60–82%").

Supporting ablations in the same section:

- **Prefetch depth (§4.5.3, Figure 9):** 0 → 42.6 ms; depth 1 → 36.0 ms (−15.5%); depth 2 → 35.9 ms;
  depth 3 → 36.5 ms. Mean exact hit rate 89.05% → 92.43% / 93.81% / 94.05%. Default depth is 1.
- **Uniform capacity alone (§4.5.1, Figure 7):** raising the per-layer cache from 12 to 120 experts
  moves the unweighted mean hit rate from 47.47% to 96.43%; 12→24 gains 21.01 pp, 84→120 gains
  only 5.13 pp. This is the diminishing-returns curve the allocator is exploiting.
- **AES (§4.5.4):** mean effective expert availability 92.43% → 99.972%, but at +4.7% TPOT
  overhead at batch 1 (Table 3).

---

## 4. Trace-evaluation contract

**Source:** arXiv:2608.07911v4, §2 (semantics), §4 (Axis I), §5 (Axis II), §6 (Axis III),
§9.3 (service discipline), §10.1–§10.4 (the reporting checklist). Stated here as checkable rules,
with the effect of violating each.

### Group A — Replay and execution model

- **A1 — State the execution unit.** Hit/miss classified once per `(step, layer)` event over the
  batch-wide **union** `E(s,l) = ⋃_{r∈A(s)} topk(r,l,s)`, or per individual expert access (§2.1).
  They are not interchangeable.
- **A2 (error, not a choice) — Defer retention to the event boundary.** Event-atomic replay: snapshot
  the cache at event start; classify all of `E(s,l)` against that snapshot; serve all misses; **only
  then** apply admission and eviction (§2.2, Definition 2). If replay is sequential, state explicitly
  whether a block resident at event start and required later can be evicted mid-event and counted
  again. Violation effect: at the 128-expert model, ρ = 40%, B = 8, sequential replay inflates LRU
  **+27.4%**, LFRU **+28.5%**, Least-Stale **+27.4%**, but LFU only +3.4%, Belady +4.3%, a same-trace
  static diagnostic 0.0% (Table 2, §4.3). The distortion is family-selective and **inverts the
  ranking**: under sequential replay the static diagnostic appears to beat LFRU by 4.04 pp; under
  event-atomic replay LFRU beats it by 5.70% relative (1.09 pp). No single correction factor recovers
  the ordering (§4.3).
- **A3 — If replay is sequential, state the intra-event order.** Ascending expert index makes
  checkpoint numbering part of the retention decision; randomizing removes that dependence but not
  A2's error (§4.1–§4.2). Violation effect: 64-expert model, ρ = 40%, LRU — sequential ascending
  index 99.73% miss ratio; sequential seeded-random 87.15–87.19%; event-atomic 56.92–57.11%
  (Table 1, §4.2).
- **A4 — Provide a conformance check.** The paper ships a hand-verifiable trace (3 layers × 4
  experts, 5 scheduling steps, 30 union accesses → 12 hits / 18 misses at the released capacity
  setting); "a simulator that does not reproduce these counts does not implement the semantics used
  in this paper" (§2.5).

### Group B — Workload construction

Rules B1–B8 (§10.2): report distinct instruction templates / prompt prefixes / final lines per
category (B1); whether the chat template was applied (B2); number of decode steps and whether the
window extends past the shared-prefix region (B3); whether concurrent requests are position-locked
(equal length, lock-step admission/retirement) (B4); positional token agreement and unique sequence
count per category (B5); within-category expert overlap restricted to low-text-overlap pairs
against a cross-category baseline (B6); effect stability across decode-position bands (B7); and a
**matched-pair control** rendering the same source records both ways (B8).

Violation effects (§5):

- A single-template probe set reported process diagnostics as the headline archetype with 57.2%
  reduction in effective miss fraction; the diversity-controlled set gives **5.9%**, while
  office/legal moves 26.9% → **53.3%** — the point-estimate ordering **inverts** (Table 4, §5.1).
- **Matched-pair rendering** (same records, fixed template vs diversified, early window t 0–16)
  shifts the apparent union reduction by **+19.4 pp (document RAG) to +31.9 pp (office/legal)**,
  by archetype (Table 7, §5.5). Mechanism (§5.2): shared template tail + short decode window +
  position-locked cohorts; in one archetype 4 of 16 requests produced byte-identical 63-token
  outputs and positional agreement was 20.65% vs 1.04% for a long-heterogeneous-document category.
- The template effect is **not readable from the full-trace miss difference** — the fixed-template
  arm has a *higher* full-trace miss fraction (21.99% vs 18.78% at ρ = 40%, B = 8) because it also
  shortens responses; only the position-resolved union comparison isolates it (§5.5).
- After correcting, only **2 of 6** archetypes survive both the contamination correction and a
  size-matched reference (§5.7, Table 8).

### Group C — Regime and comparability

- **C1** report the per-step, per-layer expert-union distribution, per-layer capacity, `r(s,l)`, and
  at least its mean `r̄` and the fraction above 1; the mean alone is not sufficient (§6.1, §6.5).
- **C2** state the cache scope (per-layer or global) and, for per-layer, how the remainder is
  allocated (§2.3).
- **C3** state the miss denominator: expert assignments before or after intra-event dedup (§2.4).
- **C4** state whether cross-request dedup is credited to the policy — it belongs to any FCFS
  scheduler and is not a policy contribution (§2.4).
- **C5** state the **service discipline** (continuous batching vs rotation of admitted sessions;
  admission and preemption rules; any fairness bound). Violation effect: at a fixed request set,
  capacity and mean batch size, changing only the service discipline moved a static pinned set from
  **7.5% worse than LFRU to 17.0% better** (Table 20, §9.3).
- **C6** for every static/pinned baseline, state fit source, split boundary, quota rule, and whether
  the initial load is counted in reported traffic (§2.3–§2.4).
- **C7** for cross-model comparison, align on `r̄` first, not batch size. Aligned on batch size three
  models span **36.8 pp** of gap; aligned on `r̄` the same measurements agree to within **1.1–4.6**
  (§6.4). Even `r̄` is necessary but not sufficient: permuting only the **temporal order** of an
  identical event multiset moves the offline-optimal gap from **44.9% to 30.8%** (§6).

### Group D — Interpreting oracle bounds

D1 report absolute transferred bytes/token alongside any gap and the implied bandwidth budget (§6.6);
D2 report the forced-admission decomposition (§7.2); **D3 (error)** do not report per-step best-of-N
batch compositions as achievable scheduling gains — the per-step figure overstated the trajectory
figure by roughly **3×** (§8.4); D4 state the scope of every negative result (model, ρ, concurrency,
service discipline, fairness constraint, lossless-or-not) (§8.2, §8.5).

### The numbers those rules protect

With all three axes controlled, the offline-optimal gap is large and stable: **44.18%–45.93%,
mean 45.09%** over 13 frozen workload compositions (Table 13, §7.1). A forced-admission oracle
attributes **84.3%–96.6%** of it to knowing which resident expert is used furthest in the future
(Table 14, §7.2). A causal next-use predictor used as an eviction rule recovers **−11.4%** of the
gap — it is worse than the causal baseline it was meant to improve — and selects an optimal victim
3.4% of the time against 2.4% for a random resident block and 20.6–22.1% for LRU and LFRU (§7.3).
LFRU is the best causal policy in 12 of 13 conditions (Table 13, §7.1).

---

## 5. The negative result

**Source:** arXiv:2608.18261v1 (Qwen3-235B-A22B Q4_K_M, 134 GB, one 8 GB GPU; routing telemetry on
Qwen3-30B-A3B; pre-registered 137M/340M training study).

**Systems measurement (§3).** Measured decode is 0.441 tok/s warm, agreeing with the
bytes-per-token ÷ bandwidth model to within 6% (the SSD's ~2.4 GB/s is 50× short of the 120 GB/s a
10 tok/s target needs). Request batching, which should amortize one disk sweep, tracks the
union-of-experts model to B = 8 then **collapses at B = 32** (0.087 tok/s) from paging thrash.

**Routing structure (§4), which is the part relevant here.** On Qwen3-30B-A3B over 8k tokens × four
domains: expert reuse between adjacent tokens is `P = 0.444` against a chance rate of 0.223
(**2.0×**); 95% of traffic flows through **52.5%** of experts; a per-layer **LRU** cache at a 13.4%
budget serves **65.9%**, **beating** static pinning (59.2%) and LFU (60.1%), with a clairvoyant
Belady oracle at 79.1% — "the exploitable signal is temporal". Per-domain expert sets are
structured: code overlaps the others by only 0.11–0.16 (Jaccard-style) versus 0.33–0.42 among
prose/math/medical (Tables 3–4).

**Ceiling (§5).** Single-stream 10 tok/s needs a ~95% hit rate → ~52% of experts resident (~70 GB),
**4×** this machine's fast memory; "placement alone cannot reach interactive speed here".

**Pre-registered training study (§6–§7).** Two auxiliary router losses — temporal reuse `L_loc`
and domain confinement `L_dom` — were trained into a 137M MoE. The mechanism works (locality
training cuts misses up to 60%; domain training reaches a 99% static-pin hit rate) but **every
configuration fails the pre-registered ≤1% perplexity gate**: λ = 0.02 preserves quality (+0.3% PPL)
but cuts misses only 15%; λ = 0.05 cuts 59–60% but costs +2.1–3.1% PPL; λ = 0.03 lands in the dead
zone. RQ2's static-pin hit@50% reaches 0.991 at +3.1% PPL. The tax does not shrink at 340M (it rises
slightly). Training-free cache-aware rerouting **stacks** with trained locality (~80% miss reduction
at ≤3.4% PPL), but **domain-primed prefetching does not help**: "an LRU cache re-warms within
O(cap) tokens of a switch, so the boundary miss burst amortises to nothing over realistic
(≥1k-token) domain runs" (§7.1).

**Does it contradict "per-layer routing entropy is exploitable for cache budgeting"?**

- It **does not address per-layer routing entropy at all**. The word "entropy" does not appear in
  the paper, and its cache simulator holds capacity **uniform** across layers
  (`c = ceil(cap · E)` per layer, Appendix E) — it never budgets capacity by layer. So there is no
  contradiction of the entropy premise from this paper; the premise is simply untested there.
- It **supports** the general cacheability of routing: strong temporal reuse (2.0×), concentrated
  popularity (95% of traffic in 52.5% of experts), and a large Belady headroom at a tiny budget.
- It **undercuts two premises a layer-wise budget might lean on**: (a) the exploitable order is
  **temporal**, not static popularity (LRU 65.9% > static 59.2% at the same 13.4% budget), which is
  the repo's own reason for choosing LFRU over a static profile; and (b) **domain-conditioned
  placement alone buys nothing at serving time** — when the domain is known at request time and
  prefetched, LRU re-warms too fast for the boundary cost to matter.
- Its negative result is **scoped** (rule D4): ≤340M parameters, multi-domain, 200M tokens
  (below compute-optimal), one consumer GPU; its own §7.1 shows the mechanism is real and stacks.

---

## What is genuinely borrowable here (and what this repo already has)

| MoE-CORE mechanism | Repo status | Verdict |
|---|---|---|
| **Nonuniform layer-wise capacity** (§3.1) | Absent — one **global** LFRU pool over 24,576 Text experts, no per-layer quotas; §9.3 records that hard partitioning "lost 3-34%" in this repo's replays, and the AdapMoE/FATE row is filed as "One global pool. Hard per-layer quotas lost 1-35% in replay" | Genuinely absent, but its isolated worth in the source paper is **1.1% TPOT**, and this repo's own replays already rejected hard quotas as a loss. Not worth adopting on this evidence. |
| **Domain-informed initialization** (§3.2) | Partly present — **S4b** warm start (`§9.3` item 4: start from the saved LFRU state; failing that a shipped broad-trace profile, `data/expert-profile.bin`, §1.1), "on by default from the user's own saved state; a shipped fallback profile only if the replay gate passes" (§19.3.5) | The repo keys the seed on **one saved run/profile**, not on a **per-request domain/archetype hint**. That per-request key is the one mechanism here the repo does not have. Its evidence is weak: **1.4% TPOT** isolated in MoE-CORE, and arXiv:2608.18261 finds domain-primed placement ≈ 0 at serving time. |
| **Routing-history-aware replacement** (§3.3 RHAC) | Present — LFRU `f / (now − last + 1)` over one global pool with count halving, plus off-path shadow replay that switches parameters when a variant wins by >5% (§9.3 items 1, 5; §9.4) | RHAC's signals (recent-frequency window, selection EMA, **current router score**, age) are richer than LFRU's single frequency/age ratio. The **router-score term** (`λ_q · q̃`) is the one ingredient LFRU has no analogue for; the repo already computes router scores in the prefetch predictor (§8.7) but does **not** use them in victim ranking. Not isolated by an ablation in MoE-CORE. |
| **Cross-layer prefetching** (§3.4) | Present — §8.7 evaluates layer `l+1`'s router on `x_l`, DMA-bounded by the calibrated completion model (~1.5–2.6 experts/layer), auto-disabled below a 0.3 useful-byte ratio | Same mechanism; nothing new. |
| **Approximate expert substitution** (§3.5) | Absent (the CPU-miss path is exact; §9.4) | Not requested; changes the selected expert set, outside the exact-execution contract. |

**One line:** the only thing genuinely absorbable that this repo lacks is a **per-request
domain/archetype key on the warm-start seed** (MoE-CORE §3.2's `R_init_l(d)` looked up at admission),
because the allocator itself is worth 1.1% in its own paper and this repo's replays already lost
3–34% to hard per-layer quotas — and even the domain key carries MoE-CORE's 1.4% isolated gain
against arXiv:2608.18261's finding that domain-primed placement amortizes to nothing.

## Sources

- MoE-CORE, arXiv:2610.01950 — https://arxiv.org/abs/2610.01950 · HTML https://arxiv.org/html/2610.01950v1
  (§2.4, §3.1, §3.2, §3.3, §3.4, §4.1–§4.5)
- Yu Zhang, arXiv:2608.07911 — https://arxiv.org/abs/2608.07911 · HTML https://arxiv.org/html/2608.07911v4
  (§2, §4, §5, §6, §7, §9.3, §10.1–§10.4)
- S. R. Suram, arXiv:2608.18261 — https://arxiv.org/abs/2608.18261 · HTML https://arxiv.org/html/2608.18261v1
  (§3, §4, §5, §6, §7, §7.1, §8)
- Repo: `docs/maintainer/qwen3_8-flash-next-design.md` §1.1, §8.7, §9.3, §9.4, §19.3.5, §19.3.6.
