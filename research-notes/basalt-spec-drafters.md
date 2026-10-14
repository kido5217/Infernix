# Basalt speculative drafter techniques — port assessment for Infernix (ticket #56)

Basalt clone at `/home/kido/trash/temp/opencode/basalt` (upstream
`https://github.com/jesdga95/basalt`); Infernix at `main` (`d3414d6f`). All Basalt
references below are to that clone; all acceptance/speed numbers are as Basalt
publishes them (publish-as-claim), not independently re-measured here.

## Basalt mechanism

### (a) Self-distilled MTP drafter, shipped per quant pack

- The IQ3_S, IQ3_XXS and Q2_0 packs carry an MTP layer fine-tuned on the model's own
  output (self-distillation), "which raises draft acceptance. Output is unchanged,
  because verify decides every token. Q4_K_XL and Q8_0 keep the original MTP layer" —
  `README.md:253-257`. The claim is qualitative; the README publishes no
  distilled-vs-original side-by-side.
- Packaging: the tuned layer is the `mtp/dense` section's raw bytes, "checked against
  the source layout", embedded with `--mtp-dense FILE` (needs `--mtp-source DIR`) and
  recorded as `meta.mtp = {"dense": {"tuning": "self_distilled", "sha256": ...}}` —
  `tools/convert/convert.py:12, 269, 298`, `tools/convert/test_convert.py:377`,
  section list `include/basalt/core/model_format.hpp:26-27`. The engine and server
  read `meta.mtp` only as provenance; the drafter's runtime size/layout is unchanged.
- The original drafter tensors are fetched from the BF16 checkpoint by HTTP range with
  per-tensor SHA-256 pins (31 `mtp.*` tensors) — `tools/convert/mtp_fetch.py:1-31`.
- Drafter runtime: one MTP layer; dense attention over its own cells (diverges from
  the model's sparse selection only below 2,051 cells), Q8_0 dense projections, 512
  routed experts all in VRAM (~708 MB), draft head = rows gathered from the main head
  with an optional `mtp/draft_vocab` subset, scored 2-bit + exact rescore —
  `include/basalt/core/mtp.hpp:1-16`, `docs/precision.md:44-49`.
- Documented end-to-end effect of spec on those (self-distilled) packs: 69.9 % /
  70.1 % draft acceptance, decode 183.9 → 319.8 / 145.1 → 239.5 tok/s (`--spec 0` →
  `--spec 7 --spec-min-p 0.5`, temp 0, 275 prompts, token- and bit-identical output) —
  `docs/precision.md:105-126`.

### (b) `spec` / `spec-min-p` / `mtp-max-t` drafting policy knobs

- `--spec N`: verify windows of N tokens (last accepted + N-1 drafts); 0/1 = plain
  decode; default comes from the model file — `src/program/generate.cpp:471-474`,
  help at `:736-738`, `serve/config.py:273-274` ("default: the model file's").
- `--spec-min-p P`: prefix gate — "a draft enters the verify window only while every
  draft before it (and itself) has at least this probability under the draft layer;
  0 = always --spec-1 drafts" — `src/program/generate.cpp:499-501`, help `:739-740`.
- `--mtp-max-t M`: cap the MTP's own windows so longer verify windows come from suffix
  matches (`--spec 6 --mtp-max-t 4`) — help `:741`, `:553`.
- The per-round choice between MTP and the suffix drafter is `DraftPolicy`
  (`include/basalt/spec/draft_policy.hpp:1-16`, `src/spec/draft_policy.cpp`): it takes
  expected committed tokens per millisecond for each side — MTP from an EMA of tokens
  committed per window size, lookup from `1+q+…+q^k` with `q` learned in 4
  match-length buckets (decayed counts; priors `{0.75, 0.88, 0.93, 0.96}`,
  `draft_policy.cpp:14-16`) — both at *measured* round costs per window size (EMA,
  unmeasured sizes scaled from a prior shape), and picks lookup only on a 3 % margin,
  with a 3-probe escape so a guessed cost cannot permanently mask a size
  (`kProbes`, `draft_policy.cpp:17, 79-86`). Documented motivation: "Taken whenever it
  proposes more than the MTP, [the lookup drafter] lost 2-8 % on ordinary text" —
  `draft_policy.hpp:4-7`. Wired in at `generate.cpp:5978-5979, 6146-6159, 6470`; a
  `--draft-log` file feeds offline policy study (`:766, 8067-8071`).
- `src/spec/controller.cpp` is an alternative full-cost-model controller (dense/expert
  miss/sync/draft priors, min-gain 5 %); it is referenced only by its own test —
  experimental, not on the serving path.
- The suffix drafter itself (`include/basalt/spec/suffix_drafter.hpp:1-12`,
  `src/spec/suffix_drafter.cpp`): weightless, CPU-only; every trigram of prompt +
  accepted output maps to its 4 most recent end positions in a fixed open-addressed
  table (~20 B/token, bounded); `propose` extends candidate matches backwards to
  `max_match`, requires `min_match` (3; on by default, `generate.cpp:548-552,
  5978`), and copies the historical continuation.
- Coupled draft sampling (`include/basalt/core/coupled_draft.hpp:1-20`, off by
  default, `--coupled-draft`): under sampling the MTP samples its draft with the
  target's own sampler chain and the same uniform as the verifying row; a draft is
  kept only on exact match, so output is a function of the seed alone and drafts only
  decide yield — raising sampled-mode agreement over argmax drafts.
- Spec-exactness is proven bitwise: every full-vocab logit row under `--spec 7` equals
  the `--spec 0` run (62,271 / 60,878 rows, none diverged) — `docs/precision.md:105-126`.

### (c) `src/spec/suffix_drafter` and `src/ngram/ple_reader`

Note the ticket's framing needs one correction found in the source: `ple_reader` is
**not** a drafter. It serves the Qwen4Exp model's own n-gram PLE
(`per_layer_token_embd.weight`, 160 × 320,001,536 IQ4_NL = 26.8 GiB, or 160 B FP8 rows
= 51.2 GB) — a per-token embedding table computed inside every layer. Its mechanism
(`include/basalt/ngram/ple_reader.hpp:1-31`, `include/basalt/kernels/ngram.hpp:1-46`):
a host hash of the last 3 token ids picks 16 rows across 16 pages; rows are never
held in RAM — unbuffered 4 KiB reads with an `issue`/`collect` split so the fetches
hide under the embedding + layer-0 compute; prefill issues a whole chunk's rows
deduplicated and sorted by offset, `max_inflight` deep; a bounded, clock-evicted
90 B/row cache; optional SSD keep-alive reads against power-state stalls; A/B arms for
mmap/RAM/sync-submit (`generate.cpp:337-341, 656-662, 779`).

## Infernix today

- Speculative stack: MTP (1-5 drafts), DFlash / DFlash2 (1-15), DFlash2 tree
  verification with an `auto` width controller, and n-gram copy drafting beside any
  backend — `README.md:48,71`, `docs/cli.md:216-251, 276-281`,
  `docs/maintainer/tree-verification.md`, `docs/ngram.md`.
- N-gram drafting (`docs/ngram.md:86-108`) is already a suffix-lookup drafter of the
  same family as Basalt's: bounded per-request CPU index over 16-, 8- and 4-token
  suffixes, exact hit comparison, backward extension to 64 committed tokens,
  `--ngram-min-match` default 12 (range 4..64), `--ngram-draft-tokens 1..63`,
  special-token-terminated sources, tool-line de-numbering, optional session archive
  (`--ngram-archive-mib`, `X-Infernix-Draft-Session`). Round selection is rule-based,
  not learned: a round with any copy proposal verifies at the ngram window, an
  all-neural round at the neural window, at the narrowest captured family (7/15/31)
  that holds the longest copy (`docs/ngram.md:49-71`). Flash-Next already documents
  Basalt's tradeoff: "Each verified column routes its own experts, so wide rounds pay
  off only at high acceptance" (`docs/ngram.md:35-36`).
- qwen4_exp (Flash-Next) MTP path (`src/models/qwen4_exp/program/program_impl.h`):
  online draft-length policy maximizing expected tokens per cost with per-position
  conditional-acceptance EWMA (prior 0.75), linear width cost 0.38 and forced probes —
  Basalt's DraftPolicy MTP half (`:3558-3601`, `:4376-4394`); a **spec-min-p equivalent
  already exists**, hardcoded at probability 0.5 for one-row rounds with measured gain
  "138.8 → 147.2 tok/s, acceptance 75.6 → 81.8 %" attributed to "Strata's
  --spec-min-p rule" (`:4389-4393`, `:1744-1753`, `:3713-3717`); a two-row pair policy
  with its own cost constants (`:4386-4390`, `:3583-3601`); and a gathered proposal
  head (`config.proposal_rows`, `:413`) = Basalt's draft_vocab idea.
- The 52 GB n-gram volume: Infernix already reads the same model PLE table from NVMe
  via `--ngram-volume` (`README.md:19, 197-203, 276-277`) — the ple_reader comparison
  is a solved problem, not a gap.
- Per-pack drafter: the converter already selects/quantizes the drafter per recipe
  (`--components …,mtp`, an independent `--source mtp_quantized=…`, "only the MTP
  drafter re-quantized (Q8 projections, Q4 experts)" in the Flash-Next recipe A) —
  `docs/weight-conversion.md:41, 67, 108-115, 377-390`. No Infernix pack ships a
  distilled drafter; the `.infernix` meta has no distillation-provenance field.
- Published Infernix drafter reality: MTP acceptance 2.37-2.38 tokens/round in the
  agentic replay (`README.md:127`); DFlash2 3.15 tokens/round at 230K on one prompt
  (`README.md:151-152`).

## Port assessment

**1. Self-distilled per-pack MTP (`mtp/dense`, `--mtp-dense`, `meta.mtp`).**
Mechanism: fine-tune the model's own MTP layer on the *quantized* pack's outputs; the
engine loads byte-identical layout, so the integration is converter + artifact only.
Infernix surface: a `--source mtp=<tuned checkpoint>` path already exists
(`--components mtp`, standalone-MTP mapping, `docs/weight-conversion.md:377-390`);
what is missing is (i) distilled checkpoints for the NVFP4/GGUF-adjacent Infernix
packs — real training work outside the engine — and (ii) a `meta.mtp`-style provenance
entry (tuning kind + SHA-256) in the `.infernix` meta section. Cost: small engine
code, external artifact work per pack (one fine-tune per quantization, the same cost
Basalt pays pack-by-pack while Q4_K_XL/Q8_0 keep the stock layer). Benefit: grounded
only in Basalt's qualitative claim ("raises draft acceptance") plus its spec-7 numbers
on distilled packs (69.9-70.1 % acceptance; `docs/precision.md:119-126`); no published
distilled-vs-stock delta exists in the clone, so the size of the win on Infernix's
2.37-tokens/round MTP baseline is unquantified. Risks: none to output (verify-gated,
both engines prove bit-identity); pack-pinning and per-pack conversion matrix churn.

**2. `spec` / `spec-min-p` / `mtp-max-t`.**
Window-size flags: Infernix has per-source windows (`--draft-tokens`,
`--ngram-draft-tokens`), so `--spec` and `--mtp-max-t` are already covered by surface.
`--spec-min-p`: the mechanism exists in qwen4_exp but hardcoded (0.5, one-row rounds
only); the port is to expose it as a flag with a per-artifact default — cheap, and
Infernix itself measured the gain. What genuinely does not exist in Infernix is
Basalt's **learned lookup-vs-MTP expected-value comparison**: Infernix gates copy
proposals on match length and captured families (static rules), while Basalt learns
per-match-bucket acceptance and measured per-width costs online and requires a 3 %
expected-tokens/ms margin, with the documented 2-8 % loss as the failure mode the
margin prevents. Infernix's kWidthCost machinery (`program_impl.h:3558-3601`) shows
where a suffix-side q-bucket and cost comparison would slot in: host-side, per lane,
affects only round/provider choice, never output (verify decides). Cost: small
(self-contained policy state, no container or converter change). Benefit: plausible
on copy-heavy agent workloads where the fixed min-match-12 admission is a proxy for
confidence; unmeasured here.

**3. Suffix drafter vs Infernix n-gram drafting.** Same mechanism class; Infernix's is
more engineered (session archives, derived tool sources, admission lookahead).
Basalt's differentiators are the trigram minimum (match ≥3) feeding the learned
bucketed acceptance — Infernix's longer-suffix + min-match-12 design deliberately
skips short matches. Porting the *index* is a reject (duplicate of existing machinery);
porting the *learned q-buckets on top of the existing index* is the item under (2).

**4. `ple_reader` vs `--ngram-volume`.** Not a drafter; Infernix already streams the
same 52 GB table from NVMe for Flash-Next (`README.md:19, 276-277`). No port needed for
this ticket; Basalt's row cache / keep-alive / issue-collect split are comparable
engineering to cross-check during any future volume-format work, and both engines'
documented concerns (SSD power states, page dedup for prefill) match.

## Recommended verdict

- **Port (small):** `spec-min-p` as a user knob with an artifact default — Infernix
  already ships the rule with a measured win; finishing it is surface work.
- **Port candidate (small, needs a benchmark to justify):** the DraftPolicy-style
  learned suffix-vs-MTP choice (per-match-bucket acceptance EWMA + measured width
  costs + margin) on the qwen4_exp n-gram path. Cheap, verify-safe, addresses a
  documented failure mode (2-8 % on ordinary text).
- **Needs-product-change:** the self-distilled per-pack drafter. The engine/converter
  surface largely exists; adopting it means committing to train and ship a tuned MTP
  layer per Infernix quant pack, with `meta.mtp`-style provenance added to the v3
  container. Basalt's published evidence is qualitative for the distillation itself;
  budget the gain against its 69.9-70.1 % distilled-pack acceptance as the only
  stated anchor.
- **Reject:** porting Basalt's suffix drafter index (Infernix n-gram drafting covers
  it, with different admission heuristics) and anything PLE-related (equivalent
  mechanism already shipped as `--ngram-volume`).
