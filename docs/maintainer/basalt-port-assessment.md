# Basalt Port Assessment

Decision record of the wayfinder map [Port from Basalt: performance and usability assessment](https://github.com/kido5217/Infernix/issues/54)
(closed October 2026). **Basalt** (`jesdga95/basalt`, MIT, a stripped-down fork of Strata) is a local
engine for Qwen3.8-Flash-Next on sm_120 — the same model and hardware class as Infernix. This document
records which of its mechanisms Infernix will adopt, which it rejects, and which are parked behind
product-contract changes. Implementations are not done here; each **port** verdict has a scoped ticket
(child of the map).

Method: five research tickets ([#55](https://github.com/kido5217/Infernix/issues/55)–[#59](https://github.com/kido5217/Infernix/issues/59)) read both trees in full and recorded findings on
throwaway `research/basalt-*` branches (pushed, never merged). Basalt's published numbers are claims
from its own environment; no new benchmarks informed the verdicts.

## Ported (one ticket each)

| # | Candidate | Source ticket | Research note (branch `research/…`) |
|---|---|---|---|
| [#61](https://github.com/kido5217/Infernix/issues/61) | Expose the draft-accept threshold as `--spec-min-p` with an artifact-carried default | #56 | `basalt-spec-drafters` |
| [#62](https://github.com/kido5217/Infernix/issues/62) | Learned per-round draft policy (n-gram vs MTP from acceptance buckets + measured width cost) | #56 | `basalt-spec-drafters` |
| [#63](https://github.com/kido5217/Infernix/issues/63) | Request-time RAM-low abort with named remedies (closes the swap-collapse gap) | #57 | `basalt-memory-serving-usability` |
| [#64](https://github.com/kido5217/Infernix/issues/64) | SSD file-cache floor term in the RAM expert plan | #57 | `basalt-memory-serving-usability` |
| [#65](https://github.com/kido5217/Infernix/issues/65) | Single `--config` JSON, CLI > file > defaults, validated at startup | #57 | `basalt-memory-serving-usability` |
| [#66](https://github.com/kido5217/Infernix/issues/66) | API-key exposure warning on non-loopback binds | #57 | `basalt-memory-serving-usability` |
| [#67](https://github.com/kido5217/Infernix/issues/67) | Converter-shipped expert profile seeding the warm-start pipeline | #58 | `basalt-expert-tier-model-file` |
| [#68](https://github.com/kido5217/Infernix/issues/68) | 2 MiB alignment of streamed expert objects (O_DIRECT/mmap SSD tier) | #58 | `basalt-expert-tier-model-file` |
| [#69](https://github.com/kido5217/Infernix/issues/69) | Decode-path full-vocabulary KLD quality gate with per-row diff output | #59 | `basalt-formats-quality-tooling` |
| [#70](https://github.com/kido5217/Infernix/issues/70) | Exact greedy-identity build gate for numerically-neutral refactors | #59 | `basalt-formats-quality-tooling` |
| [#71](https://github.com/kido5217/Infernix/issues/71) | Hours-long mixed-protocol serve soak tool (offload/preemption priority) | #59 | `basalt-formats-quality-tooling` |
| [#72](https://github.com/kido5217/Infernix/issues/72) | Whole-model precision map maintained at change time (`precision.md` discipline) | #59 | `basalt-formats-quality-tooling` |
| [#73](https://github.com/kido5217/Infernix/issues/73) | Expert-cache-lent prefill chunk sizing (`--prefill auto`), determinism contract included | #55 | `basalt-prefill-lending` |

## Parked — needs a product change or an explicit future decision

- **Peer-GPU expert tier** (#58): the peer card holds expert slots only and rows stay bit-identical via
  shared kernels, but a second GPU breaks the one-GPU product contract.
- **Self-distilled per-pack MTP drafter** (#56): engine-side cost is converter provenance only —
  Infernix already supports drafter-per-source — but the gain needs a tuned MTP layer trained per pack.
  Basalt's acceptance gain is qualitative in its docs.
- **Shared slot contexts (`max-context / N`)** (#57): full-context fixed lanes are a scheduling
  invariant here (bounded FIFO ingress, one compact decode batch, per-lane graphs); sharing rewrites
  admission, preemption, and KV semantics.
- **Per-section SHA-256 in the container** (#58): deferred; no current integrity gap.
- **Importance-driven mixed-format allocation** (#59): converter-only recipe policy over already
  registered formats; folded into the #33 follow-up rather than carrying its own ticket.

## Rejected

- **Suffix drafter** (#56): duplicate of Infernix's n-gram drafting (only the trigram min-3 admission
  differs, which is a parameter, not a mechanism).
- **`ple_reader`** (#56): not a drafter — it streams the model's own n-gram PLE table from SSD, which
  `--ngram-volume` already covers.
- **Per-device placement lines and the VRAM startup-refusal extra** (#57): Infernix's RAM/VRAM ledgers
  and `--vram-headroom-mib` pressure warnings already cover them.
- **`experts.layout`** (#58): the JSON object directory generalizes it. (It does expose that the
  streamed tier wants a uniform `record_stride` if heterogeneous packs ever ship.)
- **`hc8`** (#58): Infernix recipes B/C already store hc mixers as int8 + FP16-G32, the same scheme;
  Basalt keeps its copy only for the byte-ingest rule, and its precision evidence confirms ours.
- **64 KiB binary section header and the blanket no-requantization rule** (#58): framing taste; the v3
  container and converter recipe ownership cover the needs.
- **Serving observability: live-stats web UI and JSON `/slots`** (#57, maintainer call): skipped; the
  console panel and Prometheus text endpoint remain the observability surface.
- **GGUF format internals** (#59): i-quant codebooks, sub-scale packing, affine K-quant minima are
  format changes with no site in the closed registry and the hardware-fixed per-16 `ue4m3` block-scale
  MMA; imatrix-style rounding is already the question of #33.
- **DNS-rebinding Host allow-list** (#57): derived from a multi-tenant trust model, not this one.
