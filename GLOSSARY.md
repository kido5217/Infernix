# Infernix

Vocabulary for this repository's domain. This file fixes names only. Every contract, invariant
and mechanism lives in its maintainer document ([docs/README.md](docs/README.md) routes them);
where a definition here would overlap a contract, the authority is linked instead of restated.

## Execution and scheduling

Authority: [engine architecture](docs/maintainer/engine-architecture.md),
[resource scheduling](docs/maintainer/resource-scheduling-and-context-cache.md).

**Lane**:
A resident request position on the Engine; distinct from the request itself, which survives a pause and may resume in a different lane.
_Avoid_: batch slot, session

**Decode round**:
One execution step carrying every resident request's next tokens as a single compact batch.
_Avoid_: decode batch (that is the payload, not the step)

**Prefill turn**:
The step in which a round processes prompt tokens; alternates with decode rounds.

**Rotation point**:
A boundary at which the scheduler may switch which requests prefill or replay and which decode; prompt chunk boundaries are rotation points.

**Permit**:
A typed KV-coverage entitlement a request holds to execute a given step kind.
_Avoid_: lease (reserving that word for frame leases)

**Ingress**:
Admission of waiting requests into lanes, under a bounded FIFO with finite bypass.

**Preemption**:
Resource-pressure pause of a lane: its reservations are reclaimed and its committed state stays in the caches.

**Replay**:
Recovery of a preempted request by re-executing its recorded continuation from the deepest state the caches then hold, splitting at every point the ledger recorded.

**Ledger**:
The per-request record of committed state and split points that makes replay bit-consistent.

## Context and prefix cache

Authority: [hybrid prefix cache](docs/maintainer/hybrid-prefix-cache-spec.md),
[paged KV cache](docs/maintainer/paged-kv-cache.md).

**Context cache**:
The whole store of execution KV state (device blocks plus host backing); "KV cache" alone names the value being stored, not this mechanism.
_Avoid_: prompt cache, KV store

**Prefix cache**:
The one content-addressed store of reusable committed context, with its block index, snapshots, taps, eviction and host slab pool.

**Page / page group**:
The allocation units of the paged context store; capacity is accounted in page groups.

**Tap**:
A checkpoint taken at an exact semantic boundary, so a prefix can be resumed or reused without recomputation.

**Chunk (prefill chunk)**:
The token count a single prefill call processes; fixed at startup (`--prefill-chunk`), or resolved per request from the expert cache's live lendable frames (`--prefill auto`, ported by #73).

**Main-KV**:
A lane's live KV reservation, as distinct from cached prefix blocks.

## Expert cache and offload (Qwen4Exp route)

Authority: [Qwen4Exp design](docs/maintainer/qwen3_8-flash-next-design.md) §9.

**Expert cache**:
The VRAM region holding the hottest routed experts, sharing device memory with the KV stores.

**Frame pool / frame**:
The single device allocation for cacheable roles, in units of exactly one expert record; a frame's role (expert, staging, loaned) changes only through the runtime frame table.

**Loan (lend / give back)**:
Temporary reassignment of frames to prompt- or vision-path buffers; giving back returns them to cache policy, whose re-admission is lazy.
_Avoid_: borrow (Basalt's word for the same cycle)

**Streamed expert**:
An expert consumed through the ring during execution instead of from a resident frame.
_Avoid_: non-resident (that states absence from the cache, not the consumption path)

**Ring**:
The byte-budget streaming buffer the prompt path carves from its loan; ring depth scales with the loan.

**Resident expert**:
An expert that currently holds a device frame.

**Host bank**:
The authoritative RAM copy of the experts; always identical to what the cache holds.
_Avoid_: expert mirror

**File tier**:
The SSD backing below the host bank, read when an expert fits neither VRAM nor the RAM plan.

**LFRU**:
The expert cache's replacement policy, least-frequency-recently-used.

**Routing profile**:
A converter-shipped ranking of expert usage that seeds warm-start placement (ticket #67).

## Artifact and conversion

**Artifact (`.infernix`)**:
The v3 product container file; the only C++ product artifact and the sole Engine input.

**Recipe**:
The converter-side choice of sources, formats, packing and per-input activation permissions for one artifact.

**Numeric format**:
One of the closed registry of weight/state representations the engine implements; adding one means a format entry plus kernels.

**Pack**:
One built, shipped artifact instance (e.g. a quantization release for one model).

## Model execution

Authority: [engine architecture](docs/maintainer/engine-architecture.md).

**Model**:
The fixed mathematics, config interpretation and selected weights of one architecture; immutable at runtime.

**Program**:
The mutable execution state of one route: workspaces, context stores, CUDA Graphs; programs share no mutable state.

**Op**:
A closed mathematical or state-transition implementation with its own qualification contract ([op development](docs/maintainer/op-development.md)).

## Serving and output

Authority: [serving](docs/serving.md), [hybrid prefix cache spec](docs/maintainer/hybrid-prefix-cache-spec.md) for logprob semantics.

**Route**:
A public Engine entry path — generation or offline CausalScoring — both reached through the same artifact.

**CausalScoring**:
Offline teacher-forced scoring over a fixed sequence using the generation execution path.

**Prompt readout**:
The logprob report for a prompt's tokens, computed on raw logits before any truncation.

**Target / top logprobs**:
Full-vocabulary log-softmax values for the chosen token (`target_logprobs`) or the K highest (`top_logprobs`).
_Avoid_: prob, logprob slice

**Constrained decoding**:
Grammar-guided sampling (GBNF, regex, choice) enforced on every speculative mode.

**Drafting / verification**:
Speculative decoding's two sides: a drafter proposes tokens, the model verifies every one exactly; drafting changes speed, never output.
_Avoid_: speculation (keep it for the umbrella term "speculative decoding")
