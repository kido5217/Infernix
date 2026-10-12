# Per-layer expert-cache telemetry: what the engine has today, what is missing

Research note for wayfinder ticket #9 (map #7). Findings read from the tree on 2026-10-12 (branch point `main` @ `03bb5790`).

## 1. Telemetry today: per-request and per-tier aggregates, Debug level only

| What | Where | Shape |
|---|---|---|
| Cache counters | `src/runtime/.../expert_residency.h:54-64` (`ExpertResidency::Stats`) | `{routed, hits, promotions, seeded, landed, demotions, …}` — aggregate, no layer dimension |
| Per-request stats | `include/infernix/types.h:1214-1218` (`ExpertCacheStats`) | `{routed, hits}` only |
| Request report line | `program_impl.h:2427`, emitted `2442-2450`, `DiagnosticLevel::Debug` | `"expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, %llu landed, %llu misses CPU-served; since start %.1f%%; VRAM free %zu MiB"` |
| SSD tier | `program_impl.h:2455-2467`, Debug | `"SSD expert tier: request %llu expert reads … admitted to RAM … demoted from VRAM"` |
| Lane cost | `program_impl.h:2483` (`report_lane_cost`; diagnostic call `2512`), Debug | per-round lane timing |
| Warm start | `program_impl.h:2301`, `2303` | seed/landing of saved state |
| Serve pretty line | `src/serve/operational_log.cpp:300` | `expert cache <pct>%` |
| Console aggregate | `src/serve/console_stats.cpp:114`, `162-163` | cumulative hit-rate column |

**Enablement.** `DiagnosticLevel::Debug` maps to `logger->debug` (`src/runtime/.../engine_diagnostics.cpp:16-18`), so `--log-level debug|trace` surfaces these lines. **There is no `--debug` flag**, and without an attached observer Debug records are dropped (`qwen4_exp_instance.cpp:61`, `diagnostics.h:33`). `--request-log-jsonl` carries **no expert-cache fields at all** (`src/serve/request_log.cpp`).

**The gap this ticket exists to name:** per-layer miss counts are **not** logged (the counters are per-request and per-tier only). A per-round **route trace** — the per-layer routed ids the replay harness wants — *is* implemented and wired: `RouteTrace` writes the `NRTRACE1` format (`src/models/qwen4_exp/program/route_trace.cpp:32-39` header, `append()` `:64-88` writing `i32 routes[layers][top_k*columns]` per round), reached through the process-wide seam `testing::set_route_trace` (`route_trace.h:96`, `route_trace.cpp:105`), read into options at `src/runtime/engine/qwen4_exp_instance.cpp:213`, constructed when `options_.route_trace` is non-empty (`program_impl.h:920`), and covered by `tests/models/qwen4_exp/test_route_trace_real.cpp:154,158`. What is missing is only an **exposure**: no serve/CLI/product option sets the path (`git grep route_trace -- src/serve src/product include` is empty), so it is reachable today only from in-process tests. There is no per-layer histogram, and the shadow-replay / parameter-switching mechanism of design §9.3 item 5 and §9.4 is **design-only, not implemented**. The policy's own route log stays host-side in `route_host_` (consumed `expert_residency.cpp:432`); the trace is a separate, opt-in dump.

## 2. The replay harness

`tools/expert_cache_replay/`: `replay.py` (24,436 B), `host_tier.py` (43,532 B), `conformance.py`, `__init__.py`.

- Policies in `replay.py`: `LRU`, `LFRU` (optional halving), `LRUPerLayer`, `SLRU`, `ARC`, `LRU2`, `S3FIFO`, `SIEVE`, `DecayedLFU`, `WTinyLFU`, `StrataExchange`, `LazyLRFU`, `StaticHindsight`, `Belady`.
- Default capacities `4008, 6347, 8000, 10200, 12000` (`replay.py:556`); `ENGINE_HALVING=8192`, `COUNT_FLOOR=1/1024` (`replay.py:491-492`); `--window` default 1.
- Invocation: `python3 -m tools.expert_cache_replay.replay DIR [--capacities …] [--window W]` (Python 3.11 + numpy).
- Default traces: `DIR/warmup-trace.npz`, `DIR/a-trace.npz`, `DIR/b-trace.npz`, each `ids [steps, 48, ≥10] int32` (`replay.py:6-9, 33-34`), from `github.com/hz1ulqu01gmnZH4/qwen38-freetoken`. **Those `.npz` files are not in the tree and not on disk.**
- `host_tier.py` consumes the engine's own route-trace format (`NRTRACE1`): `python tools/expert_cache_replay/host_tier.py TRACE … [--validate]`.

**Prior evidence is in the tree.** The `LRU, equal per-layer quotas` row of design §9.3 (the one that lost 3–34%) is produced by `LRUPerLayer` (`replay.py:82-105`, class name `'LRU, equal per-layer partition'`). The halving evidence attributed to `local/workdirs/strata-forks-review/research/b4_decay.py` is **not** in the tree (there is no `local/`).

## 3. Recorded traces: none

No `.npz` traces, no `NRTRACE1`/route-trace `.bin` anywhere in the tree — only the *source* of the format (`route_trace.{h,cpp}`) and a synthetic conformance fixture `tests/fixtures/expert_cache/lfru_conformance.txt` (75,409 B). Serve request logs exist outside the tree (`/home/kido/trash/temp/infernix-log.jsonl`, 39 MB) but contain no expert-cache fields. Public trace sets are referenced design-only: `aswinkumar99/qwen3.8-flash-next-expert-traces` (design doc line 1421) and the FreeToken `warmup`/`A`/`B` set (design doc line 1385).

## 4. The seam a per-layer bias would occupy

- **Landing cap (uniform today):** `ExpertResidency::before_round`, `expert_residency.cpp:356` — `std::min<std::uint32_t>(kLandingSlots, controller_->free_frames() / layers_)` with `kLandingSlots=16` (`expert_residency.h:68`); applied uniformly at `358-364`, landing table built `367-377`.
- **Per-layer promotion budget:** `after_round(..., per_layer_budget, …)` (`expert_residency.cpp:430`), per-layer loop `441`, `budget = per_layer_budget` `459`, `controller_->on_route` `479-482`. Caller `ProgramImpl::decode_budget` (`program_impl.h:2067`, `kDecodePromotionsPerLayer=1` at `:2053`), passed at `1596/1622/1640/1825/2114`.
- **Victim choice:** `LfruPolicy::select_victims` (`expert_cache.cpp:60-100`); score `f / (now − last + 1)` (`:36-38`, inlined `:71`); protected groups → `+inf` (`:72-79`); ties evict the lower key (`:89`, `:98`); halving in `LfruPolicy::step` (`:109-112`), period `kLfruHalvingPeriod=8192` (`expert_residency.h:71`), floor `kCountFloor` (`expert_cache.h:54`).
- **Plug-in points.** For a per-layer score/protection bias: the per-layer loop at `expert_residency.cpp:441-483`, feeding `CacheController::on_route` (`expert_cache.cpp:422`) → `LfruPolicy::step` (`:102`). For per-layer landing caps: `before_round:354-364`, replacing the single `per_layer` scalar with a vector.

## 5. Build and run (needed to measure a code change)

No build tree exists in the repo today (no `build/`, no `build-windows/`); the running engine is the Nix binary `…/bin/infernix-serve`. Build and test inside the **FrInfer dev harness**, `nix develop /home/kido/network/projects/frinfer`: CUDA 13.1 (`nvcc`, `cudart`, `nsight_compute`, `nsight_systems`), `cmake`, `ninja`, `just`, `pkg-config`, ffmpeg, and Python 3.13 with numpy/pytest/jinja2. Its `shellHook` prepends `/run/opengl-driver/lib` to `LD_LIBRARY_PATH` so `libcuda.so.1` is found; the first entry builds the nsight packages from source (later entries are cached). Infernix has no flake of its own. C++ tests: `cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release`, `cmake --build build -j`, `ctest --test-dir build --output-on-failure` (or `cmake --preset dev` / `ctest --preset dev`). Python: `python3 tests/test_expert_cache_host_tier.py`. Model artifacts exist under `/home/kido/ai/models/infernix/`; the serve flags come from outside this repo (`/home/kido/network/projects/nixos-configs/users/kido/llm.nix`).

## Consequences for the map

1. The recording ticket's first step is **exposing the existing `RouteTrace` writer** through a serve/CLI option (it is wired and covered by tests but unreachable outside tests) — not building an emit path — plus configuring a local build tree.
2. The live A/B likewise needs the local build; the two arms differ by a temporary toggle.
3. A live result cannot today be explained layer by layer: only request aggregates exist. Either the trace is recorded during the A/B (preferred) or per-layer counters are added with the instrumentation.
