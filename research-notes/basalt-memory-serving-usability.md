# Basalt memory guards & serving usability — port assessment (ticket #57)

Sources read this session (read-only): Basalt clone `/home/kido/trash/temp/opencode/basalt`
(`serve/`, `README.md`, `src/platform/memory.{cpp,hpp}`, `src/program/generate.cpp`) and Infernix
`main` at `/home/kido/network/projects/Infernix`. Basalt is a Python server driving one C++ engine
process; Infernix's `infernix-serve` is a single C++ binary (httplib). Paths below are relative to
each project root.

## Basalt mechanisms

### Memory floors
- **Startup VRAM-reserve refusal.** `--vram-reserve-mib` below `kVramStallMib = 256` is refused at
  startup because driver paging would stall requests (`src/program/generate.cpp:214,1903-1906`).
  When the default reserve leaves too few expert-cache slots, the reserve auto-shrinks with a
  printed rationale, or the start is refused naming `--vram-reserve-mib` as the remedy
  (`generate.cpp:3291-3334`). After everything is loaded, free VRAM under the floor warns "requests
  may stall; close other programs" (`generate.cpp:5825-5835`).
- **RAM margin + startup expert-copy refusal.** Margin is max(2 GiB, 5 % of MemTotal)
  (`src/platform/memory.cpp:51-53`). The default all-in-RAM expert copy is refused before any
  allocation when it exceeds `MemAvailable − margin`, naming `--resident-experts` /
  `--mmap-experts` (`generate.cpp:2199-2211`).
- **File-cache floor when experts are SSD-resident.** `file_cache_floor()` = one prompt chunk's
  file reads (3.5 GiB) + two layers' experts, minimum 4 GiB, added to the margin
  (`src/platform/memory.hpp:41-62`); applied with a `RAM floor ... experts on SSD, RAM copy <=`
  startup line (`generate.cpp:4560-4589`). An explicit `--resident-budget-gib` that would break the
  floor is refused stating the max safe value (`generate.cpp:4574-4579`).
- **Load-time watchdog.** While loading, MemAvailable below 1 GiB `_exit(1)`s with the remedy named
  (`--resident-budget-gib N` sized to what fits, or the resident/mmap flags)
  (`memory.cpp:75-99`, `generate.cpp:1605,5903-5921`).
- **Request-time `RAM low` abort.** `start_ram_guard` polls MemAvailable every 100 ms
  (`memory.cpp:113-132`); the floor is margin + one prompt chunk when the SSD tier is live
  (`memory.hpp:64-70`, armed at `generate.cpp:5922-5928` with a `RAM ... free, guard ...` line).
  Each running request's stop predicate checks it (`generate.cpp:5814-5822`); the request ends with
  the error `RAM low (X MiB free < Y MiB guard): request stopped; lower --resident-budget-gib`
  returned to the client, and the engine stays up for the next request
  (`generate.cpp:8472-8484`).
- **Per-device startup placement lines.** One `[server] kv cache:` line per device naming the exact
  bytes and what holds them (`README.md:125-127`); layer-split prints per-CUDA expert-cache lines
  (`generate.cpp:3648-3678`); a machine-readable `INFO context=... vram_free_mib=...` line feeds the
  server Monitor tab (`generate.cpp:5855-5874`).

### Single `--config` JSON with CLI > file > defaults
`{"server": {...}, "engine": {...}}`; file entries are converted to command-line words so every
existing CLI check applies, then re-parsed with the real CLI appended — command line wins
(`serve/config.py:58-123,126-156`). Per-section validation refuses wrong keys, wrong types,
engine flags in `server` and vice versa, each naming the key (`config.py:69-90,105-122`);
`model`/`tokenizer` paths resolve relative to the file (`config.py:19,111-113`). The API key cannot
come from the file (`config.py:80-81`, `README.md:143`). Engine flags are validated against the
engine binary's accepted flag set (`config.py:22-29,159-163`); the startup line lists every setting
with its origin, including `--max-context 262144 (config: 1048576 / 4 slots)` when shared
(`config.py:175-188`).

### `--parallel` slot context sharing
`--parallel N` (1..8) maps to engine `--batch` (`config.py:14-18,144-148`); by default
`--max-context` is divided by N (floor 256) so N slots share one KV footprint, `--slot-full-context`
opts out at N× the cache (`config.py:149-155,280-287`, `README.md:110-118`). A prompt that does not
fit its slot is refused with a 400 naming the slot length and the max `max_tokens` that would fit
(`serve/service.py:493-505`; the ValueError → 400 mapping is at `serve/handler.py:645-647`).
`/slots` returns per-slot rows (`n_ctx`, state, held context, draft counts —
`service.py:302-345`, routed at `serve/handler.py:288-291`); `/metrics` embeds
`live.slots {total, active, queue_depth}` plus the waiting queue (`service.py:291-301,365-376`);
`/props` and `/v1/models` report the length each slot has (`handler.py:278,418`).

### Live-stats web UI
Dependency-free static page at `/` served by the server itself (`handler.py:226-252` reads
`serve/web/index.html`, `app.js`, `app.css`): polls `/metrics` and `/slots`, renders live tok/s per
stream, slot lanes and queue, expert tiers per device, VRAM/RAM, a hardware tab, and the tail of
the server log; API key entry kept in `sessionStorage` (`serve/web/app.js:14-28,287-354`).

### `--api-key` exposure rules
The key is command-line only, never config-file (`config.py:80-81`). Comparing is constant-time and
survives UTF-8/Latin-1 header mangling; a whitespace-only key is refused rather than silently
meaning "no key" (`serve/security.py:16-35`). Binding beyond loopback without a key prints
`warning: no API key, open to the network (--api-key KEY)` (`security.py:11-13`,
`serve/server.py:225-226`), and LAN URLs are printed. DNS-rebinding protection: Host must be a
known name/IP (`security.py:127-134`), browser Origins checked (`security.py:137-152`), repeatable
`--allowed-host` / `--cors-origin` (`config.py:315-318`).

## Infernix today
- **VRAM floors:** `--vram-headroom-mib` exists and automatic KV sizing refuses when the headroom +
  minimum runtime reservation does not fit, with the numbers in the error
  (`src/serve/serve_options.cpp:117-122,215-222`, `src/runtime/engine/kv_capacity.cpp:80-107`).
  Qwen4Exp refuses a plan that cannot fit, naming `--vram-headroom-mib` or free VRAM
  (`src/runtime/engine/qwen4_exp_instance.cpp:236-245`), and warns at runtime when free VRAM drops
  below half the headroom with the remedy value
  (`src/models/qwen4_exp/program/program_impl.h:2151-2160`); the expert cache shrinks elastically
  under pressure (`src/models/qwen4_exp/memory_plan.h:140-157`). No minimum-value refusal for an
  explicit tiny headroom — a user-set `--vram-headroom-mib 10` is taken as-is.
- **RAM floors:** host plan sizes the expert RAM copy against MemAvailable (cgroup-aware) minus a
  flat reserve, default 2 GiB (`src/core/host_memory.cpp:54-91`,
  `src/models/qwen4_exp/memory_plan.cpp:21-47`, `include/infernix/types.h:31`). Pinned materializer
  allocations are refused against live available RAM naming `--ram-headroom-mib`
  (`src/artifact/materializer.cpp:365-381`). **No file-cache floor**: with an SSD expert tier the
  plan has no term keeping pages resident for the file reads
  (`memory_plan.cpp` has no such concept), so the OS can evict expert pages mid-pass and the
  reclaim runs at drive speed.
- **No request-time RAM guard.** `available_host_memory_bytes()` is called only on startup paths;
  nothing polls MemAvailable while serving and no request aborts on RAM pressure — the machine
  swaps and every lane's latency collapses together.
- **Startup placement lines: present, arguably richer.** RAM ledger + summary
  (`memory_plan.cpp:50-81`), VRAM ledger + per-role summary (`qwen4_exp_instance.cpp:277-308`),
  capacity line (`src/serve/operational_log.cpp:537-549`). Single-GPU product, so the per-device
  dimension is moot.
- **No config file.** The 842-line hand-written CLI parser is the only surface
  (`src/serve/serve_options.cpp`); no `--config` anywhere in `src/serve/`.
- **Lanes:** `--max-concurrency` 1-8 with every lane holding the full `--max-context`
  (`serve_options.cpp:138,452-454,794-795`; auto KV = `--max-context × --max-concurrency`,
  `serve_options.cpp:214`) — i.e. always Basalt's `--slot-full-context` mode; no sharing option.
  Context overflow already 400s with the numeric limit in the message
  (`src/runtime/engine/engine.cpp:390,499`, `src/models/qwen3_5/frontend/processor.cpp:952,1082`,
  mapped at `src/serve/generation_service.cpp:96-98`).
- **Observability:** `/health`, `/metrics` Prometheus text (`src/serve/http_server.cpp:443-458`,
  `src/serve/metrics.cpp` — includes `requests_running`, `max_concurrency`); console stats panel
  (`src/serve/console_stats.cpp`). No JSON stats, no `/slots`, no static page, no browser UI.
- **API key:** `--api-key` required on everything except `/health` and OPTIONS
  (`src/serve/http_server.cpp:370`), redacted in logs (`serve_options.cpp:397`); startup line shows
  `auth bearer|disabled` (`operational_log.cpp:622-627`). No exposure warning for non-loopback binds
  without a key, no Host/DNS-rebinding check, no key normalization (empty-string check is the
  whole rule).

## Port assessment

| # | Candidate | Surface | Cost | User-visible benefit | Overlap |
|---|---|---|---|---|---|
| C1 | `--vram-reserve-mib` startup refusal | engine startup + `serve_options` | tiny | refuses a self-inflicted tiny headroom before the first stall | mostly overlapped: Infernix refuses only the *automatic* path; missing a minimum-value check on the explicit path |
| C2 | RAM file-cache floor for the SSD expert tier | `plan_host_memory` + startup line + refusal of `--expert-ram-mib` that breaks it | small | predictable Flash-Next decode latency when experts spill: the OS stops re-reading hot expert pages at drive speed mid-pass | real gap |
| C3 | Request-time `RAM low` abort with named remedies | guard thread + request stop-predicate hook + error mapping | small-medium (guard thread, one error kind, docs) | a memory-exhausted machine sheds requests with a clean message instead of swapping every lane into oblivion; engine survives | real gap |
| C4 | Per-device kv-cache placement lines | none | — | — | Infernix already does it better (RAM/VRAM ledgers with per-role splits, commit-limit and display state) |
| C5 | Single `--config` JSON, CLI>file>defaults | new loader reusing the existing parser (convert sections to argv words, re-parse; key/type/section validation; startup origin line) | medium | one reviewable, diffable deploy artifact; bad keys refused by name at startup; no precedence surprises | real gap (no config file at all) |
| C6 | `--parallel` slot context sharing (`--max-context / N`, `--slot-full-context`) | KV capacity model, admission, prefix-cache accounting, docs/serving contract | large | N lanes for 1× KV: long-context multi-lane serving on one 5090 without paying N× full-context cache | real gap, but it changes the fixed full-context lane invariant the scheduler's bounded ingress and compact decode batch assume |
| C6b | `/slots` endpoint + `live.slots` JSON + 400 naming slot length | one JSON route (+ gauges already exist); the 400 half is already done | small | lane occupancy and queue visible without Prometheus | 400 part overlapped; endpoint part is a real gap and a prerequisite for C7 |
| C7 | Live-stats web UI | static page (single file or embedded resource) + JSON stats route | medium | tok/s per lane, cache hit rate, VRAM/RAM in a browser for the local single-owner user; currently only a console panel or a Prometheus stack | real gap (Infernix has neither; console panel is the honest partial credit) |
| C8 | `--api-key` exposure rules | warn on non-loopback bind without a key; trim/refuse whitespace-only key | tiny | one stderr line prevents the open-to-the-LAN surprise; matches the existing `auth disabled` log | real gap; Host/DNS-rebinding allow-list is extra surface |

## Recommended verdicts
- **C1 VRAM reserve refusal — port (tiny):** reject explicit `--vram-headroom-mib` below the stall
  floor Basalt measured, or at least warn; everything else is already overlapped and Infernix's
  auto path is better (elastic shrink + named remedy).
- **C2 File-cache floor — port:** add the SSD-tier term to `plan_host_memory` (margin ∪
  chunk-reads floor), print it in the RAM ledger, and refuse an `--expert-ram-mib` that would leave
  less; direct latency protection for the flagship offload route.
- **C3 `RAM low` request abort — port:** poll MemAvailable on the existing status interval, wire
  `ram_guard_low` into the request stop path, return a named error (503-class or a new request
  error) with the remedy (`--ram-headroom-mib`, `--expert-ram-mib`), keep serving; closes the only
  unbounded failure mode (machine-wide swap collapse) Infernix still has.
- **C4 Per-device placement lines — reject:** overlap; Infernix's startup ledgers are strictly
  richer and the product is single-GPU.
- **C5 `--config` JSON — port (thin):** server+model sections only (Infernix has one engine, so no
  Basalt-style `engine`/`server` split is required — one flat section suffices), file entries fed
  through the existing parser so validation and precedence (CLI > file > defaults) come free,
  `--api-key` refused from the file; medium cost, largest everyday-usability win of the list.
- **C6 Slot context sharing — needs-product-change:** full-context fixed lanes are a scheduling
  invariant (bounded FIFO ingress, one compact decode batch, per-lane graph set); `--max-context/N`
  changes admission, preemption/replay, and the published KV capacity semantics — a deliberate
  architecture decision, not a port. The cheap piece — a JSON `/slots`-style occupancy route — can
  accompany C7.
- **C7 Live-stats web UI — port as an optional surface (product call):** a single static page +
  JSON stats endpoint fits the serving boundary and the local-owner persona; keep it read-only and
  default-on loopback like `/health`. If product wants the surface thin, the console panel +
  Prometheus remain the honest status quo.
- **C8 API-key exposure warning — port (tiny):** warn when bound beyond loopback with no key (and
  print the reachable URLs at startup); reject a whitespace-only key. Skip the DNS-rebinding Host
  allow-list — it defends a multi-tenant browser threat model this local, single-owner product
  disclaims.
