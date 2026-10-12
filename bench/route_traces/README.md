# Route-trace recording (wayfinder #11)

Frozen coding-agent sessions for the per-layer expert-cache measurement (map #7): four long
sessions (>= 32K-token prompts, repository sources returned as `role: "tool"` results, then
follow-up edit/review/test requests) and four short sessions (<= 2K-token, 4-turn edit/diff
exchanges). The eight sessions are **eight distinct tasks** (protocol rev 5 §3). Each session is
driven through the locally built `infernix-serve` at the deployed Qwen3.8-Flash-Next flags with
`--greedy` and `--route-trace`, producing one NRTRACE trace per session. Every session is decoded
**greedily** (temperature 0 / exact argmax, no sampling), so the offline trace and the live greedy
A/B run the same workload; the settings are recorded in `manifest.json`.

## Layout

| Path | Tracked | Contents |
|---|---|---|
| `make_prompts.py` | yes | builds the frozen prompt set; deterministic |
| `record.py` | yes | starts the serve per session, drives the steps, stops it, hashes the trace |
| `validate.sh` | yes | runs `host_tier.py --validate` over every recorded trace |
| `manifest.py` | yes | folds `results.json` + validate output into `manifest.json` |
| `prompts/*.json` | yes | the frozen prompt files (committed and hashed before any run) |
| `manifest.json` | yes | per-session category, prompt hash, serve command, trace path + hash, floors, decode settings |
| `out/route-traces/*.bin` | no (git-ignored) | the trace binaries; referenced by path + sha256 |

## Reproduce

```sh
nix develop .                                   # CUDA 13.1 + cmake + ninja + numpy
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build -k 0 infernix-serve infernix_qwen4_exp_route_trace_real_test \
      infernix_serve_options_test infernix_qwen4_exp_expert_cache_test infernix_qwen4_exp_host_tier_test
python3 bench/route_traces/make_prompts.py      # regenerate the frozen prompts
python3 bench/route_traces/record.py            # record all eight sessions (~10 min)
bash    bench/route_traces/validate.sh          # host_tier.py --validate per trace
python3 bench/route_traces/manifest.py          # write manifest.json
```

`record.py` starts and stops the serve once per session, so every trace is one session's rounds.
The process must have ~74 GiB of RAM and the whole GPU free; stop any other `infernix-serve`
unit first.

## Decode settings (frozen)

Every session runs with `--greedy` on the serve (which forces temperature 0 / exact argmax for all
requests, `docs/serving.md`) and with `temperature` 0.0 / `top_p` 1.0 in the request body; no
temperature, top-p or top-k truncation is requested. The same settings are recorded per session in
`manifest.json` (`decode_settings`).

## Session template

- **Long** (`long-1..4`): system prompt + read request; the repository's own sources arrive as
  tool results; then an edit request, a self-review and a test request. Four steps / four
  completions. The four sessions share the template; only the file bundle differs (frozen in the
  committed prompt files).
- **Short** (`short-1..4`): four distinct small edit/diff tasks over four different files
  (`expert_residency.h`, `route_trace.cpp`, `memory_plan.cpp`, `serve_options.cpp`), each a real
  snippet plus four terse turns (edit, justify, test, review) with thinking off.
