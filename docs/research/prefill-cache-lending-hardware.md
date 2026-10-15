# Prefill cache-lending on hardware (#76)

Qualification of the lent-prefill-chunk route (`--prefill auto[:N]`, #73) on the deployed artifact:
the two §13 rules, the published chunk trade-off, and the defects the campaign found. Raw logs,
CSVs and driver scripts are host-local under `/home/kido/trash/temp/opencode/verdict/r76/`.

## Environment

- Artifact `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix` with the NVIDIA
  n-gram volume; RTX 5090 (sm_120a, 575 W), CUDA 13.1.
- `--kv-dtype nvfp4 --max-ctx 32768 --rope-yarn-factor 4 --spec mtp --draft-tokens 4
  --lm-head-draft --expert-ram-mib 66000`; one lane unless stated.
- Bench `bench/infernix_bench` (#82), `-n 8 -r 1 --warmup 1` for prefill and `-n 512` for decode;
  serve `build/apps/infernix-serve` with `--max-concurrency 4 --preserve-thinking`.
- main at `9b4376af` — the auto route after PR #83 and #84 fixed its two runtime defects.

## Item 2 — the chunk matrix

| config | pp4096 | pp16384 | pp16384 vs fixed-4096 | tg512 | lanes |
|---|---:|---:|---:|---:|---:|
| fixed 1024 (product default) | 1751.9 | 3976.9 | −37.9 % | 141.39 | 1 |
| fixed 2048 | 1762.7 | 5396.2 | −15.8 % | 141.57 | 1 |
| fixed 4096 | 1777.5 | 6405.1 | — | 141.54 | 1 |
| auto (fallback 1024) | 1762.4 | **6749.3** | **+5.4 %** | 138.64 | 1 |
| auto (fallback 4096) | 1764.0 | **6753.1** | **+5.4 %** | 138.17 | 1 |
| fixed 4096 | 1834.6 | 6390.7 | — | 59.61/req | 4 |
| auto (fallback 4096) | 1820.9 | **6942.8** | **+8.6 %** | 59.63/req | 4 |

Speculative acceptance is 0.6622 in every row, so the prefill gain is not bought from decode
quality. The resolved chunk in these runs is 12,416 columns — the largest the lease at the
deployed pool pays for, above every fixed chunk tested.

Against the product default (fixed 1024) the auto route is **+70 %** at pp16384 (6749 vs 3977),
+0.6 % at pp4096 and −1.9 % at tg512; against the best fixed chunk (4096) it is **+5.4 %** at
pp16384, +8.6 % at four lanes, and −2.4 % / −0.0 % at tg512. The design's Strata-sourced band
(+21–59 % at 9–100 K) is a comparison against the old default; at 16 K that comparison measures
+70 %.

**Published trade-off.** §19.2/§19.3 published the *fixed* chunk's cost — 4096 costs ~491 frames
and ~4.6 % tg512 against 1024. On this build the fixed-4096-vs-1024 decode cost does not
reproduce: 141.54 vs 141.39 (−0.1 %), and its pp4096 row (413 → 659 tok/s) is an earlier build and
route (measured today 1751.9 → 1777.5; the chunk's gain has moved to the long prompt). The rows
that matter for this campaign are the paired auto-vs-fixed comparisons above, measured on one
build in one session.

## Item 7 — the equal-chunk/short-tail rule vs the fixed grid

18,000-token prompt, where the fixed grid pays a 1,616-token tail (4 × 4096 + 1616) while the auto
route uses equal chunks with the short-tail rule:

| config | pp18000 |
|---|---:|
| fixed 4096 | 5617.2 |
| auto (fallback 4096, auto:32768) | **6982.0** (+24.3 %) |

The rule's worth is the tail it removes: the same fixed chunk reaches 6405 at pp16384 (a multiple
of 4096) and 5617 at pp18000.

## Item 3 — greedy identity at equal chunking

`--prefill-chunk 8192 --prefill auto:8192` against `--prefill-chunk 8192`, 13K-token prompt, two
repetitions each: the whole response object (content plus reasoning, 64 completion tokens) is
byte-identical between routes, and each route repeats identically. Matching the chunk size loses
nothing numerically.

## Item 4 — prefix cache under the lent schedule

Auto (fallback 1024, auto:32768), the same 6,606-token prompt three times: 2.607 s / 0.309 s /
0.238 s wall with `reused_tokens` 6,599 (99.9 %). Reuse works at a per-request resolved chunk; the
repeat requests return from the hybrid snapshot.

## Item 5 — lending cost while lanes decode

Fresh 13,179-token prompt (a nonce defeats the prefix cache, so its chunks really run) plus three
concurrent 256-token decodes started 1.5 s later:

| config | long prompt | side decode rate | end-to-end per request | aggregate |
|---|---|---|---|---|
| auto (fallback 1024, auto:32768) | 5.91k t/s, TTFT 3.2 s, 3.6 s | 64.0 / 63.4 / 63.3 t/s | 34.3 / 34.0 / 33.7 t/s | 101.1 t/s |
| fixed 1024 (the same fallback chunk) | 4.63k t/s, TTFT 4.0 s, 4.4 s | 64.8 / 64.3 / 64.5 t/s | 28.6 / 28.3 / 28.1 t/s | 84.4 t/s |
| fixed 4096 | 5.85k t/s, TTFT 3.4 s, 3.8 s | 60.4 / 60.0 / 60.1 t/s | 31.2 / 31.8 / 31.5 t/s | 93.7 t/s |

Borrowing frames costs the decoding lanes nothing measurable in their own rate: while the long
prompt prefills, their decode rate is 64.0 / 63.4 / 63.3 tok/s against 64.8 / 64.3 / 64.5 at the
matching fallback chunk (−1.5 %) and 60.4 / 60.0 / 60.1 at fixed 4096 (+5.6 %). Their end-to-end
throughput is higher than both baselines (34.0 vs 28.3 and 31.5 tok/s per 123-token request)
because the long prompt's chunks finish sooner (3.6 s vs 4.4 s and 3.8 s) and free the round
earlier — an occupancy effect, not a decode-rate effect. No decode regression from the loan at
this concurrency.

A first attempt at this item reused the item-3 prompt, which the prefix cache served in 0.1 s, so
nothing prefilled while the lanes decoded and the numbers measured nothing; the fresh-prompt rerun
above is the valid one.

## Item 6 — grammar-forced tokens on a lent lane

A constrained request (`response_format` json_schema, 16K prompt) on the auto route returns valid
JSON (`{"answer":7}`) and the server serves the next request normally. This is the case that
crashed before #84.

## Item 1 — the suite

247 CTest entries at `9b4376af`, fixed route (no test parameterizes the engine's prefill mode; the
auto route's coverage is the campaign above):

- `ctest -j 8`: 244 pass, 3 fail — `infernix_gdn_replay_fold_wide_w48_l30_test`,
  `..._w64_l30_test`, `infernix_offloaded_moe_layer_test`. Each passes when re-run alone.
- `ctest -j 1`: 246 pass, 1 fail — `infernix_gdn_replay_fold_wide_w64_l48_test`, which passes in
  the parallel run and in other runs.
- A standalone repeat probe saw `..._w48_l30_test` fail once and pass on other runs.

The failures are a flaky family, not a deterministic defect of this work: four different members
fail in different runs and every one passes in another, the tested code is the GDN recurrent-state
fold, untouched by #73/#82/#83/#84, and the failing comparison is a numerical state check. Filed
separately. Running the suite at `-j 8` on one GPU additionally provokes these flakes through
contention; the serial run is the meaningful configuration for a single card.

## Item 8 — shipped expert profile cold start (not run)

The deployed artifact predates #67 and carries no `expert_profile` resource, so the loader's
pack-profile branch cannot be exercised without re-converting the artifact. Recorded as a
follow-up; the CPU-side seeding logic is unit-tested and mutation-verified.

## Defects the campaign found (all fixed and merged)

1. **Load-time throw.** `--prefill auto` threw
   `causal_softmax_attention prompt wave: unsupported head geometry` on the deployed artifact: auto
   sizing asked for the fast prompt-wave granule for every fast-kernel KV format, and this model's
   QSA geometry (24 query / 2 KV heads at D256; the prompt-wave kernel implements 24/4 and 16/2)
   has none. #83 added `causal_softmax_attention_prompt_wave_supported` and queries the granule
   only when it exists.
2. **Wide-region staging.** The prefill unit's trailing 1-column call kept the wide io view without
   entering the wide region, so the 31 MB image was copied into the 4 KiB static buffer
   (`cudaErrorInvalidValue`). #83 widened `WideWork`'s gate.
3. **Mirror lifetime.** `return_wide_work()` destroyed the pinned mirror before
   `prefill_progress`'s `sample` staged through `host_io()` (SIGSEGV in `sample` and in the walk's
   progress). #83 moved the return past that call for mirror-staging units only, leaving the fixed
   route's ordering unchanged.
4. **Forced-token staging.** A lent unit staged its io before the region was entered, so the MTP
   chunk captured the static image's address at wide-layout offsets (illegal address on a
   grammar-forced request). #84 enters the region before any staging, in `advance_prefill` and
   `append_forced`.

Both #83 and #84 changed only the auto route's behavior; the fixed route's counters
(`workspace_peak_bytes`, `kv_payload_bytes`, `spec_rounds`, `spec_fallback_steps`, acceptance) are
identical to their pre-change values in the bench runs.
