# QSA pooled-index, selection and engine-score capture (ticket #40)

Recorded data for the map "bit-plane QSA index scan (issue #5 execution)" and the pre-registration
(issue #39, rev 3): the pooled QSA index-key plane as the engine stores it, the index query the scan
consumed, the full-BF16 top-512 block selection, and **the engine's own FP32 scores** — at ≥ 128K
context and at a smaller 16K context. The capture hook lives on the throwaway branch
`proto/qsa-index-capture`, is off by default, and is never merged.

Everything here is a record, not an analysis. The pre-registration's baseline A0 is the engine's own
numbers, so an
offline reader must reconstruct `score_b` from `pooled.bin` + `index_q.bin` and compare it against
`scores.bin`, not against a reimplementation. `verify_capture.py` does exactly that and reports the
max absolute and relative difference.

## Contents

| File | What |
|---|---|
| `make_prompt.py` | deterministic prompt generator (line-numbered records) |
| `prompt_16k.txt`, `prompt_140k.txt` | the two prompts, token counts and sha256 below |
| `prompt_lines.txt` | the line counts that regenerate them |
| `manifest-16k.txt`, `manifest-140k.txt` | **Run A** (pre-extension): planes/queries/selections |
| `manifest-scores-16k.txt`, `manifest-scores-140k.txt` | **Run S**: the same captures re-recorded with the engine's FP32 scores |
| `manifest-eager-16k.txt`, `manifest-eager-140k.txt` | **Run E**: eager runs that cover all 13 planes and ≥ 8 steps at ≥ 128K |
| `manifest-*.commit.txt` | the engine revision behind each manifest (the manifests stay data-only) |
| `verify_capture.py` | recomputes scores/ids from a capture and checks them against the engine's |
| `verify_repro.py` | checks that one capture root's content files reappear byte-identically under another |
| `sample/` | small committed copies of real capture files (see "Sample") |

Each manifest is a list of `sha256  bytes  relpath` for every file of its run; its sibling
`manifest-<name>.commit.txt` records the engine revision and the commit that was checked out when the
run was captured. The captures are not committed; they live on this host at
`/home/kido/network/projects/Infernix/out/qsa-index-capture/`:

| run | contexts | where |
|---|---|---|
| A | 16K, ≥128K | `16k/`, `140k/` |
| S | 16K, ≥128K | `scores-16k/`, `scores-140k/` |
| E | 16K, ≥128K | `16k-eager/`, `140k-eager/` |

(`out/` is gitignored; a later session on this host reads them directly.)

## Which run produced what

- **Run A** is the original capture: pooled planes, index queries and selections, no scores. It is
  kept byte-identical.
- **Run S** re-ran the identical configuration (same binary path, same flags, same prompt, greedy,
  `max_tokens 8`) with the score hook. **Every content file of Run A reappears path-for-path
  byte-identical** (16K: 1542/1542; ≥128K: 892/892, `meta.txt` excluded because it now carries the
  score fields), and Run S adds exactly one `scores.bin` per selection capture (122 and 72). So the
  score files provably belong to the same steps and columns as the selections already recorded
  (`verify_repro.py` reports this, and the per-file check is the manifest comparison).
- **Run E** re-ran with `INFERNIX_QSA_CAPTURE_EAGER=1` and `max_tokens 128/64`, so the rounds the
  CUDA graph owns also run in the open. It covers **all 13 planes** (12 QSA attention layers + the
  MTP block) and, at ≥ 128K, **17 steps** — pre-registration §7 rule 9. Each selection carries its
  scores.

| | Run A (16K / ≥128K) | Run S (16K / ≥128K) | Run E (16K / ≥128K) |
|---|---|---|---|
| steps | 4 / 4 | 4 / 4 | 17 / 17 |
| selections | 122 / 72 | 122 / 72 | 300 / 260 |
| score files | 0 | 122 / 72 | 300 / 260 |
| distinct planes | 13 / 13 | 13 / 13 | 13 / 13 |
| files | 1836 / 1061 | 1958 / 1133 | 4200 / 3640 |
| bytes | 1,068,783,092 / 1,907,724,940 | 2,291,109,932 / 2,605,523,308 | 2,796,149,999 / 5,854,531,186 |

The six manifests hold **13,828** entries in total — 11,782 content files and 2,046 `meta.txt` — and
every entry re-hashes clean against its recorded sha256 and byte count.

## Prompts and contexts

| | 16K | ≥128K |
|---|---|---|
| prompt | `prompt_16k.txt` (160 lines) | `prompt_140k.txt` (980 lines) |
| prompt tokens (`/v1/messages/count_tokens`) | 22,433 | 141,703 |
| prompt sha256 | `6580aad7b58e449804025ccf10029f30f019bd3c98dad0449c5714b7edad80e3` | `af1c0d720fe0c2306cad35e90c906073e731ebcbd084235d03677c3aa39b469b` |
| positions (Run A) | 0…22432, decode to 22439 | 131072…141695, decode to 141709 |
| complete blocks at the deepest column | 5,610 | 35,426 |

The ≥128K run arms on `INFERNIX_QSA_CAPTURE_MIN_POS=131072`, so every selection it records has a
maximum absolute position of at least 131072 — more than 128K tokens of context. `--max-context
147456` is the largest that fits the 141,703-token prompt plus generated tokens.

## The hook

`src/ops/qsa/qsa_capture.{h,cu}` (new) with three call sites:

- `Forward::run` (`src/models/qwen4_exp/execution/forward.cpp`) calls `capture::step_begin`, which
  only numbers the capture steps.
- `ops::qsa_pool_keys` (`src/ops/qsa/qsa.cu`, the writer) calls `capture::plane`, which dumps the
  pooled plane pages the call covers, for every QSA layer, the MTP block included.
- `ops::detail::qsa_select` (`src/ops/qsa/qsa_select.cu`, the reader) calls `capture::selection`,
  which dumps the plane, the index query, the selection, and the engine's own FP32 scores
  (`ops::detail::qsa_select_scores`, `src/ops/qsa/qsa_select.h:69-73`).
- `ProgramImpl::replay` (`src/models/qwen4_exp/program/program_impl.h`) consults `capture::eager()`:
  with `INFERNIX_QSA_CAPTURE_EAGER` set it runs each round eagerly instead of capturing and replaying
  it, so the rounds the graph owns still reach the hooks (see "The MTP attempt").

Environment (all inert when unset):

```
INFERNIX_QSA_CAPTURE=<dir>        output directory; unset/empty disables everything
INFERNIX_QSA_CAPTURE_MIN_POS=<p>  capture only calls whose maximum absolute KV position ≥ p
INFERNIX_QSA_CAPTURE_MAX=<n>      stop after n plane and n selection captures
INFERNIX_QSA_CAPTURE_EAGER=1       run every CUDA-graph round eagerly (Run E)
```

A call whose stream is capturing a CUDA graph is skipped: a synchronous copy is illegal inside a
capture. The engine's layer-walked prefill makes one `Forward::run` cover several 4096-token chunks,
so a "step" is one such call, not one chunk.

## File layout

```
<root>/
  planes/step_XX/p_YYY/        one per qsa_pool_keys call
    pooled.bin                 the used plane pages, token-page order (see "Plane bytes")
    positions.bin              I32[columns]   absolute KV position of each column
    block_table.bin            I32[stride]    the sequence's page table (row of block_tables)
    table_rows.bin             I32[batch]     the table row of each sequence
    meta.txt                   geometry, positions, sizes, pooled_ptr
  selections/step_XX/s_YYY/    one per qsa_select call
    (the same four files) + index_q.bin selected.bin counts.bin scores.bin
```

`meta.txt` records `min_position`/`max_position`, `columns`, `complete_blocks`,
`plane_pages_used`, `plane_page_bytes`, `pooled_ptr`, the full `geometry` line, the tensor shapes,
and the score-group fields. `pooled_ptr` is the plane's device address; the same QSA layer keeps the
same pointer across a run, so plane and selection captures (and the MTP plane) can be paired by it.

### Element layouts (little-endian)

`ne` is fastest-dimension-first in this engine: element `(i, j, k)` of a 3-D tensor lies at
`i + ne[0]·(j + ne[1]·k)`.

- `index_q.bin` — BF16 `ne=[Di, index_heads, columns]`; `index_q(d, h, c)` at `d + 128·(h + 4·c)`.
- `selected.bin` — I32 `ne=[top, columns]`; column `c`'s ascending block ids at
  `selected[c·top + i]`; `counts.bin` — I32 `ne=[columns]`, `top` or `-1` (the column selects every
  block).
- `positions.bin` — I32 `ne=[columns]`, the absolute KV position of each column.
- `pooled.bin` — the pooled plane `ne=[Di/ratio, 64, 1, pages]`, **only the used pages** (the elastic
  KV pool leaves the rest of the reservation unmapped), in token-page order: token page
  `j = token >> 6` sits at byte offset `j · plane_page_bytes`, and within a page the element for
  token `t` and slot `s` is at `2·(slots·(t & 63) + s)`. `plane_page_bytes = 32·64·2 = 4096`.
- `scores.bin` — FP32 `ne=[scores_group_columns, scores_stride]`, the engine's FP32 scores for the
  selection's **last group** (the group `qsa_select_scores` exposes). The group is the last
  `min(columns, 128)` columns of the call: `scores_group_begin … columns-1`. Group column `c`
  (absolute column `scores_group_begin + c`) holds block `b` at `scores[c·scores_stride + b]` for
  `b < its complete block count`. `scores_stride = (max_context/ratio + 1 + 31)/32*32`
  (`src/ops/qsa/qsa_select.cu:794`) = 36896 floats with `--max-context 147456`; the `+1` reserves the
  block at `max_context`, and `max_context/ratio` alone (36864) is already a multiple of 32.
- The pooled key of block `b` is 128 wide: slot `s` (0..3) holds `key[32·s + lane]` of token
  `4·b + s`, i.e. `key[32·s + lane] = bf16(pooled.bin at page (4b+s)>>6)[slots·((4b+s)&63) + lane]`.

## Geometry observed

| field | value |
|---|---|
| heads / kv_heads / head_dim | 24 / 2 / 256 |
| index_heads / index_head_dim (Di) | 4 / 128 |
| ratio (R) | 4 |
| budget / top_blocks | 2048 / 512 |
| rotary_dim / theta / eps | 64 / 1e7 / 1e-6 |
| yarn_factor / original_positions | 4 / 262144 |
| plane | BF16 `[Di/R, 64, 1, pages]` = `[32, 64, 1, pages]` |
| bytes per token per layer | 64 (one 128-wide BF16 key per 4-token block) |
| bytes per block per layer | 256 |
| score stride | 36896 floats = `(max_context/ratio + 1 + 31)/32*32` at `--max-context 147456` (`src/ops/qsa/qsa_select.cu:794`) |

## Provenance

- Artifact (sha256 of each part; the unit names only part 1 and the loader finds the siblings):
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix`
    `ed86ef27d5d7ef009466f5bc328f6009f587bbe2d3f94e9a582eaf5e75e89604`
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00002-of-00003.infernix`
    `07e1889f188649cbe03c5f45c9f6f00f0b91c0891ad498067dd33b7ac0c9119d`
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00003-of-00003.infernix`
    `99bcafd5a97f53f0e1cd9e24b9dd6152154722d048519534a0ed314a8339139f`
- Engine revision: every capture was produced by the `proto/qsa-index-capture` branch tree, never
  `main` (`main` was `1e2eda5b131bff81a225c2d3c8088c94b54e4e08`). Run A was built from the tree
  committed as `d37ae276cdd1cdcebb93b6b75abc2bcac0a4b4a2` (the branch had `1e2eda5b` checked out while
  the hook was still uncommitted); Runs S and E from the tree committed as
  `69fcf7d33b7fccad74f925dabbee3fa70fe47ad5` (with `51601ea280c51d5e206fc3903004df197428476c` checked
  out while the score and eager changes were uncommitted). Each manifest has a sibling
  `manifest-<name>.commit.txt` recording its revision.
- Hardware/toolchain: NVIDIA GeForce RTX 5090, CUDA 13.1 (`sm_120a`), `nix develop .`
  (cmake 4.1.6, Python 3.13, ninja).
- Sampling: `temperature 0.0`, `max_tokens` 8 (Runs A/S) and 64 (≥128K Run E) / 128 (16K Run E) —
  greedy, so each capture is a function of the context alone.

### Serve flags (identical for every run except the port and these)

```
build/apps/infernix-serve \
  /home/kido/ai/models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume /home/kido/ai/models/infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --host 127.0.0.1 --port <8841..8849> \
  --model-id Qwen3.8-Flash-Next --kv-dtype nvfp4 \
  --max-context 147456 --rope-yarn-factor 4 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --max-concurrency 1 --expert-ram-mib 66000 --no-thinking \
  --max-pending-requests 4 --media-cache-mib 512
```

Deviations from the deployed unit `infernix-qwen38-flash-next`
(`nixos-configs/users/kido/llm.nix`): `--max-context 1048576 → 147456` (the capture uses 141,703
prompt tokens + generated), `--max-concurrency 4 → 1`, `--max-pending-requests 64 → 4`,
`--media-cache-mib 2048 → 512`, and port `8828 → 8841..8849`. `--no-thinking` was added, and
`--vision`, `--preserve-thinking`, `--reasoning-loop`, `--default-thinking-budget`,
`--request-log-jsonl` and `--pending-timeout-ms` were dropped. Everything else — the artifact and
n-gram volume paths, host, model id, KV dtype, `--rope-yarn-factor`, `--prefill-chunk`, `--spec mtp`,
`--draft-tokens`, `--lm-head-draft` and `--expert-ram-mib` — matches the unit. The capture runs the
locally built `build/apps/infernix-serve` rather than the unit's packaged `pkgs.infernix` binary,
because the capture hook is not in the packaged build.

## Reproduce

```bash
nix develop . --command cmake --build build -j --target infernix-serve
python3 docs/research/qsa-index-capture/make_prompt.py 980 > prompt_140k.txt   # the committed prompt

# Run S (scores; identical to Run A) — or add INFERNIX_QSA_CAPTURE_EAGER=1 for Run E.
INFERNIX_QSA_CAPTURE=$PWD/out/qsa-index-capture/scores-140k \
INFERNIX_QSA_CAPTURE_MIN_POS=131072 INFERNIX_QSA_CAPTURE_MAX=256 \
  nix develop . --command build/apps/infernix-serve <artifact> --ngram-volume <volume> \
    --host 127.0.0.1 --port 8848 --model-id Qwen3.8-Flash-Next --kv-dtype nvfp4 \
    --max-context 147456 --rope-yarn-factor 4 --prefill-chunk 4096 \
    --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 1 --expert-ram-mib 66000 \
    --no-thinking --max-pending-requests 4 --media-cache-mib 512 &

curl -s -H 'content-type: application/json' \
  --data '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"<prompt>"}]}' \
  http://127.0.0.1:8848/v1/messages/count_tokens
curl -s -H 'content-type: application/json' \
  --data '{"model":"Qwen3.8-Flash-Next","max_tokens":8,"temperature":0.0,"messages":[{"role":"user","content":"<prompt>"}]}' \
  http://127.0.0.1:8848/v1/messages
```

`nix develop .` is required so the process sees the NVIDIA driver libraries (`libcuda.so.1`).

## Sample

`sample/` holds committed copies (`sample/SHA256SUMS` lists their hashes):

- `selection-decode-16k/` — a complete 16K decode selection **with its `scores.bin`**
  (`columns=1`, `max_position=22439`, 5610 complete blocks). `verify_capture.py` runs on this
  directory and reports both the score difference and the reproduced 512 ids.
- `selection-mtp-warmup-16k/` — the MTP plane's width-1 draft-step capture at 16K. It is a **dense
  short-circuit** capture: with `complete_blocks 3 ≤ top_blocks 512` the engine takes the dense path,
  so `counts.bin = -1`, `selected.bin` is never written and `scores.bin` is unwritten scratch memory
  (its first floats are garbage). It is kept only as the record of that case; do not read it as a
  selection with scores.
- `selection-mtp-140k/` — a complete **MTP selection at ≥ 128K** (`position=141706`, width 1, 35426
  blocks), `pooled.bin` (9 MiB) included with the rest; its sha256 is in `manifest-eager-140k.txt`.
  The directory is complete, so `verify_capture.py` runs on it standalone.
- `selection-140k/meta.txt`, `plane-140k/meta.txt` — the metas of Run A's ≥128K selection and plane.

## Validation

`verify_capture.py CAPDIR [n_columns]` recomputes `score_b` from `pooled.bin` + `index_q.bin`, compares
it against the engine's `scores.bin`, and checks the recorded top-512 against the recomputed one. Dense
columns (`counts = -1`) are excluded from both checks and counted as excluded. For every checked column
the boundary gap `|s(engine's last selected) − s(engine's first rejected)|` is computed, and a top-512
mismatch is classified **explained** when that gap is below the column's measured score error and
**unexplained** otherwise; a mismatch in a score-less run is **unclassified**. The score error is the
max absolute difference and the max column-scaled relative difference `|Δ| / (max − min)` of the column.

| capture | blocks | cols checked | max abs diff | max rel diff (elementwise / column-scaled) | ids exact-set |
|---|---|---|---|---|---|
| `scores-140k/selections/step_02/s_001` | 33,792 | 4 | 2.651e-06 | 3.801e-07 / 1.970e-07 | 4/4 |
| `scores-16k/selections/step_04/s_111` | 5,610 | 1 | 1.131e-06 | 3.381e-07 / 8.584e-08 | 1/1 |
| `140k-eager/selections/step_04/s_064` (MTP) | 35,426 | 1 | 3.268e-06 | 3.423e-07 / 2.449e-07 | 1/1 |

`verify_capture.py --sweep out/qsa-index-capture 8` runs every selection capture (up to 8 leading
columns each, the single-capture default) and prints one row per manifest:

```
manifest         captures ids-chk  exact  expl unexpl unclass  dense  near  max|dScore| max col-scaled
------------------------------------------------------------------------------------------------------
140k                   72     408    407     0      0       1      0     0          n/a            n/a
140k-eager            260     812    811     1      0       0      0     2    3.930e-06      3.369e-07
16k                   122     576    575     0      0       1    230     0          n/a            n/a
16k-eager             300     987    986     1      0       0    230     1    3.892e-06      4.284e-07
scores-140k            72     408    407     1      0       0      0     2    3.930e-06      3.249e-07
scores-16k            122     576    575     1      0       0    230     1    3.397e-06      4.284e-07
```

`ids-chk` counts non-dense columns checked, `dense` the dense columns excluded, `near` the columns
whose boundary gap is within the measured error, and `unclass` the mismatches in the score-less Run A.
This sweep samples 8 columns per capture; pre-registration §3.1's agreement is over every captured
column.

Every checked column is within `1e-5` absolute and `1e-6` column-scaled relative (the pre-registration's
fidelity bound; the sweep maxima are 3.930e-06 and 4.284e-07). The sweep finds **two** near-tie id
flips, both **explained** and neither unexplained:

- `scores-16k/selections/step_02/s_089`, absolute column 2: the engine selects block 1011 and rejects
  378, the numpy recompute the reverse; boundary gap 6.743e-07 against a 2.674e-06 score error.
- `scores-140k/selections/step_02/s_006`, absolute column 7: the engine selects 21604 and rejects
  19086, the recompute the reverse; the recomputed scores of the two are exactly equal (gap 0) against
  a 3.306e-06 score error.

Both columns lie outside their capture's scored last group, so the capture's maximum measured error is
the classification threshold. Both flips recur in Run E and, as **unclassified** (no scores), in Run A;
aside from them the recomputed top-512 equals the engine's selected set on every checked non-dense
column.

## Inertness

With `INFERNIX_QSA_CAPTURE` unset nothing is copied, no file is written and `replay()` is untouched
(the hooks read one cached flag and return). Evidence, from the extension's rebuilt binaries:

- `build/tests/infernix_qsa_select_test` and `build/tests/infernix_qsa_test`, run with and without
  `INFERNIX_QSA_CAPTURE=<dir> INFERNIX_QSA_CAPTURE_MIN_POS=0 INFERNIX_QSA_CAPTURE_MAX=1`, produced
  byte-identical stdout apart from the hook's one-line "enabled" banner, and the enabled runs wrote
  their captures.
- The 16K request with the hook enabled and disabled produced the same response (`content` and
  `usage` equal: `input_tokens 22433`, `output_tokens 8`).

## The MTP attempt (pre-registration §7 rule 9)

**No existing flag or environment variable disables CUDA graphs for this model.** Evidence:

- `--no-cuda-graph` is parsed (`src/serve/serve_options.cpp`, `use_cuda_graph`) and reaches
  `EngineOptions::use_cuda_graph` (`include/infernix/types.h`), but the qwen4_exp Program never reads
  it: `src/runtime/engine/qwen4_exp_instance.cpp` builds `program::ProgramOptions`
  (`src/models/qwen4_exp/program/program.h`) with **no graph field**, and every decode, verify and MTP
  round goes through `ProgramImpl::replay` (`src/models/qwen4_exp/program/program_impl.h`), which
  captures on the second use of each slot and replays afterwards. `grep use_cuda_graph` over
  `src/models/qwen4_exp/` finds nothing.
- The only `getenv` calls in the tree are `INFERNIX_CUDA_SYNC` (`src/core/device.cu`), `WT_SESSION`
  (log colour) and this ticket's own `INFERNIX_QSA_CAPTURE*`. There is no graph switch.
- The MTP path is `run_mtp` → `mtp_block` → `attention` → `qsa_attention` → `qsa_select`, wrapped by
  `replay(mtp_draft_graphs_[…], …)`; after warm-up the body only runs inside the graph, which is why
  the pre-extension ≥128K capture held the MTP *plane* (written by the eager, `kv_only` prefill
  pooling) but no MTP *selection*.

**What was done instead:** a new off-by-default switch, `INFERNIX_QSA_CAPTURE_EAGER=1`, makes
`ProgramImpl::replay` run its body eagerly instead of capturing/replaying it. It executes the
engine's own kernels, not an offline reimplementation — the captured selections and scores are the
engine's. With it, Run E covers all 13 planes. At ≥128K its plane captures exist for 15 of its 17
steps, 14 of them carrying 13 distinct pooled planes (`step_02`…`step_15`) and `step_16` carrying 6;
counting the selections' distinct `pooled_ptr` instead gives 16 of 17 steps (all but `step_02`, which
carries 12). At 16K, 12 of the 13 plane-captured steps carry 13 planes and `step_13` carries 3 (15 of
17 by selection `pooled_ptr`). Pre-registration §7 rule 9 — ≥ 8 steps and all 13 planes — is met at
both contexts: 17 steps, 13 distinct pooled planes by `pooled_ptr`. Run E also includes MTP width-1
selections (e.g. `140k-eager/selections/step_04/s_064`, position 141706, 35,426 blocks, validated
above).

Limit of that evidence: Run E's decode-round schedule is not identical to Run A's (Run A captured
only the first eager execution of each graph slot; Run E runs them all), so 25 (16K) and 24 (≥128K)
of Run A's decode-phase `index_q`/`selected` files do not reappear in Run E. Every prefill-phase
content file, and all of Run A, reappears in Run S byte-identically.

## Limitations

- **No explicit QSA layer index.** The ops API passes a `QsaKVLayer`, not an index; the step's
  `layer_call` order and `pooled_ptr` are the identity available.
- **`scores.bin` is the last group only** — the accessor `qsa_select_scores` exposes one group
  (≤ 128 columns) of a multi-group call, so a 4096-column prefill capture carries scores for its last
  128 columns; decode captures (`columns ≤ 7`) carry all of them.
- **Plane captures are per-call position windows.** A plane capture's `pooled.bin` covers all pages up
  to its own `max_position`.
- One ≥128K serve start once failed with "dense weights were placed in shared system memory" because
  the preceding capture's process had not released VRAM; it was restarted after the GPU was free.
