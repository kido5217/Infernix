# QSA pooled-index and selection capture (ticket #40)

Recorded data for the map "bit-plane QSA index scan (issue #5 execution)": the pooled QSA index-key
plane as the engine stores it, the index query the scan consumed, and the full-BF16 block selection,
at ≥ 128K context and at a smaller 16K context. The capture hook lives on the throwaway branch
`proto/qsa-index-capture`, is off by default, and is never merged.

Everything here is a record, not an analysis. The intended use is an offline prototype that
recomputes `score_b = (1/√Di)·Σ_h ReLU(⟨index_q_h, pooled_key_b⟩)` from `pooled.bin` + `index_q.bin`
and compares its top-512 against `selected.bin`.

## Contents

| File | What |
|---|---|
| `make_prompt.py` | deterministic prompt generator (line-numbered records) |
| `prompt_16k.txt` | the 16K prompt (160 lines), token count and sha256 below |
| `prompt_140k.txt` | the ≥128K prompt (980 lines) |
| `manifest-16k.txt` | `<sha256>  <bytes>  <relpath>` for every file of the 16K capture |
| `manifest-140k.txt` | the same for the ≥128K capture |
| `verify_capture.py` | recomputes scores from a capture and checks the recorded selection |
| `sample/` | small full copies and metas of real capture files (see "Sample") |

The full captures are **not** committed. They live on this host at:

- 16K: `/home/kido/network/projects/Infernix/out/qsa-index-capture/16k/`
- ≥128K: `/home/kido/network/projects/Infernix/out/qsa-index-capture/140k/`

(`out/` is gitignored; a later session on this host can read them directly, and every file's sha256
and size is in the committed manifests.)

## Runs

| | 16K | ≥128K |
|---|---|---|
| engine commit | the branch's HEAD (see the commit adding this directory) | same |
| serve port | 8841 | 8842 |
| `--max-context` | 147456 | 147456 |
| prompt | `prompt_16k.txt`, 160 lines | `prompt_140k.txt`, 980 lines |
| prompt tokens (`count_tokens`) | 22433 | 141703 |
| prompt sha256 | `6580aad7b58e449804025ccf10029f30f019bd3c98dad0449c5714b7edad80e3` | `af1c0d720fe0c2306cad35e90c906073e731ebcbd084235d03677c3aa39b469b` |
| captured steps (eager `Forward::run`) | 4 | 4 |
| plane captures | 172 | 97 |
| selection captures | 122 | 72 |
| capture files | 1836 | 1061 |
| capture bytes | 1,068,783,092 | 1,907,724,940 |
| context reached | prompt positions 0..22432, decode to 22439 | positions 131072..141709; the last prefill chunk ends at 141695 |
| complete blocks per column | up to 5,610 | up to 35,424 |

The ≥128K capture arms on `INFERNIX_QSA_CAPTURE_MIN_POS=131072`, so every selection it records has a
maximum absolute position of at least 131072 — more than 128K tokens of context. Its six position
ranges are `[131072,135167]`, `[135168,139263]`, `[139264,141695]` (4096/4096/2432 columns) and three
decode-phase ranges with 7, 2 and 1 columns; each carries the 12 QSA attention layers.

## The hook

`src/ops/qsa/qsa_capture.{h,cu}` (new) with three call sites:

- `Forward::run` (`src/models/qwen4_exp/execution/forward.cpp`) calls `capture::step_begin`, which
  only numbers the capture steps.
- `ops::qsa_pool_keys` (`src/ops/qsa/qsa.cu`, the writer) calls `capture::plane`, which dumps the
  pooled plane pages the call covers, for every QSA layer, the MTP block included.
- `ops::detail::qsa_select` (`src/ops/qsa/qsa_select.cu`, the reader) calls `capture::selection`,
  which dumps the same plane plus the index query, the selected block ids and the counts.

Environment (all inert when unset):

```
INFERNIX_QSA_CAPTURE=<dir>        output directory; unset/empty disables everything
INFERNIX_QSA_CAPTURE_MIN_POS=<p>  capture only calls whose maximum absolute KV position ≥ p
INFERNIX_QSA_CAPTURE_MAX=<n>      stop after n plane and n selection captures
```

A call whose stream is capturing a CUDA graph is skipped: a synchronous copy is illegal inside a
capture, and the qwen4_exp decode/verify rounds run inside CUDA graphs after their first eager
execution. The engine's layer-walked prefill makes one `Forward::run` cover several 4096-token
chunks, so a "step" is one such call, not one chunk.

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
    (the same four files) + index_q.bin selected.bin counts.bin
```

`meta.txt` records `min_position`/`max_position`, `columns`, `complete_blocks`,
`plane_pages_used`, `plane_page_bytes`, `pooled_ptr`, the full `geometry` line, and the sizes of the
tensors. `pooled_ptr` is the plane's device address; the same QSA layer keeps the same pointer across
captures of a run, so plane and selection captures can be paired by it.

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
- The pooled key of block `b` is 128 wide: slot `s` (0..3) holds `key[32·s + lane]` of token
  `4·b + s`, i.e. `key[32·s + lane] = bf16(pooled.bin at page (4b+s)>>6)[slots·((4b+s)&63) + lane]`.

## Geometry observed

Every capture records the same geometry, read from `QsaGeometry` at runtime:

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

## Provenance

- Artifact (sha256 of each part; the unit names only part 1 and the loader finds the siblings):
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix`
    `ed86ef27d5d7ef009466f5bc328f6009f587bbe2d3f94e9a582eaf5e75e89604`
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00002-of-00003.infernix`
    `07e1889f188649cbe03c5f45c9f6f00f0b91c0891ad498067dd33b7ac0c9119d`
  - `Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00003-of-00003.infernix`
    `99bcafd5a97f53f0e1cd9e24b9dd6152154722d048519534a0ed314a8339139f`
- Hardware/toolchain: NVIDIA GeForce RTX 5090, CUDA 13.1 (`sm_120a`), `nix develop .`
  (cmake 4.1.6, Python 3.13, ninja).
- Sampling: request body `{"max_tokens": 8, "temperature": 0.0}` — greedy, so the capture is a
  function of the context alone.

### Serve flags (identical for both runs except the port)

```
build/apps/infernix-serve \
  /home/kido/ai/models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume /home/kido/ai/models/infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --host 127.0.0.1 --port <8841|8842> \
  --model-id Qwen3.8-Flash-Next --kv-dtype nvfp4 \
  --max-context 147456 --rope-yarn-factor 4 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --max-concurrency 1 --expert-ram-mib 66000 --no-thinking \
  --max-pending-requests 4 --media-cache-mib 512
```

Deviations from the deployed unit `infernix-qwen38-flash-next`
(`nixos-configs/users/kido/llm.nix`): `--max-context 1048576 → 147456` (the capture uses 141,703
prompt tokens + 8 generated), `--max-concurrency 4 → 1`, port `8828 → 8841/8842`, and `--no-thinking`
added while `--vision`, `--preserve-thinking`, `--reasoning-loop`, `--default-thinking-budget`,
`--request-log-jsonl`, `--pending-timeout-ms` and `--media-cache-mib 2048` were dropped (no images,
one bounded request). Everything else matches the unit.

## Reproduce

```bash
nix develop . --command cmake --build build -j --target infernix-serve
python3 docs/research/qsa-index-capture/make_prompt.py 980 > prompt_140k.txt   # the committed prompt

INFERNIX_QSA_CAPTURE=$PWD/out/qsa-index-capture/140k \
INFERNIX_QSA_CAPTURE_MIN_POS=131072 INFERNIX_QSA_CAPTURE_MAX=256 \
  nix develop . --command build/apps/infernix-serve <artifact> --ngram-volume <volume> \
    --host 127.0.0.1 --port 8842 --model-id Qwen3.8-Flash-Next --kv-dtype nvfp4 \
    --max-context 147456 --rope-yarn-factor 4 --prefill-chunk 4096 \
    --spec mtp --draft-tokens 4 --lm-head-draft --max-concurrency 1 --expert-ram-mib 66000 \
    --no-thinking --max-pending-requests 4 --media-cache-mib 512 &

curl -s -H 'content-type: application/json' \
  --data '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"<prompt>"}]}' \
  http://127.0.0.1:8842/v1/messages/count_tokens
curl -s -H 'content-type: application/json' \
  --data '{"model":"Qwen3.8-Flash-Next","max_tokens":8,"temperature":0.0,"messages":[{"role":"user","content":"<prompt>"}]}' \
  http://127.0.0.1:8842/v1/messages
```

`nix develop .` is required so the process sees the NVIDIA driver libraries (`libcuda.so.1`).

## Sample

`sample/` holds committed copies (`sample/SHA256SUMS` lists their hashes):

- `selection-decode-16k/` — a complete 16K decode selection (`columns=1`, `max_position=22439`,
  5610 complete blocks, 1.4 MiB of plane pages). `verify_capture.py` runs on this directory and
  reproduces the recorded 512 selected blocks (`exact-set 1/1`).
- `selection-mtp-warmup-16k/` — a complete width-1 draft-step selection (the MTP block's plane,
  positions 12).
- `selection-140k/meta.txt`, `plane-140k/meta.txt` — the metas of a ≥128K selection and plane capture
  (block counts, page counts, `pooled_ptr`); their bulk files are on this host and hashed in
  `manifest-140k.txt`.

## Validation

`verify_capture.py CAPDIR [n_columns]` recomputes `score_b` from `pooled.bin` + `index_q.bin` and
compares its top-512 to `selected.bin`:

- 16K `selections/step_02/s_050` (4096→1946 columns, 5,606 blocks): checked 6 columns, exact-set 6/6,
  order-robust 6/6.
- 16K `selections/step_04/s_111` (decode, 5,610 blocks): exact-set 1/1, order-robust 1/1.
- ≥128K `selections/step_02/s_001` (4096 columns, 33,792 blocks): checked 4 columns, exact-set 4/4,
  order-robust 4/4.

So an offline reader reconstructs the exact selected block set from the captured plane and query.

## Inertness

With `INFERNIX_QSA_CAPTURE` unset nothing is copied and no file is written; the hooks read one cached
flag and return. Evidence:

- `build/tests/infernix_qsa_select_test` and `build/tests/infernix_qsa_test`, run with and without
  `INFERNIX_QSA_CAPTURE=<dir> INFERNIX_QSA_CAPTURE_MIN_POS=0 INFERNIX_QSA_CAPTURE_MAX=1`, produced
  byte-identical stdout apart from the hook's one-line "enabled" banner, and the enabled runs wrote
  their captures.
- The 16K request with the hook enabled and disabled produced the same response (`content` and
  `usage` equal: `input_tokens 22433`, `output_tokens 8`).

## Limitations

- **MTP selection at ≥ 128K is not capturable here.** The MTP block's prefill `qsa_pool_keys` runs
  with `kv_only`, so it selects nothing; its decode-phase selection runs inside a CUDA graph after
  the first warm-up call, and the graph reuse is not controllable from these flags. The ≥128K
  capture therefore holds the 12 QSA attention layers' selections and the MTP block's *pooled planes*
  (the writer hook fires for it, in 512-token windows); MTP *selections* exist only in the 16K
  capture at the warm-up positions 12–13.
- **Plane captures are per-call position windows.** A plane capture's `pooled.bin` covers all pages up
  to its own `max_position`; the full plane for a given context comes from the capture with the
  largest `max_position` for that plane, or by using `block_table.bin` to index page by page.
- **No explicit QSA layer index.** The ops API passes a `QsaKVLayer`, not an index; the step's
  `layer_call` order and `pooled_ptr` are the identity available.
- One ≥128K serve start failed ("dense weights were placed in shared system memory") because the
  preceding capture's process had not released VRAM yet; it was restarted after the GPU was free.
