#pragma once

// The prefill chunk-size model of `--prefill auto` (design §13, ticket #73 phase 1): the pure price
// of the loan a prompt-path prefill of one chunk width takes from the expert cache, and the
// bisection over the candidate grid that picks the largest chunk the live lendable budget funds.
// Every term mirrors a parametric buffer term of the device layout (the workspace arena, the
// per-column io staging, the residency route log, the streamed ring), so a resolved chunk is
// priced exactly as the buffers it runs in are laid out (program_impl.h). Pure and host-only:
// qualified on CPU (infernix_qwen4_exp_prefill_sizing_test); the hardware behavior is #76.

#include "models/qwen4_exp/config.h"

#include <cstdint>

namespace infernix::models::qwen4_exp::prefill_sizing {

// The chunk candidates: multiples of `step` up to `ceiling`. The step is 128 tokens (every prefill
// chunk is a multiple of 128); with the fast INT8-KV prompt kernel it is whole prompt-attention
// waves (896 tokens for the 24-head model on the RTX 5090: Basalt's bare 256 grid does not fit
// Infernix's kernels), so no candidate call ends in a partial wave. The ceiling is the operator's
// `--prefill auto[:N]` value (the 8192/16384/32768 grid) cut to max_context.
struct ChunkGrid {
    std::uint32_t step    = 128;
    std::uint32_t ceiling = 0;

    // The largest grid point the grid yields, or zero when it yields none.
    [[nodiscard]] std::uint32_t max_chunk() const noexcept {
        return step != 0 && ceiling >= step ? ceiling / step * step : 0U;
    }
};

// The price of one prompt's loan: the buffer set a chunk-wide prefill call runs in, taken from
// expert frames while its experts stream. Every term is nondecreasing in the chunk, so `bytes` is
// monotone — the precondition the bisection reads. The residency route-log term is included even
// though the table is sized statically at the ceiling: the loan budget is lendable frames minus the
// fixed allocations the same layout already charged, and the per-column term keeps the price
// honest for any caller that leaves the table chunk-scaled.
struct LoanTerms {
    // The workspace arena of a call `columns` wide (the forward at its widest, sampling, draft
    // acceptance and the drafter): the plan's parametric workspace_bytes at that width.
    std::uint64_t (*arena)(void* context, std::uint32_t columns) = nullptr;
    void* arena_context = nullptr;
    // Device io staging, dense per-column arrays (ids, positions, RoPE, the gate word, the
    // logit-column list, the n-gram rows, the MTP cell arrays and the vision columns): linear in
    // the chunk. See io_bytes_per_column.
    std::uint64_t io_per_column = 0;
    // The residency route log, I32 [layers][top_k x columns]: linear in the chunk. See
    // route_log_per_column.
    std::uint64_t route_log_per_column = 0;
    // The streamed ring: the frame bytes a streamed chunk lends with its loan (2E + 1 records:
    // halves of every expert plus the slot table, the worst case stream_chunk forms when nearly
    // every expert of a layer is non-resident — the streamed regime's steady state). A chunk
    // narrower than `stream_min_columns` does not stream and lends no ring.
    std::uint64_t stream_ring        = 0;
    std::uint32_t stream_min_columns = 256; // ProgramImpl::kStreamMinColumns
    // The loan's fixed part: the ring's slot-table frame and per-buffer alignment slack.
    std::uint64_t fixed = 0;

    [[nodiscard]] std::uint64_t bytes(std::uint32_t chunk) const noexcept {
        std::uint64_t out = fixed + static_cast<std::uint64_t>(chunk) * (io_per_column + route_log_per_column);
        if (arena != nullptr) { out += arena(arena_context, chunk); }
        if (chunk >= stream_min_columns) { out += stream_ring; }
        return out;
    }
};

// The largest chunk on the grid whose loan fits `budget` bytes, by bisection on the grid (a
// monotone probe sequence, at most log2(ceiling/step) + 1 price evaluations); zero when not even
// the smallest grid chunk fits — the caller then keeps the fixed chunk, the default and fallback.
[[nodiscard]] std::uint32_t largest_fitting_chunk(const ChunkGrid& grid, const LoanTerms& terms,
                                                  std::uint64_t budget) noexcept;

// The dense io staging bytes of one call column: the chunk-scaled arrays of the layout summed per
// column (56 B of I32 arrays plus each column's n-gram rows). Mirrors the io block of
// DeviceLayout::plan_device; the per-lane arrays (block rope, slots, rows) and the sub-chunk
// blocks do not scale with the chunk and price into `fixed`.
[[nodiscard]] std::uint64_t io_bytes_per_column(const TextConfig& c) noexcept;

// The residency route log's bytes per call column: one I32 per routed expert per layer per column.
// The same term ExpertResidency::table_bytes grows by per column.
[[nodiscard]] std::uint64_t route_log_bytes_per_column(const TextConfig& c) noexcept;

// The bytes a streamed chunk's ring lends: halves of every expert of a layer plus one frame for
// the slot tables — 2E + 1 records, the widest half the stream can ask for.
[[nodiscard]] std::uint64_t stream_ring_bytes(const TextConfig& c, std::uint64_t record_stride) noexcept;

} // namespace infernix::models::qwen4_exp::prefill_sizing
