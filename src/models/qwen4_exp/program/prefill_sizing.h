#pragma once

// The prefill chunk-size model of `--prefill auto` (design §13, ticket #73 phase 1): the price of
// the loan a prompt-path prefill of one chunk width takes from the expert cache, and the bisection
// over the candidate grid that picks the largest chunk the live lendable budget funds. Every term
// mirrors a parametric buffer term of the device layout (the workspace arena, the per-column io
// staging, the residency route log, the streamed ring, a walk span's area), and the sum rounds up
// to whole frames as the lease does — the price never under-books what the loan lends, so a chunk
// that fails it only falls back to the fixed chunk, it never over-leases. It also owns the pure
// policy helpers the lending route reads (the wide-staging decision, its growth rule, the
// equal-chunk tail arithmetic), qualified on CPU
// (infernix_qwen4_exp_prefill_sizing_test); the hardware behavior is #76.

#include "models/qwen4_exp/config.h"

#include <cstdint>

namespace infernix::models::qwen4_exp::prefill_sizing {

// The chunk candidates: multiples of `step` up to `ceiling`. The step is 128 tokens (every prefill
// chunk is a multiple of 128); with any fast-kernel KV format it is whole prompt-attention waves
// (896 tokens for the 24-head model on the RTX 5090: Basalt's bare 256 grid does not fit Infernix's
// kernels), so no candidate call ends in a partial wave; the original INT8 and NVFP4 prompt kernels
// and BF16 KV keep the plain grid. The ceiling is the operator's `--prefill auto[:N]` value (the
// 8192/16384/32768 grid) cut to max_context.
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
// monotone — the precondition the bisection reads. The residency route-log term is the per-column
// share of the table (statically ceiling-sized at startup: the charge keeps the price sensitive to
// how much of it a wide route occupies).
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
    // route_log_bytes_per_column.
    std::uint64_t route_log_per_column = 0;
    // The streamed ring: the frame bytes a streamed chunk lends with its loan (2E + 1 records:
    // halves of every expert plus the slot table, the worst case stream_chunk forms when nearly
    // every expert of a layer is non-resident — the streamed regime's steady state). A chunk
    // narrower than `stream_min_columns` does not stream, and lends neither the ring nor a walk
    // span's area.
    std::uint64_t stream_ring        = 0;
    std::uint32_t stream_min_columns = 256; // ProgramImpl::kStreamMinColumns
    // The layer walk's area (design §19.3.8): a span of chunks runs over the same lease, one slot
    // of io image per chunk plus the span's residual columns — priced at the widest span the walk
    // admits (walk_chunks slots, one span's residuals) when the chunk is walk-eligible.
    std::uint64_t walk_residual = 0;
    std::uint64_t walk_chunks   = 0;
    // The lease's rounding unit (the expert frame stride; 0: no rounding — the price is then the
    // byte sum exactly) and the loan's fixed part: per-buffer alignment slack.
    std::uint64_t frame_round = 0;
    std::uint64_t fixed       = 0;

    [[nodiscard]] std::uint64_t bytes(std::uint32_t chunk) const noexcept {
        std::uint64_t out = fixed + static_cast<std::uint64_t>(chunk) * (io_per_column + route_log_per_column);
        if (arena != nullptr) { out += arena(arena_context, chunk); }
        if (chunk >= stream_min_columns) {
            out += stream_ring + walk_residual + walk_chunks * static_cast<std::uint64_t>(chunk) * io_per_column;
        }
        if (frame_round != 0) { out = (out + frame_round - 1) / frame_round * frame_round; }
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

// Whether a prefill unit of one lane must run in the wide io staging: decided from the lane's
// resolved chunk — the widest call its plan can hold — never from one call's width: a span or a
// later call of the prompt may be wider than the first (#73 review, Important 1). Zero: the static
// staging covers every call of the prompt.
constexpr std::int32_t wide_staging_columns(std::uint32_t lane_chunk, std::int32_t static_cap) noexcept {
    return lane_chunk > static_cap ? static_cast<std::int32_t>(lane_chunk) : 0;
}

// The wide staging mirror's growth decision when a prompt needs `needed` columns and a mirror of
// `held` columns stands (a prompt of a larger resolved chunk than the last lender's, or a second
// prefilling lane): reuse it, replace it, or defer — a running layer walk's chunk slots hold the
// current layout's stride, so the mirror may not change under it; the wider prompt then runs its
// wide calls in a temporary per-call region (the transient two-lanes-differ case; #76 exercises it
// on the hardware).
enum class IoGrow { Allocate, Reuse, Replace, Transient };
constexpr IoGrow io_grow_decision(bool mirror_held, std::int32_t held, std::int32_t needed,
                                  bool walk_holds_layout) noexcept {
    if (!mirror_held) { return IoGrow::Allocate; }
    if (needed <= held) { return IoGrow::Reuse; }
    return walk_holds_layout ? IoGrow::Transient : IoGrow::Replace;
}

// The equal-chunk tail rule of design §13 (ported by #73; its measured benefit is #76's): a
// segment of `tokens` over `chunk`-wide calls leaves a tail of `tokens % chunk`; a tail of zero,
// or below `min_tail_tokens` (the separate short-tail call that moves only its own routed experts),
// keeps the plain chunk grid. Otherwise the segment is read as the same call count ⌈tokens/chunk⌉
// in equal widths ⌈tokens/count⌉ (the last call takes the remainder, never wider than the others).
// Returns the segment width, or 0 to keep stepping by the chunk.
constexpr std::uint32_t equal_chunk_width(std::uint32_t tokens, std::uint32_t chunk,
                                          std::uint32_t min_tail_tokens) noexcept {
    if (min_tail_tokens == 0 || chunk == 0 || tokens <= chunk) { return 0; }
    const std::uint32_t tail = tokens % chunk;
    if (tail == 0 || tail < min_tail_tokens) { return 0; }
    const std::uint32_t count = (tokens + chunk - 1) / chunk;
    const std::uint32_t width = (tokens + count - 1) / count;
    return width < chunk ? width : 0;
}

} // namespace infernix::models::qwen4_exp::prefill_sizing
