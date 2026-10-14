#include "models/qwen4_exp/program/prefill_sizing.h"

namespace infernix::models::qwen4_exp::prefill_sizing {

std::uint32_t largest_fitting_chunk(const ChunkGrid& grid, const LoanTerms& terms,
                                    std::uint64_t budget) noexcept {
    const std::uint32_t top = grid.max_chunk();
    if (top == 0) { return 0; }
    std::uint32_t lo = 1;                  // grid points: n = chunk / step, in [1, top / step]
    std::uint32_t hi = top / grid.step;
    std::uint32_t best = 0;
    while (lo <= hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (terms.bytes(mid * grid.step) <= budget) {
            best = mid;
            if (mid == hi) { break; }      // lo = mid + 1 would wrap at the grid's top
            lo = mid + 1;
        } else {
            if (mid == 1) { break; }       // the smallest grid chunk does not fit: no chunk fits
            hi = mid - 1;
        }
    }
    return best * grid.step;
}

std::uint64_t io_bytes_per_column(const TextConfig& c) noexcept {
    // The chunk-scaled io arrays of the layout, per call column: ids 4 + positions 4 + rope 12 +
    // the gate word 4 + the logit-column list 4 (28), the MTP ids 4 + cells 4 + rope 12 (20), and
    // the vision and MTP-vision column lists 4 + 4 (8); then the column's n-gram rows.
    const std::uint64_t rows = static_cast<std::uint64_t>(c.ple.table.row_bytes) * c.ple.heads();
    return 56 + rows;
}

std::uint64_t route_log_bytes_per_column(const TextConfig& c) noexcept {
    return sizeof(std::int32_t) * static_cast<std::uint64_t>(c.moe.top_k) * c.num_hidden_layers;
}

std::uint64_t stream_ring_bytes(const TextConfig& c, std::uint64_t record_stride) noexcept {
    // Two halves of every expert plus one frame for the slot tables (program_impl.h stream_chunk's
    // widest possible ring: half = min(E, absent + evicted/layer)).
    return (2ULL * c.moe.experts + 1) * record_stride;
}

} // namespace infernix::models::qwen4_exp::prefill_sizing
