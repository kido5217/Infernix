// The prefill chunk-size model of `--prefill auto` (program/prefill_sizing.h; design §13, ticket
// #73 phase 1), host only. Checks: grid rounding on the 128 grid and the INT8-KV wave grid; the
// bisection agrees with a linear scan over every grid point at many budgets and returns only grid
// points; no-fit budgets return 0 (the fixed-chunk fallback); the price is monotone in the chunk
// for the real term builders (the bisection's precondition); the per-column io and route-log terms
// and the ring term equal the layout's formulas on a synthetic Qwen4Exp-shaped config; the probe
// count stays logarithmic. The hardware behavior is #76.

#include "models/qwen4_exp/program/prefill_sizing.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ps = infernix::models::qwen4_exp::prefill_sizing;
using infernix::models::qwen4_exp::TextConfig;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// A Qwen3.8-Flash-Next-shaped configuration for the term checks (the published model's classes).
TextConfig production_shape() {
    TextConfig c;
    c.num_hidden_layers = 48;
    c.attention.heads   = 24;
    c.attention.kv_heads = 4;
    c.attention.head_dim = 128;
    c.moe.experts        = 512;
    c.moe.top_k          = 10;
    c.moe.intermediate   = 512;
    c.ple.embed_dim      = 2048;
    c.ple.ngram.ngram_size = 3;
    c.ple.ngram.heads_per_ngram = 8;
    c.ple.table.row_bytes = 256; // the FP8 row volume: heads x head_width bytes per head row set
    return c;
}

// A price with a linear arena, counted so the bisection's probe count is observable.
struct LinearArena {
    std::uint64_t base = 0, per_column = 0;
    int probes         = 0;
};

std::uint64_t linear_arena(void* context, std::uint32_t columns) {
    auto* a = static_cast<LinearArena*>(context);
    ++a->probes;
    return a->base + a->per_column * columns;
}

// Oracle: the largest grid point whose price fits, by scanning every grid point.
std::uint32_t scan_chunk(const ps::ChunkGrid& grid, const ps::LoanTerms& terms, std::uint64_t budget) {
    std::uint32_t best = 0;
    for (std::uint32_t chunk = grid.step; chunk <= grid.max_chunk(); chunk += grid.step) {
        if (terms.bytes(chunk) <= budget) { best = chunk; }
    }
    return best;
}

void test_grid() {
    check((ps::ChunkGrid{.step = 128, .ceiling = 8000}.max_chunk() == 7936), "the 128 grid rounds the ceiling down");
    check((ps::ChunkGrid{.step = 896, .ceiling = 8192}.max_chunk() == 8064), "the wave grid rounds to whole waves");
    check((ps::ChunkGrid{.step = 896, .ceiling = 896}.max_chunk() == 896), "a one-wave grid yields its wave");
    check((ps::ChunkGrid{.step = 128, .ceiling = 64}.max_chunk() == 0), "a ceiling below the step yields no chunk");
    check((ps::ChunkGrid{.step = 0, .ceiling = 4096}.max_chunk() == 0), "a stepless grid yields no chunk");
}

ps::LoanTerms linear_terms(LinearArena& arena, std::uint64_t io_per_column, std::uint64_t ring) {
    ps::LoanTerms terms;
    terms.arena           = &linear_arena;
    terms.arena_context   = &arena;
    terms.io_per_column   = io_per_column;
    terms.route_log_per_column = 1920; // 48 x 10 x 4
    terms.stream_ring     = ring;
    terms.fixed           = ring != 0 ? 64ULL << 20 : 0; // one frame of slot tables
    return terms;
}

void test_bisection_matches_scan() {
    LinearArena arena{.base = 128ULL << 20, .per_column = 12 << 10};
    const ps::LoanTerms terms = linear_terms(arena, 9 << 10, 2900ULL << 20);
    for (const ps::ChunkGrid grid : {ps::ChunkGrid{.step = 128, .ceiling = 32768},
                                     ps::ChunkGrid{.step = 896, .ceiling = 16384}}) {
        // Budgets at every boundary the price crosses, plus ones below and far above the range.
        for (std::uint64_t chunk = grid.step; chunk <= grid.max_chunk(); chunk += grid.step) {
            const std::uint64_t exact = terms.bytes(chunk);
            for (const std::uint64_t budget : {exact, exact - 1, exact + 1}) {
                const std::uint32_t got = ps::largest_fitting_chunk(grid, terms, budget);
                const std::uint32_t want = scan_chunk(grid, terms, budget);
                if (got != want) {
                    check(false, "bisection differs from the scan at budget " + std::to_string(budget) +
                                     " (got " + std::to_string(got) + ", want " + std::to_string(want) + ")");
                }
                if (got != 0 && got % grid.step != 0) {
                    check(false, "a resolved chunk left the grid");
                }
            }
        }
    }
    check(ps::largest_fitting_chunk({.step = 128, .ceiling = 32768}, terms, 0) == 0,
          "no budget funds any chunk (the fixed-chunk fallback)");
    check(ps::largest_fitting_chunk({.step = 128, .ceiling = 32768}, terms, ~0ULL) == 32768,
          "an unbounded budget funds the grid's top");
    check(ps::largest_fitting_chunk({.step = 128, .ceiling = 64}, terms, ~0ULL) == 0,
          "an empty grid resolves nothing");
}

void test_probe_count() {
    // 32768/128 = 256 grid points: a bisection asks the arena at most log2(256) + 2 times.
    LinearArena arena{.base = 128ULL << 20, .per_column = 12 << 10};
    const ps::LoanTerms terms = linear_terms(arena, 9 << 10, 0);
    const std::uint32_t got = ps::largest_fitting_chunk({.step = 128, .ceiling = 32768}, terms, 2ULL << 30);
    check(got != 0 && got <= 32768, "a bounded budget resolves an in-range chunk");
    check(arena.probes <= 10, "the bisection probes the price " + std::to_string(arena.probes) + " times");
}

void test_monotone_price() {
    // The bisection's precondition: the real term builders, every grid chunk from 128 to 32768,
    // a nondecreasing price.
    const TextConfig c = production_shape();
    LinearArena arena{.base = 1ULL << 20, .per_column = 24 << 10};
    ps::LoanTerms terms = linear_terms(arena, ps::io_bytes_per_column(c), ps::stream_ring_bytes(c, 2936012));
    std::uint64_t previous = 0;
    for (std::uint32_t chunk = 128; chunk <= 32768; chunk += 128) {
        const std::uint64_t price = terms.bytes(chunk);
        check(price >= previous, "the price is not monotone at chunk " + std::to_string(chunk));
        previous = price;
    }
}

void test_term_builders() {
    const TextConfig c = production_shape();
    // io: the layout's dense arrays, 4+4+12+4+4 + 4+4+12 + 4+4 bytes per column, plus the rows.
    const std::uint64_t rows = 256ULL * (3 - 1) * 8; // row_bytes x heads, heads = (ngram_size - 1) x heads
    check(ps::io_bytes_per_column(c) == 56 + rows, "the io per-column term equals the layout's dense arrays");
    check(ps::route_log_bytes_per_column(c) == 4ULL * 10 * 48, "the route-log term equals the residency table's");
    check(ps::stream_ring_bytes(c, 2936012) == (2ULL * 512 + 1) * 2936012ULL, "the ring term is 2E + 1 records");
}

void test_ring_regime() {
    // A chunk below the stream threshold lends no ring; at it, the ring appears — the price's only
    // non-linear term, and still an increase. Above the gate the ring term is constant.
    const TextConfig c = production_shape();
    LinearArena arena{.base = 0, .per_column = 1024};
    ps::LoanTerms terms = linear_terms(arena, ps::io_bytes_per_column(c), ps::stream_ring_bytes(c, 2936012));
    const std::uint64_t narrow = terms.bytes(128);
    const std::uint64_t wide   = terms.bytes(256);
    check(wide > narrow + terms.stream_ring, "a streaming chunk prices arena, io and the ring");
    check(terms.bytes(32768) - terms.bytes(256) ==
              (32768 - 256) * (terms.io_per_column + terms.route_log_per_column + arena.per_column),
          "above the gate the ring term is constant and only the linear terms grow");
    // No ring below the gate: an unstreamed chunk is its fixed and linear terms exactly.
    ps::LoanTerms unstreamed = terms;
    unstreamed.stream_min_columns = 4096;
    check(unstreamed.bytes(2048) == unstreamed.fixed +
                                      2048ULL * (unstreamed.io_per_column + unstreamed.route_log_per_column) +
                                      arena.base + arena.per_column * 2048,
          "an unstreamed chunk prices no ring");
}

void test_frame_rounding() {
    // The price rounds up to whole frames as the lease does (review Minor 4): a budget of k frames
    // funds exactly the chunks whose rounded price is at most k frames — never under-booked.
    const std::uint64_t stride = 2936012;
    LinearArena arena{.base = 1000, .per_column = 100};
    ps::LoanTerms terms = linear_terms(arena, 56, 0);
    terms.frame_round   = stride;
    check(terms.bytes(128) % stride == 0, "the rounded price is a whole frame count");
    check(terms.bytes(128) >= 1000 + 128ULL * 156, "rounding never under-books the raw sum");
    bool monotone = true;
    for (std::uint32_t chunk = 128; chunk <= 8064; chunk += 128) {
        monotone = monotone && terms.bytes(chunk) <= terms.bytes(chunk + 128);
    }
    check(monotone, "frame rounding keeps the price monotone");
    const ps::ChunkGrid grid{.step = 128, .ceiling = 8192};
    const std::uint64_t at = terms.bytes(4096);
    const std::uint32_t fit = ps::largest_fitting_chunk(grid, terms, at);
    check(fit >= 4096 && terms.bytes(fit) == at, "a frame-priced budget funds up to its frame boundary");
    check(ps::largest_fitting_chunk(grid, terms, at - 1) < fit,
          "a budget one byte below a frame boundary falls to a cheaper chunk");
}

void test_walk_price() {
    // A walk-eligible chunk prices its span's area (review Minor 5); an unstreamed chunk prices
    // neither ring nor walk area.
    const std::uint64_t ring = 1ULL << 30, residual = 1ULL << 20, chunks = 256;
    LinearArena arena{.base = 0, .per_column = 1024};
    ps::LoanTerms terms = linear_terms(arena, 56, ring);
    terms.walk_residual = residual;
    terms.walk_chunks   = chunks;
    const std::uint64_t with = terms.bytes(256);
    check(with == ring + residual + chunks * 256ULL * 56 + terms.fixed +
                      256ULL * (56 + terms.route_log_per_column) + arena.per_column * 256,
          "a streamed chunk prices ring, walk residuals and walk io slots");
    terms.stream_min_columns = 4096;
    check(terms.bytes(2048) < with, "an unstreamed chunk prices no walk area");
}

void test_wide_staging_and_grow() {
    // The io view is decided from the lane's resolved chunk, not one call's width (Important 1):
    // a plan whose first call is narrow and whose later span calls reach the chunk still stages
    // wide, and a prompt that never exceeds the static cap never widens.
    check(ps::wide_staging_columns(896, 4096) == 0, "a chunk inside the static staging stays static");
    check(ps::wide_staging_columns(8064, 4096) == 8064,
          "the span's first call narrow, a later call at the chunk: the view is the chunk's");
    // The mirror's growth rule (Important 2): two differing resolved chunks reuse, replace or go
    // transient as the walk holds the layout or not.
    check(ps::io_grow_decision(false, 0, 8064, false) == ps::IoGrow::Allocate, "first wide prompt allocates");
    check(ps::io_grow_decision(true, 8064, 4096, false) == ps::IoGrow::Reuse,
          "a mirror held at the larger width serves a narrower prompt");
    check(ps::io_grow_decision(true, 1024, 8064, false) == ps::IoGrow::Replace, "a growing need replaces it");
    check(ps::io_grow_decision(true, 1024, 8064, true) == ps::IoGrow::Transient,
          "a walk holding the stride defers the growth to a per-unit mirror");
}

void test_equal_chunks() {
    // The §13 tail rule as pure arithmetic (Important 3): equal widths only when the tail is at
    // least the short-tail bound; zero (disabled) keeps the grid; a divisible segment needs none.
    check(ps::equal_chunk_width(7000, 4096, 1024) == 3500, "two equal chunks replace a 2904-tail split");
    check(ps::equal_chunk_width(5220, 4096, 1024) == 2610, "a 1124 tail widens to two equal calls");
    check(ps::equal_chunk_width(4608, 4096, 1024) == 0, "a 512 short tail stays its own call");
    check(ps::equal_chunk_width(8192, 4096, 1024) == 0, "a divisible segment keeps the grid");
    check(ps::equal_chunk_width(4000, 4096, 1024) == 0, "one call is already equal");
    check(ps::equal_chunk_width(7000, 4096, 0) == 0, "min_tail 0 disables the rule (the fixed route)");
    bool widths_ok = true;
    for (std::uint32_t len = 4097; len < 40000; len += 61) {
        const std::uint32_t w = ps::equal_chunk_width(len, 4096, 1024);
        widths_ok = widths_ok && (w == 0 || (w <= 4096 && (len - 1) / w + 1 == (len + 4095) / 4096));
    }
    check(widths_ok, "an equal width never raises the call count and never exceeds the chunk");
}

} // namespace

int main() {
    test_grid();
    test_bisection_matches_scan();
    test_probe_count();
    test_monotone_price();
    test_term_builders();
    test_ring_regime();
    test_frame_rounding();
    test_walk_price();
    test_wide_staging_and_grow();
    test_equal_chunks();
    if (failures == 0) { std::fprintf(stderr, "prefill sizing: all checks passed\n"); }
    return failures == 0 ? 0 : 1;
}
