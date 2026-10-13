// Throwaway capture hook for ticket #40 (branch proto/qsa-index-capture, never merged).
// Writes the pooled index plane, the index query and the block selection of selected layers as
// raw little-endian bytes, plus a per-capture meta.txt. See qsa_capture.h.

#include "ops/qsa/qsa_capture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace infernix::ops::capture {
namespace {

namespace fs = std::filesystem;

constexpr int kPlanePageTokens = 64; // kPagedKVPageSize

struct State {
    bool initialised = false;
    bool on          = false;
    fs::path dir;
    long long min_pos = 0;
    int max_captures  = 256;
    bool eager_flag   = false;

    std::mutex mutex;
    int step        = 0;
    int layer       = 0;
    int selections  = 0;
    int plane_calls = 0;
};

State& state() {
    static State s;
    return s;
}

bool capturing_stream(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return true; // unknown: do not touch the stream
    }
    return status != cudaStreamCaptureStatusNone;
}

void initialise() {
    State& s = state();
    if (s.initialised) { return; }
    s.initialised = true;
    const char* dir = std::getenv("INFERNIX_QSA_CAPTURE");
    if (dir == nullptr || *dir == '\0') { return; }
    s.dir = dir;
    if (const char* p = std::getenv("INFERNIX_QSA_CAPTURE_MIN_POS")) { s.min_pos = std::strtoll(p, nullptr, 10); }
    if (const char* n = std::getenv("INFERNIX_QSA_CAPTURE_MAX")) {
        const int value = std::atoi(n);
        if (value > 0) { s.max_captures = value; }
    }
    if (const char* e = std::getenv("INFERNIX_QSA_CAPTURE_EAGER")) {
        s.eager_flag = (*e != '\0' && *e != '0');
    }
    std::error_code ec;
    fs::create_directories(s.dir, ec);
    s.on = true;
    std::fprintf(stderr, "[qsa-capture] enabled dir=%s min_pos=%lld max=%d\n", s.dir.c_str(), s.min_pos,
                 s.max_captures);
}

void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

bool dump(const fs::path& path, const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    if (bytes != 0 && cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "[qsa-capture] copy failed for %s: %s\n", path.string().c_str(),
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    write_bytes(path, host.data(), bytes);
    return true;
}

std::vector<std::byte> copy_to_host(const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    if (bytes != 0 && cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "[qsa-capture] copy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return {};
    }
    return host;
}

void print_shape(std::ostream& out, const char* name, const Tensor& tensor, std::size_t element) {
    out << name << " ne=[";
    for (int i = 0; i < 4; ++i) { out << (i ? "," : "") << tensor.ne[i]; }
    out << "] dtype=" << static_cast<int>(tensor.dtype) << " numel=" << tensor.numel()
        << " bytes=" << static_cast<std::size_t>(tensor.numel()) * element << "\n";
}

const char* timestamp() {
    static std::string text;
    const auto now      = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buffer[32];
    std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &tm);
    text = buffer;
    return text.c_str();
}

// A call's positions and block-table row, and the plane pages its tokens occupy.
struct Call {
    std::vector<std::byte> positions;
    std::vector<std::byte> rows;
    std::vector<std::byte> table;
    std::int32_t min_pos = 0;
    std::int32_t max_pos = 0;
    std::int32_t stride  = 0;
    std::int32_t row     = 0;
    std::int32_t pages   = 0; // token pages 0..pages-1 back the positions
};

bool read_call(const QsaBatch& batch, Call& call) {
    const std::int64_t position_count = batch.positions.numel();
    if (position_count <= 0 || batch.block_tables.data == nullptr || batch.table_rows.data == nullptr) {
        return false;
    }
    call.positions = copy_to_host(batch.positions.data, static_cast<std::size_t>(position_count) * 4);
    if (call.positions.empty()) { return false; }
    const auto* values = reinterpret_cast<const std::int32_t*>(call.positions.data());
    call.max_pos = *std::max_element(values, values + position_count);
    call.min_pos = *std::min_element(values, values + position_count);

    call.stride = static_cast<std::int32_t>(batch.block_tables.ne[0]);
    call.rows   = copy_to_host(batch.table_rows.data, static_cast<std::size_t>(batch.table_rows.numel()) * 4);
    if (call.rows.empty()) { return false; }
    call.row = reinterpret_cast<const std::int32_t*>(call.rows.data())[0];
    if (call.row < 0 || call.row >= batch.block_tables.ne[1]) { return false; }
    call.table = copy_to_host(static_cast<const std::byte*>(batch.block_tables.data) +
                                  static_cast<std::size_t>(call.row) * static_cast<std::size_t>(call.stride) * 4,
                              static_cast<std::size_t>(call.stride) * 4);
    if (call.table.empty()) { return false; }
    call.pages = call.max_pos / kPlanePageTokens + 1;
    return call.pages <= call.stride;
}

// The plane pages the scan reads, in token-page order. The elastic KV pool leaves the pages above
// the sequence's allocation unmapped, so a whole-plane copy is not addressable.
bool read_plane(const Tensor& pooled_pages, const Call& call, std::int32_t slots, cudaStream_t stream,
                std::vector<std::byte>& plane) {
    const std::size_t page_bytes = static_cast<std::size_t>(slots) * kPlanePageTokens * 2;
    const auto* pages            = reinterpret_cast<const std::int32_t*>(call.table.data());
    plane.resize(static_cast<std::size_t>(call.pages) * page_bytes);
    const auto* base = static_cast<const std::byte*>(pooled_pages.data);
    for (std::int32_t j = 0; j < call.pages; ++j) {
        const std::int32_t page = pages[j];
        if (page < 0 || static_cast<std::uint32_t>(page) >= pooled_pages.ne[3]) {
            std::fprintf(stderr, "[qsa-capture] page %d out of range\n", page);
            return false;
        }
        if (cudaMemcpyAsync(plane.data() + static_cast<std::size_t>(j) * page_bytes,
                            base + static_cast<std::size_t>(page) * page_bytes, page_bytes,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::fprintf(stderr, "[qsa-capture] plane page copy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "[qsa-capture] plane sync failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

std::size_t plane_page_bytes(const QsaGeometry& geometry) {
    return static_cast<std::size_t>(geometry.index_head_dim / geometry.ratio) * kPlanePageTokens * 2;
}

void write_plane_meta(std::ostream& meta, const QsaGeometry& geometry, const Call& call,
                      const Tensor& pooled_pages, std::size_t page_bytes) {
    meta << "min_position " << call.min_pos << "\n";
    meta << "max_position " << call.max_pos << "\n";
    meta << "complete_blocks " << (static_cast<std::int64_t>(call.max_pos) + 1) / geometry.ratio << "\n";
    meta << "plane_pages_used " << call.pages << "\n";
    meta << "plane_page_bytes " << page_bytes << "\n";
    meta << "pooled_ptr " << static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pooled_pages.data))
         << "\n";
    print_shape(meta, "pooled", pooled_pages, 2);
}

} // namespace

bool enabled() {
    initialise();
    return state().on;
}

bool eager() {
    initialise();
    State& s = state();
    return s.on && s.eager_flag;
}

void step_begin(const Tensor& positions, cudaStream_t stream) {
    initialise();
    State& s = state();
    if (!s.on) { return; }
    std::lock_guard lock(s.mutex);
    if (capturing_stream(stream)) { return; }
    ++s.step;
    s.layer = 0;
    (void)positions;
}

void plane(const Tensor& pooled_pages, const QsaBatch& batch, const QsaGeometry& geometry, cudaStream_t stream) {
    initialise();
    State& s = state();
    if (!s.on) { return; }
    std::lock_guard lock(s.mutex);
    if (capturing_stream(stream)) { return; }
    if (s.plane_calls >= s.max_captures) { return; }
    Call call;
    if (!read_call(batch, call)) { return; }
    if (static_cast<long long>(call.max_pos) < s.min_pos) { return; }
    std::vector<std::byte> plane_bytes;
    if (!read_plane(pooled_pages, call, geometry.index_head_dim / geometry.ratio, stream, plane_bytes)) { return; }

    const int index = ++s.plane_calls;
    char name[64];
    std::snprintf(name, sizeof name, "step_%02d", s.step);
    fs::path dir = s.dir / "planes" / name;
    std::snprintf(name, sizeof name, "p_%03d", index);
    dir /= name;
    std::error_code ec;
    fs::create_directories(dir, ec);

    write_bytes(dir / "positions.bin", call.positions.data(), call.positions.size());
    write_bytes(dir / "table_rows.bin", call.rows.data(), call.rows.size());
    write_bytes(dir / "block_table.bin", call.table.data(), call.table.size());
    write_bytes(dir / "pooled.bin", plane_bytes.data(), plane_bytes.size());

    std::ofstream meta(dir / "meta.txt");
    meta << "captured_at " << timestamp() << "\n";
    meta << "kind plane\n";
    meta << "step " << s.step << "\n";
    meta << "plane_call " << index << "\n";
    meta << "layer_call " << s.layer++ << "\n";
    meta << "batch " << batch.batch << "\n";
    meta << "width " << batch.width << "\n";
    meta << "columns " << batch.batch * batch.width << "\n";
    meta << "positions_count " << batch.positions.numel() << "\n";
    meta << "geometry heads=" << geometry.heads << " kv_heads=" << geometry.kv_heads
         << " head_dim=" << geometry.head_dim << " index_heads=" << geometry.index_heads
         << " index_head_dim=" << geometry.index_head_dim << " rotary_dim=" << geometry.rotary_dim
         << " budget=" << geometry.budget << " ratio=" << geometry.ratio << " theta=" << geometry.theta
         << " eps=" << geometry.eps << " yarn_factor=" << geometry.yarn_factor
         << " original_positions=" << geometry.original_positions << "\n";
    meta << "plane_slots " << geometry.index_head_dim / geometry.ratio << "\n";
    meta << "plane_layout [Di/ratio, 64, 1, pages] contiguous; element(slots*page_i, slot) = "
            "slots*64*page + slots*(token&63) + slot, page = block_table[token>>6]; pooled.bin holds "
            "the used plane pages in token-page order (page j at byte offset j*plane_page_bytes)\n";
    meta << "bytes_per_token_per_layer " << (geometry.index_head_dim / geometry.ratio) * 2 << "\n";
    meta << "bytes_per_block_per_layer " << geometry.index_head_dim * 2 << "\n";
    write_plane_meta(meta, geometry, call, pooled_pages, plane_page_bytes(geometry));
    print_shape(meta, "block_tables", batch.block_tables, 4);
}

void selection(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
               const QsaGeometry& geometry, std::int32_t max_context, const void* scratch,
               const detail::QsaSelectOutput& output, cudaStream_t stream) {
    initialise();
    State& s = state();
    if (!s.on) { return; }
    std::lock_guard lock(s.mutex);
    if (capturing_stream(stream)) { return; }
    const int layer = s.layer++; // the engine's qsa_select order within the running step
    if (s.selections >= s.max_captures) { return; }

    const std::int32_t columns = batch.batch * batch.width;
    const std::int32_t top     = geometry.budget / geometry.ratio;
    const std::int32_t slots   = geometry.index_head_dim / geometry.ratio;

    Call call;
    if (!read_call(batch, call)) { return; }
    if (static_cast<long long>(call.max_pos) < s.min_pos) { return; }
    std::vector<std::byte> plane_bytes;
    if (!read_plane(pooled_pages, call, slots, stream, plane_bytes)) { return; }

    const int capture = ++s.selections;
    char name[64];
    std::snprintf(name, sizeof name, "step_%02d", s.step);
    fs::path dir = s.dir / "selections" / name;
    std::snprintf(name, sizeof name, "s_%03d", capture);
    dir /= name;
    std::error_code ec;
    fs::create_directories(dir, ec);

    write_bytes(dir / "positions.bin", call.positions.data(), call.positions.size());
    write_bytes(dir / "table_rows.bin", call.rows.data(), call.rows.size());
    write_bytes(dir / "block_table.bin", call.table.data(), call.table.size());
    write_bytes(dir / "pooled.bin", plane_bytes.data(), plane_bytes.size());
    dump(dir / "index_q.bin", index_q.data, static_cast<std::size_t>(index_q.numel()) * 2);
    dump(dir / "selected.bin", output.selected, static_cast<std::size_t>(top) * static_cast<std::size_t>(columns) * 4);
    dump(dir / "counts.bin", output.counts, static_cast<std::size_t>(columns) * 4);

    // The engine's own FP32 scores, exposed by qsa_select_scores for the selection's last group.
    const detail::QsaSelectScores scores = detail::qsa_select_scores(scratch, geometry, max_context);
    const std::int32_t group             = std::min(columns, detail::kQsaSelectGroupColumns);
    const std::int32_t group_begin       = group > 0 ? ((columns - 1) / group) * group : 0;
    const std::int32_t group_columns     = columns - group_begin;
    const std::size_t score_bytes =
        static_cast<std::size_t>(group_columns) * static_cast<std::size_t>(scores.stride) * sizeof(float);
    dump(dir / "scores.bin", scores.scores, score_bytes);

    std::ofstream meta(dir / "meta.txt");
    meta << "captured_at " << timestamp() << "\n";
    meta << "kind selection\n";
    meta << "step " << s.step << "\n";
    meta << "capture " << capture << "\n";
    meta << "layer_call " << layer << "\n";
    meta << "batch " << batch.batch << "\n";
    meta << "width " << batch.width << "\n";
    meta << "columns " << columns << "\n";
    meta << "positions_count " << batch.positions.numel() << "\n";
    meta << "max_context " << max_context << "\n";
    meta << "top_blocks " << top << "\n";
    meta << "geometry heads=" << geometry.heads << " kv_heads=" << geometry.kv_heads
         << " head_dim=" << geometry.head_dim << " index_heads=" << geometry.index_heads
         << " index_head_dim=" << geometry.index_head_dim << " rotary_dim=" << geometry.rotary_dim
         << " budget=" << geometry.budget << " ratio=" << geometry.ratio << " theta=" << geometry.theta
         << " eps=" << geometry.eps << " yarn_factor=" << geometry.yarn_factor
         << " original_positions=" << geometry.original_positions << "\n";
    meta << "plane_slots " << slots << "\n";
    meta << "plane_layout [Di/ratio, 64, 1, pages] contiguous; element(slots*page_i, slot) = "
            "slots*64*page + slots*(token&63) + slot, page = block_table[token>>6]; pooled.bin holds "
            "the used plane pages in token-page order (page j at byte offset j*plane_page_bytes)\n";
    meta << "bytes_per_token_per_layer " << slots * 2 << "\n";
    meta << "bytes_per_block_per_layer " << geometry.ratio * slots * 2 << "\n";
    write_plane_meta(meta, geometry, call, pooled_pages, plane_page_bytes(geometry));
    print_shape(meta, "index_q", index_q, 2);
    print_shape(meta, "block_tables", batch.block_tables, 4);
    print_shape(meta, "table_rows", batch.table_rows, 4);
    meta << "selected_shape [top,columns]=[" << top << "," << columns << "] i32\n";
    meta << "counts_shape [columns]=[" << columns << "] i32 (top, or -1 when the column is dense)\n";
    meta << "scores_group " << group << "\n";
    meta << "scores_group_begin " << group_begin << "\n";
    meta << "scores_group_columns " << group_columns << "\n";
    meta << "scores_stride " << scores.stride << "\n";
    meta << "scores_shape [columns_in_group,stride]=[" << group_columns << "," << scores.stride
         << "] f32; the last group is absolute columns " << group_begin << ".." << (columns - 1)
         << "; group column c holds block b at scores[c*stride+b] for b < its complete block count\n";
    meta << "scores_bytes_written " << score_bytes << "\n";
}

} // namespace infernix::ops::capture
