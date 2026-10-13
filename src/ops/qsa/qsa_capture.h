#pragma once

// Throwaway capture hook for ticket #40 (branch proto/qsa-index-capture, never merged).
// Off unless INFERNIX_QSA_CAPTURE names an output directory. The file layout and the
// capture command are documented in docs/research/qsa-index-capture.md.
//
// Environment:
//   INFERNIX_QSA_CAPTURE          output directory (unset/empty = disabled)
//   INFERNIX_QSA_CAPTURE_MIN_POS  a call is captured only when the maximum absolute KV position
//                                  of its batch reaches this value (default 0)
//   INFERNIX_QSA_CAPTURE_MAX      stop after this many captures of each kind (default 256)
//   INFERNIX_QSA_CAPTURE_EAGER    run every CUDA-graph round eagerly instead of capturing and
//                                  replaying it, so rounds the graph owns (the MTP draft and
//                                  every later decode round) still reach the hooks (default off)

#include "ops/qsa/qsa_select.h"

#include <cuda_runtime.h>

namespace infernix::ops::capture {

/// True when INFERNIX_QSA_CAPTURE names a directory.
bool enabled();

/// True when INFERNIX_QSA_CAPTURE_EAGER is set: the qwen4_exp Program runs each decode/verify/MTP
/// round eagerly instead of capturing it into a CUDA graph. Off by default.
bool eager();

/// Called once per forward step before its layers run; only numbers the captures. A step whose
/// stream is capturing a CUDA graph is skipped (a synchronous copy is illegal inside a capture).
void step_begin(const Tensor& positions, cudaStream_t stream);

/// Called by qsa_pool_keys (the writer) after the pool kernel: writes the pooled plane pages the
/// call covers, for every QSA layer including the MTP block.
void plane(const Tensor& pooled_pages, const QsaBatch& batch, const QsaGeometry& geometry,
           cudaStream_t stream);

/// Called by qsa_select after the selection: writes the index query, the selection and the same
/// plane, so a single capture is self-contained, plus the engine's own FP32 scores for the
/// selection's last group (ops::detail::qsa_select_scores).
void selection(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
               const QsaGeometry& geometry, std::int32_t max_context, const void* scratch,
               const detail::QsaSelectOutput& output, cudaStream_t stream);

} // namespace infernix::ops::capture
