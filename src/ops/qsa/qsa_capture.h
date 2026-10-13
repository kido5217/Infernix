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

#include "ops/qsa/qsa_select.h"

#include <cuda_runtime.h>

namespace infernix::ops::capture {

/// True when INFERNIX_QSA_CAPTURE names a directory.
bool enabled();

/// Called once per forward step before its layers run; only numbers the captures. A step whose
/// stream is capturing a CUDA graph is skipped (a synchronous copy is illegal inside a capture).
void step_begin(const Tensor& positions, cudaStream_t stream);

/// Called by qsa_pool_keys (the writer) after the pool kernel: writes the pooled plane pages the
/// call covers, for every QSA layer including the MTP block.
void plane(const Tensor& pooled_pages, const QsaBatch& batch, const QsaGeometry& geometry,
           cudaStream_t stream);

/// Called by qsa_select after the selection: writes the index query and the selection, plus the
/// same plane, so a single capture is self-contained.
void selection(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
               const QsaGeometry& geometry, std::int32_t max_context,
               const detail::QsaSelectOutput& output, cudaStream_t stream);

} // namespace infernix::ops::capture
