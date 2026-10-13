"""NVFP4 weight-transform prototype (issue #30, pre-registration rev 5).

Measures the primary metric of the frozen pre-registration — the end-to-end FP64 GEMM output
error ``e_arm = ||Y_arm - Y_oracle||_F / ||Y_oracle||_F`` — for six arms on 25 gating BF16
matrices (S1, the local Qwen3.8-Flash-Next BF16 GGUF) and 24 informational ModelOpt NVFP4
matrices (S2).  The specification is the rev-5 comment on GitHub issue #29
(``kido5217/Infernix``); this script implements it and records every interpretation.

Usage (inside the repo dev shell, CUDA not required):

    # 1. Capture the activation batch from the repo's own FP64 reference forward (S2 weights).
    #    This writes the MoE input x and expert-0 intermediate h for each gating layer.
    python3 -m tools.flash_next.reference \
        --model /home/kido/trash/ai/models/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
        --layers 43 \
        --tokens 760,6511,314,9338,369,279,7526,3177,303,279,3046,321,264,3478,7277,3990 \
        --record-taps 0,6,12,18,24,30,36,42 --out capture.npz

    # 2. Correctness anchor: C2 with the source divisor must reproduce S2's stored scale words.
    python3 -m tools.proto.nvfp4_transform_prototype --self-check --out selfcheck.json

    # 3. The measurement.
    python3 -m tools.proto.nvfp4_transform_prototype --activations capture.npz --out report.json

    # 4. (optional) Regenerate the pre-fix DualQuant sensitivity report.  ``--bcd-variant released``
    #    swaps G3/X1's DualQuant for the released implementation's reciprocal beta init and its
    #    2^-6 E4M3 projection floor (the two defects corrected before the reported run), which is
    #    what the committed ``…report.bcd-reciprocal.json`` was produced with.
    python3 -m tools.proto.nvfp4_transform_prototype --activations capture.npz \
        --bcd-variant released --out report.bcd-reciprocal.json

S1 is read from the fixed path in ``S1_ROOT`` below; both roots are recorded in the report.

The script reuses the repo's own codec (``tools/convert/quantization/nvfp4.py``,
``tools/artifact/codecs/nvfp4_expert.py``, ``tools/flash_next/a4_reference.py``); it
re-implements the OSFP4 optimizer and the DualQuant BCD from their papers (nothing is vendored).

Report self-description.  ``environment`` carries the interpreter, the torch/numpy versions and the
git sha of this checkout; ``reproduction`` carries the exact capture, self-check and report
commands, derived from this process's own argv and executable plus the module's frozen S2 root,
gating layers and prompt tokens (never a hand-typed string), as both argv lists and shell lines.
Every arm record in ``matrices`` carries the actual per-matrix ``weight_divisor`` (binary32) and
``weight_target`` as well as the ``activation_divisor`` (binary32) and ``activation_target``, so the
divisor choice is auditable per matrix and not only named as a rule.  G3/X1 additionally carry
their existing ``fold_relative_l2`` plus a new ``e_no_fold`` (the same arm scored with the fold
disabled: alpha_inv applied exactly as a separate per-output-row weight scale instead of being
folded, and re-rounded, into the stored E4M3 block scale); their difference isolates the fold's
contribution, summarized in the top-level ``g3_fold_diagnostic``.  The top-level ``bcd_variant``
records which DualQuant variant produced G3/X1.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tools.artifact.codecs.nvfp4_expert import dequantize_nvfp4_mul
from tools.convert.quantization.nvfp4 import (
    quantize_nvfp4,
    quantize_nvfp4_mse,
    weight_divisor,
)
from tools.flash_next.a4_reference import a4_dequantize, a4_quantize

# ---------------------------------------------------------------------------- frozen constants

S1_ROOT = Path(
    "/home/kido/trash/ai/models/hf/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF"
    "/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/BF16"
)
S2_ROOT = Path(
    "/home/kido/trash/ai/models/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4"
    "/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47"
)
GATING_LAYERS = (0, 6, 12, 18, 24, 30, 36, 42)
# The committed prompt set: "The capital of France is the largest city in the country and a major
# European center" (Qwen byte-level BPE; the S2 checkpoint's vocab.json).
PROMPT_TEXT = "The capital of France is the largest city in the country and a major European center"
PROMPT_TOKENS = (760, 6511, 314, 9338, 369, 279, 7526, 3177, 303, 279, 3046, 321, 264, 3478, 7277, 3990)
SEED = 0
ABSMAX_TARGET = 6.0  # C1 / X1 weight divisor target (repo absmax rule)
MSE_TARGET = 4.0     # C2 / G1 / G2 / G3 weight divisor target (repo MSE rule)
E4M3_MAX = 448.0
E4M3_MIN_POS = 2.0 ** -9  # smallest positive E4M3 subnormal
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
F32 = torch.float32
F64 = torch.float64

ARM_ORDER = ("C1", "C2", "G1", "G2", "G3", "X1")
GATING_ARMS = ("G1", "G2", "G3")  # candidates that can clear the bar


# ---------------------------------------------------------------------------- GGUF (S1) reader


class GgufShards:
    """Minimal GGUF v3 header reader; BF16/F32 tensors only.  No new dependency."""

    _SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
               5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
               12: ("<d", 8)}
    _ITEM_BYTES = {0: 4, 1: 2, 30: 2}  # F32, F16, BF16

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tensors: dict[str, tuple[Path, int, int, tuple[int, ...], int]] = {}
        for path in sorted(self.root.glob("*.gguf")):
            self._read_header(path)

    def _read_string(self, f) -> str:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def _skip(self, f, vt: int) -> None:
        if vt == 8:
            self._read_string(f)
        elif vt == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (cnt,) = struct.unpack("<Q", f.read(8))
            for _ in range(cnt):
                self._skip(f, et)
        else:
            f.read(self._SCALAR[vt][1])

    def _value(self, f, vt: int):
        if vt == 8:
            return self._read_string(f)
        fmt, size = self._SCALAR[vt]
        return struct.unpack(fmt, f.read(size))[0]

    def _read_header(self, path: Path) -> None:
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF", path
            (version,) = struct.unpack("<I", f.read(4))
            assert version == 3, (path, version)
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))
            alignment = 32
            for _ in range(kv_count):
                key = self._read_string(f)
                (vt,) = struct.unpack("<I", f.read(4))
                if vt == 9:
                    (et,) = struct.unpack("<I", f.read(4))
                    (cnt,) = struct.unpack("<Q", f.read(8))
                    for _ in range(cnt):
                        self._skip(f, et)
                else:
                    value = self._value(f, vt)
                    if key == "general.alignment":
                        alignment = int(value)
            entries = []
            for _ in range(tensor_count):
                name = self._read_string(f)
                (nd,) = struct.unpack("<I", f.read(4))
                ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
                (gtype,) = struct.unpack("<I", f.read(4))
                (offset,) = struct.unpack("<Q", f.read(8))
                entries.append((name, ne, gtype, offset))
            header_end = f.tell()
            data_offset = (header_end + alignment - 1) // alignment * alignment
        for name, ne, gtype, offset in entries:
            self.tensors[name] = (path, data_offset, offset, ne, gtype)

    def expert_matrix_f64(self, name: str, expert: int = 0) -> torch.Tensor:
        """Read one expert's ``[N, K]`` slice (GGUF ``ne`` order: expert index last), exact BF16."""
        path, data_offset, offset, ne, gtype = self.tensors[name]
        if gtype not in (30, 0):
            raise ValueError(f"{name}: expected BF16/F32, got ggml type {gtype}")
        item = self._ITEM_BYTES[gtype]
        ndims = len(ne)
        if ndims == 3:
            per_expert = ne[0] * ne[1]
            start = expert * per_expert
        else:
            per_expert = ne[0] * ne[1]
            start = 0
        with open(path, "rb") as f:
            f.seek(data_offset + offset + start * item)
            raw = f.read(per_expert * item)
        if gtype == 30:
            u16 = np.frombuffer(raw, dtype="<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        else:
            f32 = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        # ne[0] is the innermost (K) axis, ne[1] the N axis; experts are the outermost block.
        matrix = f32.reshape(ne[1], ne[0])
        return torch.from_numpy(np.ascontiguousarray(matrix))


# ---------------------------------------------------------------------------- E4M3 helpers


def e4m3_words(values: torch.Tensor) -> torch.Tensor:
    """FP32 -> E4M3FN words, round-to-nearest-even with saturation (as ``_e4m3_rn_satfinite``)."""
    return torch.clamp(values.to(F32), max=E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_values(words: torch.Tensor) -> torch.Tensor:
    return words.view(torch.float8_e4m3fn).to(F32)


def e4m3_grid_values() -> torch.Tensor:
    """Every positive finite E4M3 value, ascending (words 0x01..0x7E)."""
    words = torch.arange(1, 0x7F, dtype=torch.uint8)
    return e4m3_values(words)


def pack_e2m1_codes(values: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 codes for ``values [N, K]`` at per-16 ``step [N, K/16]`` (even K in low nibble)."""

    n, k = values.shape
    groups = values.reshape(n, k // 16, 16).to(F32)
    normalized = torch.clamp(groups.abs() / step[..., None], max=6.0)
    midpoints = torch.tensor(E2M1_MIDPOINTS)
    magnitudes = torch.tensor(E2M1_MAGNITUDES)
    code = torch.bucketize(normalized, midpoints)
    signed = code | ((groups < 0) & (code > 0)).to(torch.int32) << 3
    words = signed.to(torch.uint8).reshape(n, k)
    return (words[:, 0::2] | (words[:, 1::2] << 4)).contiguous()


# ---------------------------------------------------------------------------- quantizer helpers


def divisor_for(amax: float, target: float) -> float:
    return weight_divisor(amax, target)


def f32_round(value: float) -> float:
    """Round one value to its binary32 representation (the divisor columns of pre-reg table 1)."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantize_weight(W: torch.Tensor, divisor: float, rule: str):
    """Quantize ``W [N, K]`` (FP64) with the repo codec; return the word struct and FP64 dequant."""

    W32 = W.to(F32).contiguous()
    words = quantize_nvfp4(W32, divisor) if rule == "absmax" else quantize_nvfp4_mse(W32, divisor)
    represented = dequantize_nvfp4_mul(words.codes, words.scales, 1.0) / float(divisor)
    return words, represented


def fold_row_scale(words, row_factor: torch.Tensor):
    """Fold a per-output-row factor (``alpha_inv``) into the stored E4M3 scale words."""

    decoded = e4m3_values(words.scales)  # [N, K/16]
    folded = e4m3_words(decoded * row_factor.to(F32)[:, None])
    return folded


def dequantize_words(codes: torch.Tensor, scales: torch.Tensor, divisor: float) -> torch.Tensor:
    return dequantize_nvfp4_mul(codes, scales, 1.0) / float(divisor)


# ---------------------------------------------------------------------------- activations


def activation_a4(values_f64: torch.Tensor, divisor: float) -> torch.Tensor:
    """Group-16 A4 quantize -> dequantize of ``values [K, T]`` (the repo's own A4 rule)."""

    rows = values_f64.transpose(0, 1).contiguous()  # [T, K], groups along the last axis
    bf16 = rows.to(F32).to(torch.bfloat16)
    input_scale = torch.tensor(1.0 / float(divisor), dtype=F32)
    codes, scales = a4_quantize(bf16, input_scale)
    return a4_dequantize(codes, scales, input_scale).transpose(0, 1).to(F64).contiguous()


def activation_divisor(values_f64: torch.Tensor, target: float) -> float:
    """The repo's divisor rule applied to the activation batch (see ``DEVIATIONS``)."""

    amax = float(values_f64.abs().max())
    return divisor_for(amax, target)


def round_bf16(values: torch.Tensor) -> torch.Tensor:
    return values.to(F32).to(torch.bfloat16).to(F64)


def sha256_f64(values: torch.Tensor) -> str:
    array = np.ascontiguousarray(values.detach().to(F64).numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


# ---------------------------------------------------------------------------- OSFP4 (2610.08231)
#
# Re-implemented from the paper (Eq 3-7, 12, 16, 22-25, 38-45, App. B/C).  The phi table is the
# smoothed relative FP4 error phi(2^t) = integral_{t-1}^{t} (1 - 2^-v Q_FP4(2^v))^2 dv.

_LN2 = math.log(2.0)
_FP4_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_LOG2_BOUNDS = np.log2(0.5 * (_FP4_POS[:-1] + _FP4_POS[1:]))
LUT_LOG2_MIN, LUT_LOG2_MAX, LUT_STEP = -4.0, 8.0, 2.0 ** -12
_PHI_TABLE: torch.Tensor | None = None


def _g_antiderivative(t: np.ndarray, quantized: float) -> np.ndarray:
    p1 = np.exp2(-t)
    return t + (2.0 * quantized / _LN2) * p1 - (quantized * quantized / (2.0 * _LN2)) * (p1 * p1)


def _integrate_g(start: float, end: float) -> float:
    result = 0.0
    edges = np.concatenate(([-np.inf], _LOG2_BOUNDS, [np.inf]))
    for region, quantized in enumerate(_FP4_POS):
        low = max(start, float(edges[region]))
        high = min(end, float(edges[region + 1]))
        if high <= low:
            continue
        if quantized == 0.0:
            result += high - low
        else:
            result += _g_antiderivative(high, quantized) - _g_antiderivative(low, quantized)
    return result


def phi_table() -> torch.Tensor:
    global _PHI_TABLE
    if _PHI_TABLE is None:
        count = int(round((LUT_LOG2_MAX - LUT_LOG2_MIN) / LUT_STEP)) + 1
        grid = LUT_LOG2_MIN + LUT_STEP * np.arange(count, dtype=np.float64)
        values = np.array([_integrate_g(t - 1.0, t) for t in grid], dtype=np.float64)
        _PHI_TABLE = torch.from_numpy(values).to(F32)
    return _PHI_TABLE


def phi_lut(value: torch.Tensor) -> torch.Tensor:
    table = phi_table()
    starts = table[:-1].contiguous()
    slopes = (table[1:] - table[:-1]).contiguous()
    entries = starts.shape[0] + 1
    magnitude = value.abs().clamp(min=2.0 ** LUT_LOG2_MIN)
    pos = (torch.log2(magnitude) - LUT_LOG2_MIN) / LUT_STEP
    pos = pos.clamp(min=0.0, max=float(entries - 2))
    index = pos.detach().to(torch.int32)
    frac = pos - index.to(pos.dtype)
    flat = index.reshape(-1)
    cell_start = torch.index_select(starts, 0, flat).view_as(frac)
    cell_slope = torch.index_select(slopes, 0, flat).view_as(frac)
    return cell_start + frac * cell_slope


def _weight_loss(wg, alpha, gamma_w, coeff_w):
    groups, rows, _ = wg.shape
    phi = phi_lut(wg / (alpha.view(groups, 1, -1) * gamma_w))
    return (coeff_w * phi).sum(dim=(1, 2)) / rows


def _activation_loss(ax, alpha, gamma_x, coeff_x, rows: int):
    groups = ax.shape[0]
    samples = ax.shape[2]
    phi = phi_lut(alpha.view(groups, -1, 1) * ax / gamma_x)
    return (coeff_x * phi).sum(dim=(1, 2)) / (rows * samples)


def osfp4_optimize(wg, ax, metric, *, joint: bool, optimize_alpha: bool = True,
                   steps: int = 80, lr: float = 0.12):
    """Joint continuous optimization of A and the FP4 block scales (paper App. B).

    ``wg`` is ``[G, R, 16]``; ``ax`` is the activation groups ``[G, 16, S]`` (used only for the
    joint W4A4 loss); ``metric`` is ``diag(Sigma_X)`` shaped ``[G, 16]``.
    """

    groups, rows, _ = wg.shape
    eps = 1e-12
    alpha = torch.ones(groups, 16, dtype=F32)
    gamma_w_init = (wg.abs().amax(dim=2, keepdim=True) / 6.0).clamp(min=eps)
    log_alpha = alpha.log().clone().requires_grad_(True) if optimize_alpha else None
    log_gamma_w = gamma_w_init.log().clone().requires_grad_(True)
    log_gamma_x = None
    if joint:
        log_gamma_x = (ax.abs().amax(dim=1, keepdim=True) / 6.0).clamp(min=eps).log().clone().requires_grad_(True)
    coefficients_w = wg.square() * metric[:, None, :]
    coefficients_x = None
    if joint:
        column_energy = wg.square().sum(dim=1)  # [G, 16]
        coefficients_x = ax.square() * column_energy[:, :, None]
    parameters = [p for p in (log_alpha, log_gamma_w, log_gamma_x) if p is not None]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        current_alpha = log_alpha.exp() if log_alpha is not None else alpha
        loss = _weight_loss(wg, current_alpha, log_gamma_w.exp(), coefficients_w).sum()
        if joint:
            loss = loss + _activation_loss(ax, current_alpha, log_gamma_x.exp(), coefficients_x,
                                           rows).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        if log_alpha is not None:
            shift = log_gamma_w.mean(dim=(1, 2), keepdim=True)
            log_alpha.add_(shift.squeeze(-1))
            log_gamma_w.sub_(shift)
            if log_gamma_x is not None:
                log_gamma_x.add_(shift)
    result_alpha = log_alpha.exp().detach() if log_alpha is not None else alpha
    result_gamma_x = log_gamma_x.exp().detach() if log_gamma_x is not None else None
    return result_alpha, log_gamma_w.exp().detach(), result_gamma_x


def osfp4_select_scale_words(wg, divisor: float, gamma_w, metric) -> torch.Tensor:
    """The paper's fine E4M3 block-scale selection (Eq 45) realized in the repo's representation.

    The repo stores one positive E4M3 word ``s`` per 16-group and a per-matrix FP32 divisor ``d``,
    so the effective step is ``s/d``; candidates are the 126 E4M3 words whose effective step lies
    in ``[gamma/1.2, gamma/0.3]`` (App. C).  Returns ``[G, R]`` uint8 E4M3 words.
    """

    groups, rows, _ = wg.shape
    grid_values = e4m3_grid_values()          # 126 positive finite E4M3 values
    grid_words = torch.arange(1, 0x7F, dtype=torch.int32)
    steps = grid_values / float(divisor)
    lower = (gamma_w / 1.2).squeeze(-1)
    upper = (gamma_w / 0.3).squeeze(-1)
    midpoints = torch.tensor(E2M1_MIDPOINTS)
    magnitudes = torch.tensor(E2M1_MAGNITUDES)
    best_error = torch.full((groups, rows), float("inf"))
    best_word = torch.zeros((groups, rows), dtype=torch.uint8)
    for index in range(grid_values.numel()):
        step = float(steps[index])
        valid = (step >= lower) & (step <= upper)
        if not bool(valid.any()):
            continue
        normalized = (wg.abs() / step).clamp(max=6.0)
        code = torch.bucketize(normalized, midpoints)
        reconstructed = torch.sign(wg) * magnitudes[code] * step
        error = ((wg - reconstructed).square() * metric[:, None, :]).sum(dim=-1)
        error = torch.where(valid, error, torch.full_like(error, float("inf")))
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_word = torch.where(better, torch.full_like(best_word, int(grid_words[index])), best_word)
    empty = torch.isinf(best_error)
    if bool(empty.any()):
        below = empty & (upper < steps[0])
        above = empty & (lower > steps[-1])
        best_word = torch.where(below, torch.full_like(best_word, int(grid_words[0])), best_word)
        best_word = torch.where(above, torch.full_like(best_word, int(grid_words[-1])), best_word)
        remaining = empty & ~below & ~above
        if bool(remaining.any()):
            distance = (gamma_w.squeeze(-1).unsqueeze(-1) - steps).abs()
            nearest = distance.argmin(dim=-1).to(torch.uint8)
            best_word = torch.where(remaining, (nearest.to(torch.int32) + 1).to(torch.uint8), best_word)
    return best_word


# ---------------------------------------------------------------------------- DualQuant (2610.13767)
#
# The paper's dual-scale BCD (Eq 2-5, Algorithm 1): W ~ Diag(alpha)^-1 Q(Diag(alpha) W Diag(beta))
# Diag(beta)^-1, with per-coordinate closed-form updates and projection of the inverse scales onto
# their E4M3 admissible set (Sec. 4.2).  Q is the format-native per-16 NVFP4 quantizer (d = 1:
# per-group absmax-to-6 E4M3 scale + E2M1 code).


def dualquant_bcd(W64: torch.Tensor, *, iterations: int = 15, variant: str = "paper"):
    """DualQuant's dual-scale BCD (paper Algorithm 1).

    ``variant`` selects the beta initialization and the E4M3 projection floor:

    - ``paper`` — the paper's ``beta_0,j = sum_i |W_ij|`` and a floor at the smallest positive
      E4M3 value ``2^-9`` (the admissible set includes E4M3 subnormals).  This is the reported run.
    - ``released`` — the released implementation's reciprocal init ``beta = 1 / sum_i |W_ij|`` and
      a floor at the smallest E4M3 *normal* ``2^-6``.  These are the two pre-fix defects the
      prototype corrected; this variant exists only to regenerate the committed
      ``…report.bcd-reciprocal.json`` sensitivity report.  It is not a candidate arm.
    """

    if variant not in ("paper", "released"):
        raise ValueError(f"unknown DualQuant variant {variant!r}")
    W = W64.to(F32)
    eps = 1e-8
    if variant == "paper":
        floor = 2.0 ** -9
        beta = W.abs().sum(dim=0).clamp(min=eps)              # paper init: beta_0 = per-column l1 norm
    else:
        floor = 2.0 ** -6
        beta = 1.0 / W.abs().sum(dim=0).clamp(min=eps)        # released init: reciprocal l1 norm
    beta_inv = 1.0 / beta
    row_scale = (W * beta[None, :]).abs().amax(dim=1).clamp(min=eps) / 6.0
    alpha_inv = row_scale                                   # paper init: format-native absmax row scale
    alpha = 1.0 / alpha_inv
    for _ in range(iterations):
        scaled = alpha[:, None] * W * beta[None, :]
        words = quantize_nvfp4(scaled, 1.0)
        q_hat = dequantize_nvfp4_mul(words.codes, words.scales, 1.0).to(F32)
        residual = q_hat * beta_inv[None, :]
        alpha_inv = (W * residual).sum(dim=1) / (residual.square().sum(dim=1) + eps)
        alpha_inv = e4m3_values(e4m3_words(alpha_inv.clamp(min=floor)))
        alpha = 1.0 / alpha_inv
        scaled = alpha[:, None] * W * beta[None, :]
        words = quantize_nvfp4(scaled, 1.0)
        q_hat = dequantize_nvfp4_mul(words.codes, words.scales, 1.0).to(F32)
        column = q_hat * alpha_inv[:, None]
        beta_inv = (W * column).sum(dim=0) / (column.square().sum(dim=0) + eps)
        beta_inv = e4m3_values(e4m3_words(beta_inv.clamp(min=floor)))
        beta = 1.0 / beta_inv
    return alpha.detach(), beta.detach(), alpha_inv.detach(), beta_inv.detach()


# ---------------------------------------------------------------------------- matrices


@dataclasses.dataclass
class Matrix:
    matrix_id: str
    source: str
    layer: int
    kind: str
    n: int
    k: int
    oracle: torch.Tensor        # [N, K] FP64
    activation: torch.Tensor    # [K, T] FP64
    sha256: str
    input_scale: float | None = None


_S1_TENSOR = {"gate": "ffn_gate_exps.weight", "up": "ffn_up_exps.weight",
              "down": "ffn_down_exps.weight"}


def activation_for(activations: dict, layer: int, kind: str) -> torch.Tensor:
    if kind == "down":
        return activations["expert0_intermediate"][layer].to(F64).transpose(0, 1).contiguous()
    return activations["moe_input"][layer].to(F64).transpose(0, 1).contiguous()


def load_s1(shards: GgufShards, activations: dict) -> list[Matrix]:
    matrices = []
    for layer in GATING_LAYERS:
        for kind in ("gate", "up", "down"):
            name = f"blk.{layer}.{_S1_TENSOR[kind]}"
            weight = shards.expert_matrix_f64(name, 0)
            matrices.append(Matrix(f"S1/L{layer}/{kind}", "S1", layer, kind,
                                   weight.shape[0], weight.shape[1], weight,
                                   activation_for(activations, layer, kind), sha256_f64(weight)))
    weight = shards.expert_matrix_f64("blk.0.ffn_gate_shexp.weight", 0)
    matrices.append(Matrix("S1/L0/shexp", "S1", 0, "shexp", weight.shape[0], weight.shape[1],
                           weight, activation_for(activations, 0, "shexp"), sha256_f64(weight)))
    return matrices


def load_s2(store, activations: dict) -> list[Matrix]:
    from tools.convert.sources.modelopt import nvfp4_matrix_words

    matrices = []
    for layer in GATING_LAYERS:
        for kind, proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            n, k = (640, 2560) if kind != "down" else (2560, 640)
            leaf = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}"
            words = nvfp4_matrix_words(store, leaf, (n, k))
            multiplier = float(words.weight_scale_2)
            weight = dequantize_nvfp4_mul(words.codes, words.scales, multiplier)
            matrices.append(Matrix(f"S2/L{layer}/{kind}", "S2", layer, kind, n, k, weight,
                                   activation_for(activations, layer, kind), sha256_f64(weight),
                                   float(words.input_scale)))
    return matrices


# ---------------------------------------------------------------------------- arms


def _secondary(W_eff2: torch.Tensor, W_oracle: torch.Tensor) -> tuple[float, float]:
    n, k = W_oracle.shape
    a = W_oracle.reshape(n, k // 16, 16)
    b = W_eff2.reshape(n, k // 16, 16)
    numerator = (b - a).square().sum(dim=-1).sqrt()
    denominator = a.square().sum(dim=-1).sqrt()
    relative = torch.where(denominator > 0, numerator / denominator.clamp(min=1e-300),
                           torch.zeros_like(numerator))
    return float(relative.mean()), float(relative.max())


def _a4_pair(activation: torch.Tensor, target: float) -> tuple[torch.Tensor, float]:
    divisor = activation_divisor(activation, target)
    return activation_a4(activation, divisor), divisor


def _osfp4_alpha(weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    k = weight.shape[1]
    groups = k // 16
    wg = weight.to(F32).reshape(weight.shape[0], groups, 16).permute(1, 0, 2).contiguous()
    ax = activation.to(F32).reshape(groups, 16, activation.shape[1]).contiguous()
    metric = activation.to(F32).square().mean(dim=1).reshape(groups, 16)
    alpha, _, _ = osfp4_optimize(wg, ax, metric, joint=True)
    return alpha.reshape(k)


def _osfp4_selection(weight: torch.Tensor, activation: torch.Tensor):
    groups = weight.shape[1] // 16
    wg = weight.to(F32).reshape(weight.shape[0], groups, 16).permute(1, 0, 2).contiguous()
    metric = activation.to(F32).square().mean(dim=1).reshape(groups, 16)
    _, gamma_w, _ = osfp4_optimize(wg, None, metric, joint=False, optimize_alpha=False)
    divisor = divisor_for(float(weight.abs().max()), MSE_TARGET)
    scales = osfp4_select_scale_words(wg, divisor, gamma_w, metric).t().contiguous()
    return divisor, scales


def run_arms(weight: torch.Tensor, activation: torch.Tensor, *,
             bcd_variant: str = "paper") -> dict[str, dict]:
    """Evaluate C1, C2, G1, G2, G3, X1 on one matrix; return each arm's primary and secondary.

    Every arm record carries the *actual* per-matrix weight and activation divisors used (their
    binary32 stored values) together with the target each was derived from, so the divisor choice
    is auditable per matrix and not only named as a rule.  ``bcd_variant`` selects G3/X1's
    DualQuant variant (see :func:`dualquant_bcd`); it does not add an arm.
    """

    oracle = weight @ activation
    norm = float(oracle.norm())
    results: dict[str, dict] = {}

    def finish(name: str, y_arm: torch.Tensor, weight_effective: torch.Tensor, *,
               weight_divisor: float, weight_target: float, activation_divisor: float,
               activation_target: float, extra=None) -> None:
        mean, worst = _secondary(weight_effective, weight)
        entry = {"e": float((y_arm - oracle).norm() / norm),
                 "secondary_mean": mean, "secondary_worst": worst,
                 "weight_divisor": f32_round(float(weight_divisor)),
                 "weight_target": float(weight_target),
                 "activation_divisor": f32_round(float(activation_divisor)),
                 "activation_target": float(activation_target)}
        if extra:
            entry.update(extra)
        results[name] = entry

    # C1 -- repo absmax (target 6)
    divisor = divisor_for(float(weight.abs().max()), ABSMAX_TARGET)
    _, represented = quantize_weight(weight, divisor, "absmax")
    activation_q, act_divisor = _a4_pair(activation, ABSMAX_TARGET)
    finish("C1", represented @ activation_q, represented,
           weight_divisor=divisor, weight_target=ABSMAX_TARGET,
           activation_divisor=act_divisor, activation_target=ABSMAX_TARGET)

    # C2 -- repo MSE (target 4)
    divisor = divisor_for(float(weight.abs().max()), MSE_TARGET)
    _, represented = quantize_weight(weight, divisor, "mse")
    activation_q, act_divisor = _a4_pair(activation, MSE_TARGET)
    finish("C2", represented @ activation_q, represented,
           weight_divisor=divisor, weight_target=MSE_TARGET,
           activation_divisor=act_divisor, activation_target=MSE_TARGET)

    # G1 -- OSFP4 diagonal smoothing A (W A^-1 quantized by C2's MSE rule; A X on activations)
    alpha = _osfp4_alpha(weight, activation)
    alpha_bf16 = alpha.to(torch.bfloat16).to(F32)
    alpha64 = alpha_bf16.to(F64)
    transformed = weight / alpha64[None, :]
    divisor = divisor_for(float(transformed.abs().max()), MSE_TARGET)
    _, represented = quantize_weight(transformed, divisor, "mse")
    activation_q, act_divisor = _a4_pair(round_bf16(activation * alpha64[:, None]), MSE_TARGET)
    finish("G1", represented @ activation_q, represented * alpha64[None, :],
           weight_divisor=divisor, weight_target=MSE_TARGET,
           activation_divisor=act_divisor, activation_target=MSE_TARGET)

    # G2 -- OSFP4 calibrated E4M3 block-scale selection, no smoothing
    divisor, scales = _osfp4_selection(weight, activation)
    step = (e4m3_values(scales) / divisor).contiguous()
    codes = pack_e2m1_codes(weight.to(F32), step)
    represented = dequantize_words(codes, scales, divisor)
    activation_q, act_divisor = _a4_pair(activation, MSE_TARGET)
    finish("G2", represented @ activation_q, represented,
           weight_divisor=divisor, weight_target=MSE_TARGET,
           activation_divisor=act_divisor, activation_target=MSE_TARGET)

    # G3 -- DualQuant alpha/beta BCD on top of C2's rule
    alpha, _, alpha_inv, beta_inv = dualquant_bcd(weight, variant=bcd_variant)
    alpha64 = alpha.to(F64)
    beta = (1.0 / beta_inv).to(F64)
    beta_inv64 = beta_inv.to(F64)
    transformed = alpha64[:, None] * weight * beta[None, :]
    divisor = divisor_for(float(transformed.abs().max()), MSE_TARGET)
    words, _ = quantize_weight(transformed, divisor, "mse")
    folded = fold_row_scale(words, alpha_inv)
    represented = dequantize_words(words.codes, folded, divisor)
    activation_q, act_divisor = _a4_pair(round_bf16(activation * beta_inv64[:, None]), MSE_TARGET)
    unfolded = dequantize_words(words.codes, words.scales, divisor) * alpha_inv.to(F64)[:, None]
    finish("G3", represented @ activation_q, represented * beta_inv64[None, :],
           weight_divisor=divisor, weight_target=MSE_TARGET,
           activation_divisor=act_divisor, activation_target=MSE_TARGET,
           extra={
               "bcd_variant": bcd_variant,
               "fold_word_changed_fraction": float((folded != words.scales).to(F64).mean()),
               "fold_relative_l2": float((represented - unfolded).norm() / unfolded.norm()),
               "e_no_fold": float((unfolded @ activation_q - oracle).norm() / norm),
           })

    # X1 -- G3's transform on C1's absmax rule (exploratory, non-gating)
    divisor = divisor_for(float(transformed.abs().max()), ABSMAX_TARGET)
    words, _ = quantize_weight(transformed, divisor, "absmax")
    folded = fold_row_scale(words, alpha_inv)
    represented = dequantize_words(words.codes, folded, divisor)
    activation_q, act_divisor = _a4_pair(round_bf16(activation * beta_inv64[:, None]), ABSMAX_TARGET)
    unfolded = dequantize_words(words.codes, words.scales, divisor) * alpha_inv.to(F64)[:, None]
    finish("X1", represented @ activation_q, represented * beta_inv64[None, :],
           weight_divisor=divisor, weight_target=ABSMAX_TARGET,
           activation_divisor=act_divisor, activation_target=ABSMAX_TARGET,
           extra={
               "bcd_variant": bcd_variant,
               "fold_word_changed_fraction": float((folded != words.scales).to(F64).mean()),
               "fold_relative_l2": float((represented - unfolded).norm() / unfolded.norm()),
               "e_no_fold": float((unfolded @ activation_q - oracle).norm() / norm),
           })

    return results



# ---------------------------------------------------------------------------- bar


def arm_summary(rows: list[dict]) -> dict:
    """Per-arm median / range / worst of the primary metric and the win count (pre-reg 4)."""

    if not rows:
        return {}
    summary = {}
    for arm in ARM_ORDER:
        values = [row["arms"][arm]["e"] for row in rows]
        worst = int(np.argmax(values))
        wins = sum(
            1 for row in rows
            if row["arms"][arm]["e"] < min(row["arms"]["C1"]["e"], row["arms"]["C2"]["e"])
        )
        summary[arm] = {"matrices": len(rows), "median_e": float(np.median(values)),
                        "min_e": float(min(values)), "max_e": float(max(values)),
                        "worst_matrix": rows[worst]["matrix_id"], "wins_vs_best_control": wins}
    return summary


def bar_verdict(rows: list[dict], arm: str) -> dict:
    if not rows:
        return {"wins": 0, "matrices": 0, "median_relative_reduction": float("nan"),
                "worst_ratio": float("nan"), "worst_matrix": None, "clears": False}
    wins = 0
    reductions = []
    ratios = []
    worst_matrix = None
    worst_ratio = -1.0
    for row in rows:
        best_control = min(row["arms"]["C1"]["e"], row["arms"]["C2"]["e"])
        e_arm = row["arms"][arm]["e"]
        if e_arm < best_control:
            wins += 1
        reductions.append((best_control - e_arm) / best_control)
        ratio = e_arm / best_control
        ratios.append(ratio)
        if ratio > worst_ratio:
            worst_ratio, worst_matrix = ratio, row["matrix_id"]
    median = float(np.median(reductions))
    return {"wins": wins, "matrices": len(rows), "median_relative_reduction": median,
            "worst_ratio": worst_ratio, "worst_matrix": worst_matrix,
            "clears": bool(wins >= 22 and median >= 0.10 and worst_ratio <= 1.02)}


# ---------------------------------------------------------------------------- self-check


def self_check(store) -> dict:
    """C2 with the SOURCE divisor must reproduce S2's stored scale words (pre-registration 1)."""

    from tools.convert.sources.modelopt import nvfp4_matrix_words

    rows = []
    print("== self-check: repo MSE at the source divisor 1/weight_scale_2 (S2) ==")
    for layer in GATING_LAYERS:
        for kind, proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            n, k = (640, 2560) if kind != "down" else (2560, 640)
            leaf = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}"
            words = nvfp4_matrix_words(store, leaf, (n, k))
            multiplier = float(words.weight_scale_2)
            source_divisor = f32_round(1.0 / multiplier)
            weight = dequantize_nvfp4_mul(words.codes, words.scales, multiplier)
            amax = float(weight.abs().max())
            d6 = weight_divisor(amax, 6.0)
            requantized = quantize_nvfp4_mse(weight.to(F32), source_divisor)
            total = int(words.scales.numel())
            match = int((requantized.scales == words.scales).sum())
            row = {"matrix": f"S2/L{layer}/{kind}", "match": match, "total": total,
                   "decoded_amax": amax, "source_divisor": source_divisor,
                   "weight_divisor6": d6, "divisor6_equals_source": bool(d6 == source_divisor)}
            rows.append(row)
            print(f"  L{layer:2d} {kind:5s} {match:6d}/{total:6d} match  amax={amax:.17g}  "
                  f"1/ws2={source_divisor!r}  wd(6)={d6!r}  equal={d6 == source_divisor}")
    return {"rows": rows, "all_match": all(r["match"] == r["total"] for r in rows)}


# ---------------------------------------------------------------------------- activations


def load_activations(path: Path) -> dict:
    data = np.load(path)
    moe = {}
    intermediate = {}
    for layer in GATING_LAYERS:
        moe[layer] = torch.from_numpy(np.asarray(data[f"moe_input_{layer}"])).to(F64)
        intermediate[layer] = torch.from_numpy(
            np.asarray(data[f"expert0_intermediate_{layer}"])
        ).to(F64)
    return {"moe_input": moe, "expert0_intermediate": intermediate}


def synthetic_activations(store) -> dict:
    """Fixed-seed fallback batch (pre-registration 3), scaled to the S2 input_scale.

    ``g = input_scale`` is ModelOpt's activation global scale, whose associated activation absmax
    is ``448 * 6 / g``; each gating layer's synthetic ``x``/``h`` is rescaled to that amax.  The
    reported run does not exercise this path (provenance is forward-capture).
    """

    moe: dict[int, torch.Tensor] = {}
    intermediate: dict[int, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(SEED)
    for layer in GATING_LAYERS:
        def activation_absmax(proj: str) -> float:
            leaf = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}"
            scale = float(store.read_flat(leaf + ".input_scale").reshape(()).item())
            return 448.0 * 6.0 / scale

        x = torch.randn(len(PROMPT_TOKENS), 2560, generator=generator)
        moe[layer] = (x * (activation_absmax("gate_proj") / x.abs().max())).to(F64)
        h = torch.randn(len(PROMPT_TOKENS), 640, generator=generator)
        intermediate[layer] = (h * (activation_absmax("down_proj") / h.abs().max())).to(F64)
    return {"moe_input": moe, "expert0_intermediate": intermediate}


# ---------------------------------------------------------------------------- reporting


INTERPRETATIONS = [
    "The pre-registration is the rev-5 comment on issue #29; this script implements it as written.",
    "Activation divisor: the repo has no weight-derived activation divisor (tensor-formats 3.3), so "
    "'the repo's rule for the arm's weight configuration' is read as weight_divisor(amax(T_X(X)), "
    "target_arm): the arm's target (6 for C1/X1, 4 for C2/G1/G2/G3) applied to the arm's own "
    "transformed activation batch.",
    "The A4 quantizer is applied to bf16(T_X(X)), matching the BF16 activation-storage boundary the "
    "reference forward uses (bf16(x[rows]) at reference.py:313).",
    "G1: A is derived by the paper's Adam optimization (App. B; 80 steps, lr 0.12, joint W4A4 loss, "
    "d_lr = diag(Sigma_X)), then rounded to BF16 as the paper's explicit-runtime-smoothing case, and "
    "W A^-1 is quantized by C2's MSE rule at the arm divisor (pre-reg G1 definition).",
    "G2: the continuous gamma_W is optimized by the paper's weight-only Adam loss with A = I, then "
    "the fine E4M3 selection (Eq 45, window [gamma/1.2, gamma/0.3]) is realized over the repo's "
    "stored E4M3 words, whose effective step is word/divisor; no canonicalization is applied because "
    "A = I removes the rescaling gauge.",
    "G3: the paper's Algorithm 1 (Sec. 4.3) on the whole matrix, not the released code's per-16-"
    "column-block scope.  Initialization is the paper's (Sec. 5.3): beta_0,j = sum_i |W_ij|, "
    "alpha^-1_0,i = max_j |W_ij beta_j| / 6.  Both inverse scales are projected after every update "
    "onto the positive finite E4M3 set with a floor at its smallest positive value 2^-9 (the paper's "
    "admissible set includes E4M3 subnormals).  The released implementation instead clamps the "
    "projection at the smallest E4M3 *normal* 2^-6, inverts the beta initialization, and splits the "
    "matrix per 16-column block; those divergences are recorded, not silently reconciled.  "
    "Sensitivity: an earlier run using the released reciprocal beta init and the 2^-6 clamp made "
    "alpha^-1 uniform (0.015625) and is preserved as the committed bcd-reciprocal sensitivity "
    "report; only G3/X1 differ (pre-correction G3 won 11/25 with median -0.0011, still below the bar).",
    "Secondary metric: the represented weight is the arm's effective weight on the untransformed X "
    "(W_eff = W_q * A for G1, W_q * Diag(beta)^-1 for G3/X1, W_q otherwise), compared per 16-element "
    "K group against the FP64 oracle.  On the S2 layer-0 matrices this reproduces the pre-registration's "
    "stated C2 values (gate 1.9e-2/3.8e-2, up 2.5e-2/5.2e-2, down 1.9e-2/3.8e-2).",
    "S1/S2 arm weight divisors use the FP64 source amax (BF16 values widened, or the exact NVFP4 "
    "decode), matching pre-registration table 1, which prints the amax at full precision because the "
    "divisor is computed from it.",
    "Activation provenance fallback: the reported run uses the forward capture; the --synthetic path "
    "is a fixed-seed (seed 0) standard-normal batch rescaled per gating layer to the S2 input_scale's "
    "associated activation absmax 448*6/g, recorded but not exercised.",
    "Reported divisors: every arm record carries the actual per-matrix weight divisor (its binary32 "
    "stored value) and the target it was derived from, plus the same for the activation divisor, so "
    "the divisor choice is auditable per matrix and not only named as a rule.  Both come from "
    "weight_divisor(amax, target); the value is already binary32 and is stored verbatim.",
    "Self-description: the `reproduction` block records the exact capture, self-check and report "
    "commands, built from this process's argv and executable plus the module's frozen S2 root, "
    "gating layers and prompt tokens (never a hand-typed string); `environment` records the "
    "interpreter, torch/numpy versions and the git sha of this checkout.",
    "Alpha-fold attribution (G3/X1).  `fold_relative_l2` is the fold's own re-rounding loss: the "
    "relative L2 distance between the folded representation (alpha_inv multiplied into, and "
    "re-rounded within, the stored E4M3 block scale) and the same representation with alpha_inv "
    "applied exactly.  `e_no_fold` re-scores the arm with the fold disabled, i.e. alpha_inv carried "
    "as an exact separate per-output-row weight scale instead of folded into the block scale; "
    "`e - e_no_fold` isolates the fold's contribution to G3's primary metric.  The top-level "
    "`g3_fold_diagnostic` summarizes it.  The prototype's earlier causal wording (that the fold's "
    "re-rounding loss 'causes' G3's gating loss) is superseded by that measured separation.",
    "Pre-fix sensitivity: `--bcd-variant released` reproduces the committed "
    "`…report.bcd-reciprocal.json` by running G3/X1 with the released DualQuant implementation's "
    "reciprocal beta init (beta = 1 / sum_i |W_ij|) and E4M3 projection floor 2^-6 in place of the "
    "paper's beta_0,j = sum_i |W_ij| and 2^-9 floor.  It changes only G3/X1, is not a candidate arm, "
    "and does not gate; `bcd_variant` records which variant was run.",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_name() -> str:
    """This module's dotted name (``tools.proto.nvfp4_transform_prototype``)."""

    try:
        relative = Path(__file__).resolve().relative_to(_repo_root())
        return ".".join(relative.with_suffix("").parts)
    except ValueError:
        return Path(__file__).stem


def _git_sha() -> str | None:
    """The checkout's HEAD sha, read from git; ``None`` when the tree is not a git checkout."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sibling_output(stem: str, suffix: str) -> Path:
    return Path(__file__).resolve().parent / f"{stem}.{suffix}"


def reproduction_block(args) -> dict:
    """The exact capture / self-check / report commands, derived from this process, not typed.

    ``sys.executable`` and ``sys.argv`` supply the interpreter and the report invocation; the
    capture command is rebuilt from the frozen ``S2_ROOT``, ``GATING_LAYERS`` and ``PROMPT_TOKENS``
    constants and the activation file this run consumed; the self-check command targets the module's
    canonical ``…selfcheck.json`` sibling.  Each command is recorded both as an argv list (exact) and
    as a shell-quoted line.
    """

    python = sys.executable or "python3"
    module = _module_name()
    capture_out = str(args.activations) if args.activations else "capture.npz"
    commands = {
        "capture": [python, "-m", "tools.flash_next.reference",
                    "--model", str(S2_ROOT),
                    "--layers", str(max(GATING_LAYERS) + 1),
                    "--tokens", ",".join(str(token) for token in PROMPT_TOKENS),
                    "--record-taps", ",".join(str(layer) for layer in GATING_LAYERS),
                    "--out", capture_out],
        "self_check": [python, "-m", module, "--self-check",
                       "--out", str(_sibling_output(Path(__file__).stem, "selfcheck.json"))],
        "report": [python, "-m", module, *sys.argv[1:]],
    }
    return {
        "python_executable": python,
        "module": module,
        "commands": commands,
        "shell": {name: " ".join(shlex.quote(part) for part in argv)
                  for name, argv in commands.items()},
    }


def fold_diagnostic(rows: list[dict]) -> dict:
    """Isolate G3's alpha-fold re-rounding contribution on the gating set (finding 4).

    ``e - e_no_fold`` is the fold's contribution to the primary metric on each matrix.  The two
    median relative reductions (best per-matrix control = min(C1, C2), the bar's own control) show
    the fold's share of G3's shortfall from the bar: with the fold disabled G3's shortfall is
    smaller, so the fold is separable from the rest of G3's transform.
    """

    if not rows:
        return {}
    folded = np.array([row["arms"]["G3"]["e"] for row in rows])
    no_fold = np.array([row["arms"]["G3"]["e_no_fold"] for row in rows])
    fold_l2 = np.array([row["arms"]["G3"]["fold_relative_l2"] for row in rows])
    control = np.array([min(row["arms"]["C1"]["e"], row["arms"]["C2"]["e"]) for row in rows])
    reduction_folded = float(np.median((control - folded) / control))
    reduction_no_fold = float(np.median((control - no_fold) / control))
    shortfall_folded = -reduction_folded
    shortfall_no_fold = -reduction_no_fold
    contribution = folded - no_fold
    worst = int(np.argmax(contribution))
    return {
        "arm": "G3",
        "matrices": len(rows),
        "median_e_folded": float(np.median(folded)),
        "median_e_no_fold": float(np.median(no_fold)),
        "median_fold_relative_l2": float(np.median(fold_l2)),
        "median_fold_contribution": float(np.median(contribution)),
        "worst_fold_contribution": float(contribution[worst]),
        "worst_fold_matrix": rows[worst]["matrix_id"],
        "median_relative_reduction_folded": reduction_folded,
        "median_relative_reduction_no_fold": reduction_no_fold,
        "fold_share_of_median_shortfall": (shortfall_folded - shortfall_no_fold) / shortfall_folded,
        "attribution_isolated": True,
        "note": "e - e_no_fold isolates the alpha-fold E4M3 re-rounding; e_no_fold carries "
                "alpha_inv exactly as a separate per-output-row weight scale instead of folding it "
                "into the block scale.  The fold accounts for only part of G3's shortfall: the "
                "remainder is G3's own transform, so the earlier claim that the fold 'causes' G3's "
                "loss is superseded by these two measured reductions.",
    }


def print_report(report: dict) -> None:
    print("\n== per-matrix primary metric e_arm = ||Y_arm - Y_oracle||_F / ||Y_oracle||_F ==")
    header = f"{'matrix':16s} {'sha256':10s} " + " ".join(f"{a:>10s}" for a in ARM_ORDER)
    print(header)
    for row in report["matrices"]:
        cells = " ".join(f"{row['arms'][a]['e']:10.5f}" for a in ARM_ORDER)
        print(f"{row['matrix_id']:16s} {row['sha256'][:8]:10s} {cells}")
    print("\n== bar (gating set S1, best per-matrix control = min(C1, C2)) ==")
    for arm, verdict in report["bar"].items():
        state = "CLEARED" if verdict.get("clears") else "not cleared"
        if verdict.get("not_run"):
            print(f"  {arm}: NOT RUN ({verdict['reason']})")
        else:
            print(f"  {arm}: {state}  wins {verdict['wins']}/{verdict['matrices']}  "
                  f"median reduction {verdict['median_relative_reduction']:.4f}  "
                  f"worst ratio {verdict['worst_ratio']:.4f} ({verdict['worst_matrix']})")
    for label, key in (("gating S1", "arm_summary"), ("informational S2", "arm_summary_informational")):
        print(f"\n== per-arm e_arm summary ({label}) ==")
        for arm, stats in report[key].items():
            print(f"  {arm}: median {stats['median_e']:.5f}  range [{stats['min_e']:.5f}, "
                  f"{stats['max_e']:.5f}]  worst {stats['worst_matrix']}  "
                  f"wins {stats['wins_vs_best_control']}/{stats['matrices']}")
    diagnostic = report.get("g3_fold_diagnostic")
    if diagnostic:
        print(f"\n== G3 alpha-fold diagnostic (gating S1) ==")
        print(f"  median e folded {diagnostic['median_e_folded']:.5f}  no-fold "
              f"{diagnostic['median_e_no_fold']:.5f}  fold contribution "
              f"{diagnostic['median_fold_contribution']:.5f}  median fold_relative_l2 "
              f"{diagnostic['median_fold_relative_l2']:.5f}")
        print(f"  median relative reduction folded "
              f"{diagnostic['median_relative_reduction_folded']:.5f}  no-fold "
              f"{diagnostic['median_relative_reduction_no_fold']:.5f}  "
              f"fold share of shortfall {diagnostic['fold_share_of_median_shortfall']:.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=Path, help="capture.npz from the reference tap")
    parser.add_argument("--out", type=Path, help="write the raw JSON report here")
    parser.add_argument("--self-check", action="store_true", help="run only the S2 correctness anchor")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the recorded fixed-seed synthetic fallback instead of a capture")
    parser.add_argument("--dev-matrix-limit", type=int, default=0,
                        help="DEV ONLY: truncate each matrix set (voids the run)")
    parser.add_argument("--bcd-variant", choices=("paper", "released"), default="paper",
                        help="G3/X1 DualQuant variant: 'paper' (reported run) or 'released' "
                             "(regenerates the pre-fix bcd-reciprocal sensitivity report)")
    args = parser.parse_args(argv)

    from tools.convert.sources.safetensors import SafetensorsSource

    store = SafetensorsSource(S2_ROOT)
    if args.self_check:
        result = self_check(store)
        print(f"\nself-check all-match: {result['all_match']}")
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))
        return 0 if result["all_match"] else 1

    if args.synthetic:
        activation_source = "synthetic-fallback"
        activations = synthetic_activations(store)
    else:
        if not args.activations:
            parser.error("--activations is required (or pass --synthetic)")
        activation_source = "forward-capture"
        activations = load_activations(args.activations)

    shards = GgufShards(S1_ROOT)
    matrices = load_s1(shards, activations) + load_s2(store, activations)
    if args.dev_matrix_limit:
        matrices = matrices[: args.dev_matrix_limit]

    report: dict = {
        "protocol": "kido5217/Infernix issue #30 pre-registration rev 5",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__, "numpy": np.__version__,
                        "git_sha": _git_sha()},
        "reproduction": reproduction_block(args),
        "bcd_variant": args.bcd_variant,
        "seed": SEED,
        "prompt": {"text": PROMPT_TEXT, "tokens": list(PROMPT_TOKENS),
                   "token_count": len(PROMPT_TOKENS)},
        "sources": {"S1_root": str(S1_ROOT), "S2_root": str(S2_ROOT)},
        "activation_provenance": activation_source,
        "activation_file": str(args.activations) if args.activations else None,
        "activation_sha256": hashlib.sha256(
            np.ascontiguousarray(np.concatenate(
                [activations["moe_input"][l].to(F32).numpy().ravel() for l in GATING_LAYERS] +
                [activations["expert0_intermediate"][l].to(F32).numpy().ravel() for l in GATING_LAYERS]
            )).tobytes()
        ).hexdigest(),
        "interpretations": INTERPRETATIONS,
        "dev_mode": bool(args.dev_matrix_limit),
        "matrices": [],
        "bar": {},
    }

    for matrix in matrices:
        arms = run_arms(matrix.oracle, matrix.activation, bcd_variant=args.bcd_variant)
        report["matrices"].append({
            "matrix_id": matrix.matrix_id, "source": matrix.source, "layer": matrix.layer,
            "kind": matrix.kind, "n": matrix.n, "k": matrix.k,
            "input_scale": matrix.input_scale, "sha256": matrix.sha256, "arms": arms,
        })
        print(f"done {matrix.matrix_id}  " + " ".join(f"{a}:{arms[a]['e']:.5f}" for a in ARM_ORDER))

    gating_rows = [row for row in report["matrices"] if row["source"] == "S1"]
    informational_rows = [row for row in report["matrices"] if row["source"] == "S2"]
    report["bar"] = {arm: bar_verdict(gating_rows, arm) for arm in GATING_ARMS}
    report["bar_informational"] = {arm: bar_verdict(informational_rows, arm) for arm in GATING_ARMS}
    report["arm_summary"] = arm_summary(gating_rows)
    report["arm_summary_informational"] = arm_summary(informational_rows)
    report["g3_fold_diagnostic"] = fold_diagnostic(gating_rows)

    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
