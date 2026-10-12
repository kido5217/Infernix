"""Per-layer routing statistics of the recorded coding-agent traces (wayfinder ticket #17).

Reads the engine's version-2 NRTRACE1 route traces with host_tier.read_trace (the canonical reader)
and, over the *generation* rounds only (ForcedTokens, Decode, Verify; prefill chunks are a separate
regime), reports per layer:

  * routing entropy of the layer's expert-usage distribution (bits),
  * concentration (share of uses taken by the top-64 experts),
  * distinct active experts,
  * temporal drift (entropy over the first vs the second half, set turnover, novel-use share),
per session and pooled, plus the cross-session consistency statistics that decide whether a static
per-layer prior has anything stable to act on (across-layer range, per-layer across-session spread,
pairwise Spearman rank correlation, and a within-session bootstrap noise floor).

Conventions
-----------
* One routed use = one (live generation column, top-k slot). Generation columns are exactly the
  accepted generated tokens: their live count equals the manifest's `generated_tokens` for every
  session (checked below). Rejected draft columns and non-live prefill columns are excluded.
* A layer's expert-usage distribution is the empirical histogram of its expert ids over those uses.
  Unselected experts carry probability zero and are dropped, so H is the entropy of the observed
  distribution (0..log2(512) = 9 bits; the observed support size is the distinct-active count).

Usage (from the repository root; host Python 3.13 ships numpy):

    python3 tools/expert_cache_replay/per_layer_routing_stats.py --json out/per-layer-routing-stats.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

# Reuse the engine's route-trace reader (same directory) instead of writing a second parser.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_tier import DECODE, FORCED, VERIFY, read_trace  # noqa: E402

GEN_KINDS = (FORCED, DECODE, VERIFY)
TRACES = 'out/route-traces/*.bin'
MANIFEST = 'bench/route_traces/manifest.json'
TOP_CUT = 64          # concentration reference cut (512 experts/layer)
SUBSAMPLE_TOKENS = 400  # common token budget for the sample-size-controlled comparison
BOOTSTRAP = 200


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a count vector, over its observed support only."""
    total = counts.sum()
    if total == 0:
        return float('nan')
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def layer_token_ids(trace) -> np.ndarray:
    """Stack the live generation columns in execution order into ids[layers, tokens, top_k]."""
    per_round = []
    for r in trace.rounds:
        if r.kind not in GEN_KINDS:
            continue
        cols = r.rows * r.width
        live = np.flatnonzero(r.live[:cols])
        if live.size:
            per_round.append(r.routes[:, live, :])
    return np.concatenate(per_round, axis=1)


def layer_counts(ids: np.ndarray, experts: int) -> np.ndarray:
    """counts[layers, experts] from ids[layers, tokens, top_k]."""
    layers = ids.shape[0]
    out = np.zeros((layers, experts), dtype=np.int64)
    for l in range(layers):
        out[l] = np.bincount(ids[l].ravel(), minlength=experts)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (no scipy)."""
    def ranks(v):
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ranks within tie groups
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float('nan')


def stats_for(counts: np.ndarray, top_cut: int = TOP_CUT) -> dict:
    n = counts.sum(axis=1)
    H = np.array([entropy(counts[l]) for l in range(counts.shape[0])])
    conc = np.array([np.sort(counts[l])[::-1][:top_cut].sum() / n[l] for l in range(counts.shape[0])])
    distinct = (counts > 0).sum(axis=1)
    return {'n': n, 'entropy': H, 'top_share': conc, 'distinct': distinct}


def bootstrap_noise(ids: np.ndarray, experts: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Per-layer std of the entropy under token-resampling within the session (the noise floor)."""
    layers, tokens, _ = ids.shape
    H = np.empty((draws, layers))
    for d in range(draws):
        idx = rng.integers(0, tokens, size=tokens)
        sub = ids[:, idx, :]
        for l in range(layers):
            H[d, l] = entropy(np.bincount(sub[l].ravel(), minlength=experts))
    return H.std(axis=0)


def subsampled_entropy(ids: np.ndarray, experts: int, take: int, reps: int, rng) -> np.ndarray:
    """Mean per-layer entropy over `reps` random token subsets of size `take` (sample-size control)."""
    layers, tokens, _ = ids.shape
    if take >= tokens:
        take = tokens
    acc = np.zeros((reps, layers))
    for r in range(reps):
        idx = rng.choice(tokens, size=take, replace=False)
        sub = ids[:, idx, :]
        for l in range(layers):
            acc[r, l] = entropy(np.bincount(sub[l].ravel(), minlength=experts))
    return acc.mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', default=TRACES)
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--json', help='dump the full metrics to this path')
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    gen_tokens = {s['name']: s['generated_tokens'] for s in manifest['sessions']}

    names, per_session = [], {}
    for path in sorted(glob.glob(args.traces)):
        name = os.path.splitext(os.path.basename(path))[0]
        trace = read_trace(path)
        layers, experts = trace.header['layers'], trace.header['experts']
        ids = layer_token_ids(trace)
        # Cross-check the reader against the manifest: live generation columns == generated_tokens.
        assert ids.shape[1] == gen_tokens[name], (name, ids.shape[1], gen_tokens.get(name))
        counts = layer_counts(ids, experts)
        st = stats_for(counts)
        half = ids.shape[1] // 2
        c_first, c_second = layer_counts(ids[:, :half], experts), layer_counts(ids[:, half:], experts)
        H_first = np.array([entropy(c_first[l]) for l in range(layers)])
        H_second = np.array([entropy(c_second[l]) for l in range(layers)])
        sets_first = [(c_first[l] > 0) for l in range(layers)]
        sets_second = [(c_second[l] > 0) for l in range(layers)]
        jaccard = np.array([
            (sets_first[l] & sets_second[l]).sum() / (sets_first[l] | sets_second[l]).sum()
            for l in range(layers)])
        novel = np.array([
            c_second[l][~sets_first[l]].sum() / c_second[l].sum() for l in range(layers)])
        per_session[name] = {
            'layers': layers, 'experts': experts, 'tokens': int(ids.shape[1]), 'ids': ids,
            'counts': counts, 'n': st['n'], 'entropy': st['entropy'],
            'top_share': st['top_share'], 'distinct': st['distinct'],
            'H_first': H_first, 'H_second': H_second,
            'jaccard': jaccard, 'novel': novel}
        names.append(name)

    layers, experts = per_session[names[0]]['layers'], per_session[names[0]]['experts']

    # Pooled across the 8 sessions.
    pooled_counts = sum(per_session[n]['counts'] for n in names)
    pooled = stats_for(pooled_counts)

    # Cross-session consistency of the per-layer entropy signal.
    Hmat = np.stack([per_session[n]['entropy'] for n in names], axis=1)   # [48, 8]
    layer_mean = Hmat.mean(axis=1)
    layer_std = Hmat.std(axis=1)
    pairwise = [spearman(Hmat[:, i], Hmat[:, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    long_names = [n for n in names if n.startswith('long')]
    short_names = [n for n in names if n.startswith('short')]

    def group_stats(group):
        G = np.stack([per_session[n]['entropy'] for n in group], axis=1)
        return {'range': float(G.mean(axis=1).max() - G.mean(axis=1).min()),
                'across_session_std': float(G.std(axis=1).mean()),
                'means': G.mean(axis=1)}

    gl, gs = group_stats(long_names), group_stats(short_names)
    group_spearman = spearman(gl['means'], gs['means'])

    # Test-retest reliability: does the first half of a session order layers like the second half?
    split_half = [spearman(per_session[n]['H_first'], per_session[n]['H_second']) for n in names]

    # Sample-size-controlled comparison (all sessions at the same token budget) and the noise floor.
    rng = np.random.default_rng(17)
    subs = np.stack([subsampled_entropy(per_session[n]['ids'], experts, SUBSAMPLE_TOKENS, 40, rng) for n in names], axis=1)
    noise = np.stack([bootstrap_noise(per_session[n]['ids'], experts, BOOTSTRAP, rng) for n in names], axis=1)
    conc_corr = spearman(pooled['entropy'], pooled['top_share'])

    def rng_str(v):
        return f'{v.min():.3f}..{v.max():.3f}'

    print(f'sessions: {", ".join(f"{n}({per_session[n]["tokens"]} tok)" for n in names)}')
    print(f'layers={layers} experts={experts} top_k={read_trace(sorted(glob.glob(args.traces))[0]).header["top_k"]} '
          f'max entropy=log2({experts})={math.log2(experts):.4f} bits')
    print()
    print(f'across-layer entropy range (full data)   per session: ' +
          ', '.join(f'{n}={rng_str(per_session[n]["entropy"])}' for n in names))
    print(f'across-layer range of the POOLED entropy: {rng_str(pooled["entropy"])} '
          f'(R={pooled["entropy"].max()-pooled["entropy"].min():.4f} bits)')
    print(f'per-layer entropy spread across 8 sessions: mean std={layer_std.mean():.4f} bits, '
          f'median={np.median(layer_std):.4f}, max={layer_std.max():.4f} bits')
    print(f'  within regime: long range={gl["range"]:.4f}, long across-session std={gl["across_session_std"]:.4f} | '
          f'short range={gs["range"]:.4f}, short across-session std={gs["across_session_std"]:.4f}')
    print(f'  long-group mean vs short-group mean layer ordering: Spearman={group_spearman:.3f}')
    print(f'sample-size-controlled (T={SUBSAMPLE_TOKENS}): across-layer range={rng_str(subs.mean(axis=1))} '
          f'(R={subs.mean(axis=1).max()-subs.mean(axis=1).min():.4f}), per-layer across-session std mean={subs.std(axis=1).mean():.4f}')
    print(f'within-session bootstrap noise (T={per_session[names[0]]["tokens"]}-ish): '
          f'per-layer std mean={noise.mean():.4f}, max={noise.max():.4f} bits')
    print(f'pairwise Spearman of layer entropy across sessions: mean={np.nanmean(pairwise):.3f}, '
          f'min={np.nanmin(pairwise):.3f}, max={np.nanmax(pairwise):.3f} (of {len(pairwise)} pairs)')
    print(f'split-half test-retest Spearman (first vs second half, per session): '
          f'mean={np.nanmean(split_half):.3f}, min={np.nanmin(split_half):.3f}, max={np.nanmax(split_half):.3f}')
    print(f'entropy vs pooled top-64 share: Spearman={conc_corr:.3f}')
    print(f'entropy range / per-layer across-session std = '
          f'{(layer_mean.max()-layer_mean.min())/layer_std.mean():.2f}x')
    print(f'entropy range / bootstrap noise = {(layer_mean.max()-layer_mean.min())/noise.mean():.2f}x')
    print()
    print('| layer | pooled H | ' + ' | '.join(f'{n} H' for n in names) +
          ' | mean H | std | pooled top-64 | pooled distinct |')
    print('|---' * (5 + len(names)) + '|')
    for l in range(layers):
        rows = [f'{l}', f'{pooled["entropy"][l]:.4f}']
        rows += [f'{per_session[n]["entropy"][l]:.4f}' for n in names]
        rows += [f'{layer_mean[l]:.4f}', f'{layer_std[l]:.4f}',
                 f'{pooled["top_share"][l]:.4f}', f'{int(pooled["distinct"][l])}']
        print('| ' + ' | '.join(rows) + ' |')
    print()
    print('Temporal drift (first vs second half of each session, per layer):')
    print('| layer | mean dH | max dH | mean Jaccard | max novel-use share |')
    print('|---|---|---|---|---|')
    dH = np.abs(np.stack([per_session[n]['H_first'] - per_session[n]['H_second'] for n in names], axis=1))
    jac = np.stack([per_session[n]['jaccard'] for n in names], axis=1)
    nov = np.stack([per_session[n]['novel'] for n in names], axis=1)
    for l in range(layers):
        print(f'| {l} | {dH[l].mean():.4f} | {dH[l].max():.4f} | {jac[l].mean():.3f} | {nov[l].max():.3f} |')
    print(f'drift summary: mean |dH|={dH.mean():.4f} bits (max {dH.max():.4f}), '
          f'mean Jaccard={jac.mean():.3f}, mean novel-use share={nov.mean():.3f}')

    if args.json:
        dump = {
            'convention': 'generation rounds only; live columns; top-k slots as uses; entropy over observed experts',
            'sessions': names,
            'tokens': {n: per_session[n]['tokens'] for n in names},
            'layer_entropy': {n: per_session[n]['entropy'].tolist() for n in names},
            'layer_top64_share': {n: per_session[n]['top_share'].tolist() for n in names},
            'layer_distinct': {n: per_session[n]['distinct'].tolist() for n in names},
            'pooled_entropy': pooled['entropy'].tolist(),
            'pooled_top64_share': pooled['top_share'].tolist(),
            'pooled_distinct': pooled['distinct'].tolist(),
            'H_first': {n: per_session[n]['H_first'].tolist() for n in names},
            'H_second': {n: per_session[n]['H_second'].tolist() for n in names},
            'jaccard': {n: per_session[n]['jaccard'].tolist() for n in names},
            'novel_share': {n: per_session[n]['novel'].tolist() for n in names},
            'summary': {
                'pooled_range': float(pooled['entropy'].max() - pooled['entropy'].min()),
                'layer_mean_range': float(layer_mean.max() - layer_mean.min()),
                'layer_std_mean': float(layer_std.mean()),
                'layer_std_max': float(layer_std.max()),
                'long_range': gl['range'], 'long_across_session_std': gl['across_session_std'],
                'short_range': gs['range'], 'short_across_session_std': gs['across_session_std'],
                'long_short_ordering_spearman': float(group_spearman),
                'subsample_range': float(subs.mean(axis=1).max() - subs.mean(axis=1).min()),
                'subsample_session_std_mean': float(subs.std(axis=1).mean()),
                'bootstrap_noise_mean': float(noise.mean()),
                'bootstrap_noise_max': float(noise.max()),
                'spearman_mean': float(np.nanmean(pairwise)),
                'spearman_min': float(np.nanmin(pairwise)),
                'spearman_max': float(np.nanmax(pairwise)),
                'split_half_spearman_mean': float(np.nanmean(split_half)),
                'entropy_top64_spearman': float(conc_corr),
                'drift_mean_dH': float(dH.mean()), 'drift_max_dH': float(dH.max()),
                'drift_mean_jaccard': float(jac.mean()), 'drift_mean_novel': float(nov.mean()),
            },
            'pairwise_spearman': {f'{names[i]}|{names[j]}': pairwise[k]
                                  for k, (i, j) in enumerate((i, j) for i in range(len(names)) for j in range(i + 1, len(names)))},
        }
        with open(args.json, 'w') as f:
            json.dump(dump, f, indent=1)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
