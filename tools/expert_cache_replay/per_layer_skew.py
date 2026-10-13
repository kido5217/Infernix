"""Per-layer residency-skew replay prototype for wayfinder ticket #12.

Answers one question against the frozen protocol (issue #10, first comment): **at the deployed
capacity, does a per-layer residency skew beat the engine's global LFRU, and by how much per 1 % of
the pool?**

This is a *prototype*: a cheap, honest, reproducible artifact, not production code. It does not
change `host_tier.py`; it imports that module's canonical `read_trace` and mirrors its VRAM policy
(the LFRU score ``f / (age + 1)``, the recorded per-round promotion budget, the event-atomic
grouping, the one-round copy visibility) so the two can be compared directly.

Model of the replay (VRAM frame pool only). The deployment's host tier (`--expert-ram-mib 66000`,
66000 MiB > 48 x 512 experts x 2,764,800 B) holds the whole 24,576-expert key space, so the RAM tier
never binds and this is exactly `host_tier.py`'s `--validate` regime: full RAM, the trace's own
recorded per-round promotion budgets, and a miss = a routed expert absent from the frame pool. The
metric is therefore **live routed VRAM misses per generated token**, summed over the generation
rounds (Decode + Verify + Forced) the protocol fixes as the token unit.

Semantics:

* **event-atomic** (the calibrated contract, matching `host_tier.py`): one layer-call is one policy
  tick; the layer's unique live routed experts are ranked and admitted together, and the whole group
  is protected from eviction while it is processed.
* **sequential / per-access** (control arm (v)): every routed access is one policy tick, ranked and
  admitted on its own with no in-group protection, so a hot expert used earlier in the round can be
  evicted by a later access of the same round. The capacity is identical; only replay semantics
  change.

Knobs (per layer):

* **score weight** ``w_l`` (arm b): the LFRU score becomes ``w_l * f / (age + 1)``; a shared pool
  with non-uniform weights keeps more residents in high-weight layers.
* **capacity** ``K_l`` (arm a): hard per-layer quotas. Implemented on the shared pool by restricting
  victim choice to layer ``l`` when it is at its quota (and to over-quota layers while the pool
  fills), which is equivalent to independent per-layer LFRU caches summing to ``V``.
* **warm-start seed** (arm c): an initial resident set, per session category.

Usage (repository root, Python 3.13 + numpy):

    python3 tools/expert_cache_replay/per_layer_skew.py calibrate
    python3 tools/expert_cache_replay/per_layer_skew.py arms --json out/per-layer-skew.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from concurrent.futures import ProcessPoolExecutor

import numpy as np

# The canonical reader and trace constants live in host_tier.py; import rather than duplicate.
from host_tier import (DECODE, FORCED, PREFILL, VERIFY, read_trace)  # noqa: F401

GENERATION_KINDS = (DECODE, VERIFY, FORCED)
TRACE_DIR = 'out/route-traces'
SESSIONS = ['long-1', 'long-2', 'long-3', 'long-4', 'short-1', 'short-2', 'short-3', 'short-4']
CATEGORY = {name: name.split('-')[0] for name in SESSIONS}
DEFAULT_VRAM = 0  # 0 = each trace's own header frames (the deployed/startup capacity)


# ---------------------------------------------------------------------------------------------- model

class SkewSim:
    """VRAM frame pool under LFRU with per-layer score weights and/or hard per-layer quotas.

    Full-RAM equivalent of `host_tier.TierSim` (mode `--ram-slots all`, `--vram-budget recorded`,
    prior none): the RAM/SSD tiers do not bind, so only the frame pool and its promotions matter.
    """

    def __init__(self, header, vram, *, quota=None, weights=None, atomic=True,
                 seed_keys=None, seed_count=0, per_round_vram=False):
        self.L, self.E, self.K = header['layers'], header['experts'], header['top_k']
        self.N = self.L * self.E
        self.V = max(0, min(int(vram), self.N - 1))
        self.atomic = atomic
        # Protocol §2 point (b): follow the trace's own per-round frame count instead of one fixed
        # V, so the same arm can also be read at the time-weighted mean capacity.
        self.per_round_vram = per_round_vram
        self.lay = np.arange(self.N, dtype=np.int64) // self.E
        if weights is None:
            self.wkey = None
            self.uniform = True
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (self.L,):
                raise ValueError(f'weights must have one entry per layer ({self.L})')
            self.wkey = np.repeat(w, self.E)
            self.uniform = bool(np.all(w == w[0]))
        if quota is None:
            self.quota = None
        else:
            k = np.asarray(quota, dtype=np.int64)
            if k.shape != (self.L,):
                raise ValueError(f'quota must have one entry per layer ({self.L})')
            self.quota = k

        # VRAM state, mirroring host_tier.TierSim's full mode.
        self.v_count = np.zeros(self.N, dtype=np.int64)   # f, never reset (host_tier never halves)
        self.v_last = np.zeros(self.N, dtype=np.int64)
        self.v_now = 0
        self.v_pos = np.full(self.N, -1, dtype=np.int64)
        self.v_keys = np.zeros(max(self.V, 1), dtype=np.int64)
        self.v_n = 0
        self.visible = np.zeros(self.N, dtype=np.int64)   # first round a frame is readable
        self.mark = np.zeros(self.N, dtype=bool)
        self.cnt = np.zeros(self.L, dtype=np.int64)        # residents per layer
        self.round = 0

        # Metrics: live routed keys and their VRAM hits, per trace kind.
        self.live_keys = {k: 0 for k in GENERATION_KINDS}
        self.live_hits = {k: 0 for k in GENERATION_KINDS}
        self.tokens = {k: 0 for k in GENERATION_KINDS}
        self.reads = {k: 0 for k in GENERATION_KINDS}
        self.initial_residents = 0

        if seed_keys is not None:
            self._seed(seed_keys, seed_count)

    # ------------------------------------------------------------------ basic pool ops
    def _insert(self, k):
        self.v_pos[k] = self.v_n
        self.v_keys[self.v_n] = k
        self.v_n += 1
        self.cnt[self.lay[k]] += 1

    def _erase(self, k):
        at = self.v_pos[k]
        tail = self.v_keys[self.v_n - 1]
        self.v_keys[at] = tail
        self.v_pos[tail] = at
        self.v_n -= 1
        self.v_pos[k] = -1
        self.cnt[self.lay[k]] -= 1

    def _admit(self, k):
        self._insert(k)
        self.visible[k] = self.round + 1

    def _vscore(self, keys):
        score = self.v_count[keys] / (self.v_now - self.v_last[keys] + 1)
        return score if self.uniform else self.wkey[keys] * score

    def _victims(self, need, layer):
        """The `need` lowest-score unmarked residents, subject to per-layer quotas."""
        cand = self.v_keys[:self.v_n]
        cand = cand[~self.mark[cand]]
        if self.quota is not None and cand.size:
            if self.cnt[layer] >= self.quota[layer]:
                cand = cand[self.lay[cand] == layer]
            else:
                over = np.flatnonzero(self.cnt > self.quota)
                if over.size:
                    cand = cand[np.isin(self.lay[cand], over)]
        if cand.size == 0:
            return []
        score = self._vscore(cand)
        if need == 1:
            low = score.min()
            return [int(cand[score == low].min())]
        order = np.lexsort((cand, score))[:need]
        return [int(x) for x in cand[order]]

    def _seed(self, keys, count):
        seen = set()
        cap = self.V if self.V else 0
        for k in keys:
            k = int(k)
            if k in seen or k < 0 or k >= self.N or self.v_n >= cap:
                continue
            seen.add(k)
            self.v_count[k] = count
            self.v_last[k] = 0
            self._admit(k)
        self.initial_residents = self.v_n

    # ------------------------------------------------------------------ policy step, one layer
    def _layer_step_atomic(self, group, budget, layer):
        self.v_now += 1
        if group.size == 0:
            return
        self.v_count[group] += 1
        self.v_last[group] = self.v_now
        misses = group[self.v_pos[group] < 0]
        if misses.size == 0 or self.V == 0:
            return
        self.mark[group] = True
        try:
            if budget >= misses.size:
                # host_tier's unbudgeted engine path: evict the lowest-score residents, admit all.
                total = self.v_n + misses.size
                for v in (self._victims(total - self.V, layer) if total > self.V else []):
                    self._erase(v)
                for k in misses:
                    if self.v_n < self.V:
                        self._admit(int(k))
                return
            if budget == 0:
                return
            score = self._vscore(misses)
            admitted = 0
            for i in np.lexsort((misses, -score)):
                if admitted >= budget:
                    break
                k = int(misses[i])
                if self.v_n >= self.V:
                    vic = self._victims(1, layer)
                    if not vic or not score[i] > self._vscore(np.array([vic[0]]))[0]:
                        break
                    self._erase(vic[0])
                self._admit(k)
                admitted += 1
        finally:
            self.mark[group] = False

    def _layer_step_access(self, acc, budget, layer):
        """Sequential/per-access control: every access is a policy tick, no in-group protection."""
        left = budget
        if acc.size == 0:
            self.v_now += 1
            return
        for k in acc:
            k = int(k)
            self.v_now += 1
            self.v_count[k] += 1
            self.v_last[k] = self.v_now
            if self.v_pos[k] >= 0 or left <= 0 or self.V == 0:
                continue
            if self.v_n >= self.V:
                vic = self._victims(1, layer)
                if not vic or not self._vscore(k) > self._vscore(vic[0]):
                    continue
                self._erase(vic[0])
            self._admit(k)
            left -= 1

    # ------------------------------------------------------------------ round
    def step(self, r, budget):
        if self.per_round_vram and getattr(r, 'frames', 0):
            self.V = max(0, min(int(r.frames), self.N - 1))
        # Policy side: distinct live routed experts per layer (host_tier's `live_routes`).
        # Accounting for every layer runs before any admission of the round, exactly as
        # host_tier does; otherwise a same-round eviction could hide a later layer's hit.
        layer_base = (np.arange(self.L) * self.E)[:, None]
        live = r.routes[:, r.live, :].reshape(self.L, -1) + layer_base
        is_gen = r.kind in GENERATION_KINDS
        if is_gen:
            # host_tier.py:739-748 counts a FORCED round as rows*width columns and a Decode/Verify
            # round as its live count. No Forced rounds exist in the current traces, but the unit is
            # the protocol's, so both simulators must agree for traces that carry them.
            self.tokens[r.kind] += (int(r.rows) * int(r.width) if r.kind == FORCED
                                    else int(np.count_nonzero(r.live)))
        groups = []
        for layer in range(self.L):
            group = np.unique(live[layer])
            groups.append(group)
            if is_gen:
                self.live_keys[r.kind] += int(group.size)
                self.live_hits[r.kind] += int(np.count_nonzero(
                    (self.v_pos[group] >= 0) & (self.visible[group] <= self.round)))
        for layer in range(self.L):
            if self.atomic:
                self._layer_step_atomic(groups[layer], budget, layer)
            else:
                self._layer_step_access(live[layer], budget, layer)
        self.round += 1

    def run(self, rounds):
        for r in rounds:
            self.step(r, r.budget)
        return self.summary()

    def summary(self):
        gen_tokens = sum(self.tokens[k] for k in GENERATION_KINDS)
        gen_live = sum(self.live_keys[k] for k in GENERATION_KINDS)
        gen_hits = sum(self.live_hits[k] for k in GENERATION_KINDS)
        misses = gen_live - gen_hits
        return {
            'generated_tokens': gen_tokens,
            'live_routed_experts': gen_live,
            'live_vram_hits': gen_hits,
            'live_vram_misses': misses,
            'misses_per_token': misses / gen_tokens if gen_tokens else 0.0,
            'hit_rate': gen_hits / gen_live if gen_live else 0.0,
            'resident_frames': int(self.v_n),
            'initial_residents': int(self.initial_residents),
        }


# ---------------------------------------------------------------------------------------------- data

@dataclass
class Session:
    name: str
    category: str
    header: dict
    rounds: list
    frames: int


def load_sessions(names=None, trace_dir=TRACE_DIR):
    names = names or SESSIONS
    out = []
    for name in names:
        trace = read_trace(f'{trace_dir}/{name}.bin')
        out.append(Session(name, CATEGORY[name], trace.header, trace.rounds, trace.header['frames']))
    return out


def simulate(session, vram=None, per_round=False, **kwargs):
    vram = session.frames if not vram else vram
    sim = SkewSim(session.header, vram, per_round_vram=per_round, **kwargs)
    summary = sim.run(session.rounds)
    summary['vram'] = int(sim.V)
    return summary


def pooled(rows):
    """Pooled misses/token over a list of per-session summaries."""
    tokens = sum(r['generated_tokens'] for r in rows)
    misses = sum(r['live_vram_misses'] for r in rows)
    return misses / tokens if tokens else 0.0


# ---------------------------------------------------------------------------------------------- priors

def usage_counts(sessions, header, include_self=True, exclude=None):
    """Live routed use counts per (layer, expert) over the sessions, as host_tier.use_counts does."""
    counts = np.zeros(header['layers'] * header['experts'], dtype=np.int64)
    for s in sessions:
        if not include_self and s is exclude:
            continue
        for r in s.rounds:
            if r.kind not in GENERATION_KINDS:
                continue
            routes = r.routes[:, r.live, :]
            keys = routes.reshape(routes.shape[0], -1) + (np.arange(routes.shape[0]) * header['experts'])[:, None]
            np.add.at(counts, keys.ravel(), 1)
    return counts


def category_profile(sessions, category, exclude=None):
    """The warm-start domain key: expert usage over the sessions of one category."""
    same = [s for s in sessions if s.category == category]
    counts = usage_counts(same, sessions[0].header, include_self=exclude is None, exclude=exclude)
    return counts


def entropy_per_layer(session):
    """Per-layer routing entropy H_l (bits) of one session's live routed uses, zero-prob dropped."""
    L, E = session.header['layers'], session.header['experts']
    counts = np.zeros((L, E), dtype=np.int64)
    for r in session.rounds:
        if r.kind not in GENERATION_KINDS:
            continue
        routes = r.routes[:, r.live, :]  # [L, live, top_k]
        keys = routes.reshape(L, -1)
        for layer in range(L):
            np.add.at(counts[layer], keys[layer], 1)
    H = np.zeros(L)
    for layer in range(L):
        c = counts[layer][counts[layer] > 0].astype(np.float64)
        p = c / c.sum()
        H[layer] = -(p * np.log2(p)).sum()
    return H, counts


# ---------------------------------------------------------------------------------------------- response curves

@dataclass
class MiniRound:
    kind: int
    tokens: int
    budget: int
    live: np.ndarray
    routes: np.ndarray


_SESS_CACHE: dict = {}


def _init_sessions(trace_dir):
    if not _SESS_CACHE:
        for s in load_sessions(trace_dir=trace_dir):
            _SESS_CACHE[s.name] = s


def _layer_mini_rounds(sessions, layer):
    return [MiniRound(r.kind, r.tokens, r.budget, r.live, r.routes[layer:layer + 1])
            for s in sessions for r in s.rounds]


def _curve_job(job):
    V, layer, names, Kgrid = job
    sessions = [_SESS_CACHE[n] for n in names]
    header = sessions[0].header
    one = {'layers': 1, 'experts': header['experts'], 'top_k': header['top_k']}
    rounds = _layer_mini_rounds(sessions, layer)
    return V, layer, np.array([SkewSim(one, c).run(rounds)['hit_rate'] for c in Kgrid])


def capacity_grid(V, L, E=512, span=8):
    k0 = V // L
    step = max(2, k0 // 12)
    return np.unique(np.clip(k0 + step * np.arange(-span, span + 1), 0, L * E - 1)), k0


def response_curves(Vs, names, trace_dir, jobs=1, span=8):
    """H[l](K) per layer: LFRU hit rate on the pooled traces at each capacity in the grid."""
    _init_sessions(trace_dir)
    header = _SESS_CACHE[names[0]].header
    L, E = header['layers'], header['experts']
    grids, task = {}, []
    for V in Vs:
        Kg, _ = capacity_grid(V, L, E, span)
        grids[V] = Kg
        task.extend((V, l, names, Kg) for l in range(L))
    out = {V: [None] * L for V in Vs}
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=_init_sessions, initargs=(trace_dir,)) as pool:
            for V, layer, H in pool.map(_curve_job, task):
                out[V][layer] = H
    else:
        for job in task:
            V, layer, H = _curve_job(job)
            out[V][layer] = H
    return {V: (grids[V], np.asarray(out[V])) for V in Vs}


def even_split(V, L):
    k0, rem = divmod(V, L)
    K = np.full(L, k0, dtype=np.int64)
    K[:rem] += 1
    return K


def _hit(curves, Kgrid, layer, k):
    return float(np.interp(max(k, 0), Kgrid, curves[layer]))


def margins(curves, Kgrid, K, deltaK):
    """G+ (gain from +deltaK) and D- (loss from -deltaK) in hit-rate points."""
    L = curves.shape[0]
    G = np.array([_hit(curves, Kgrid, l, K[l] + deltaK) - _hit(curves, Kgrid, l, K[l]) for l in range(L)])
    D = np.array([_hit(curves, Kgrid, l, K[l]) - _hit(curves, Kgrid, l, K[l] - deltaK) for l in range(L)])
    return G, D


def allocate(curves, Kgrid, V, deltaK, eps_drop, tau_low, min_k=1, max_transfers=100000):
    """MoE-CORE 3.1 incremental marginal-loss transfer: K_l -> K_l +/- deltaK while G+_r > D-_d."""
    L = curves.shape[0]
    K = even_split(V, L)
    G = D = np.zeros(L)
    for _ in range(max_transfers):
        G, D = margins(curves, Kgrid, K, deltaK)
        donors = np.flatnonzero(D < eps_drop)
        recv = np.flatnonzero(G > tau_low)
        if donors.size == 0 or recv.size == 0:
            break
        recv = recv[np.argsort(-G[recv])]
        donors = donors[np.argsort(D[donors])]
        moved = False
        for r in recv:
            for d in donors:
                if d == r or K[d] - deltaK < min_k:
                    continue
                if G[r] > D[d]:
                    K[d] -= deltaK
                    K[r] += deltaK
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return K, G, D


def allocate_at(V, curves, L, deltaK=None, eps_drop=None, tau_low=None):
    Kgrid, CU = curves[V]
    k0 = V // L
    dK = deltaK if deltaK else max(1, k0 // 8)
    K0 = even_split(V, L)
    G0, D0 = margins(CU, Kgrid, K0, dK)
    eps = float(np.median(D0)) if eps_drop is None else eps_drop
    tau = float(np.median(G0)) if tau_low is None else tau_low
    K, G, D = allocate(CU, Kgrid, V, dK, eps, tau)
    return {'K': K, 'deltaK': dK, 'eps_drop': eps, 'tau_low': tau,
            'G0': G0, 'D0': D0, 'G': G, 'D': D}


# ---------------------------------------------------------------------------------------------- priors / seeds

def pooled_entropy(sessions):
    header = sessions[0].header
    counts = np.zeros((header['layers'], header['experts']), dtype=np.int64)
    for s in sessions:
        counts += entropy_per_layer(s)[1]
    L = header['layers']
    H = np.zeros(L)
    for layer in range(L):
        c = counts[layer][counts[layer] > 0].astype(np.float64)
        p = c / c.sum()
        H[layer] = -(p * np.log2(p)).sum()
    return H


def entropy_weights(H, amp=1.0):
    z = (H - H.mean()) / (H.std() if H.std() else 1.0)
    return np.exp(amp * z)


def seed_orders(sessions, loo=True):
    """Per session, the domain-key order: experts ranked by usage over the other sessions of its
    category (leave-one-out). This models an admission-time profile lookup keyed long/short."""
    orders = {}
    for s in sessions:
        pool = [o for o in sessions if o.category == s.category and (not loo or o is not s)]
        counts = usage_counts(pool, s.header)
        orders[s.name] = np.lexsort((np.arange(counts.size), -counts))
    return orders


# ---------------------------------------------------------------------------------------------- arms driver

def _sim_job(job):
    name, V, arm, kwargs = job[:4]
    mode = job[4] if len(job) > 4 else 'fixed'
    r = simulate(_SESS_CACHE[name], vram=V, **kwargs)
    return {'session': name, 'V': int(V), 'arm': arm, 'mode': mode,
            'misses_per_token': r['misses_per_token'],
            'generated_tokens': r['generated_tokens'], 'misses': r['live_vram_misses'],
            'hit_rate': r['hit_rate']}


def _run(jobs, trace_dir, workers):
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_sessions, initargs=(trace_dir,)) as pool:
            return list(pool.map(_sim_job, jobs))
    return [_sim_job(j) for j in jobs]


def _pooled(rows):
    toks = sum(r['generated_tokens'] for r in rows)
    return sum(r['misses'] for r in rows) / toks if toks else 0.0


def cmd_arms(args):
    _init_sessions(args.trace_dir)
    sessions = [_SESS_CACHE[n] for n in SESSIONS]
    header = sessions[0].header
    L = header['layers']
    sweep = [int(x) for x in args.sweep.split(',')]
    deployed_V = sorted({s.frames for s in sessions})
    all_V = sorted(set(deployed_V) | set(sweep))

    # ---- entropy prior -------------------------------------------------------
    H = pooled_entropy(sessions)
    Hper = {s.name: entropy_per_layer(s)[0] for s in sessions}
    w_ent = entropy_weights(H, args.amp)
    w_perm = np.random.default_rng(17).permutation(w_ent)
    print(f'per-layer routing entropy H_l: pooled range {H.min():.3f}..{H.max():.3f} bits, '
          f'std {H.std():.4f}; session std mean {np.mean([h.std() for h in Hper.values()]):.4f}')
    print('H_l = ' + ' '.join(f'{h:.2f}' for h in H))

    # ---- capacity response curves + allocator --------------------------------
    curves = response_curves(all_V, SESSIONS, args.trace_dir, jobs=args.jobs)
    alloc = {V: allocate_at(V, curves, L) for V in all_V}
    Kvec = {V: alloc[V]['K'] for V in all_V}

    # ---- seed orders ---------------------------------------------------------
    orders = seed_orders(sessions, loo=not args.seed_inclusive)
    share = args.seed_share

    def seed_keys(name, V):
        return orders[name][:max(1, int(round(share * V)))]

    # ---- columns: deployed (header frames) + the per-round profile + the sweep ----
    columns = ([('deployed', lambda s: s.frames, 'fixed'),
                ('profile', lambda s: s.frames, 'profile')]
               + [(str(v), (lambda s, v=v: v), 'fixed') for v in sweep])

    amps = list(dict.fromkeys([0.25, 0.5, args.amp]))
    w_by_amp = {a: entropy_weights(H, a) for a in amps}
    w_ent = w_by_amp[args.amp]
    w_perm = np.random.default_rng(17).permutation(w_ent)
    print(f'entropy weights at amp {args.amp}: {w_ent.min():.3f}..{w_ent.max():.3f}')

    def kwargs_for(arm, s, V):
        if arm == 'sequential':
            return {'atomic': False}
        if arm == 'quota_even':
            return {'quota': even_split(V, L)}
        if arm == 'quota':
            return {'quota': Kvec[V]}
        if arm.startswith('w_ent'):
            return {'weights': w_by_amp[float(arm[5:])]}
        if arm == 'w_permuted':
            return {'weights': w_perm}
        if arm == 'seed':
            return {'seed_keys': seed_keys(s.name, V), 'seed_count': args.seed_count}
        if arm == 'seed+w_entropy':
            return {'seed_keys': seed_keys(s.name, V), 'seed_count': args.seed_count,
                    'weights': w_by_amp[args.amp]}
        return {}

    arms = (['uniform', 'sequential', 'quota_even', 'quota']
            + [f'w_ent{a:g}' for a in amps] + ['w_permuted', 'seed', 'seed+w_entropy'])
    def job_kwargs(arm, s, V, mode):
        kw = kwargs_for(arm, s, V)
        return {**kw, 'per_round': True} if mode == 'profile' else kw

    jobs = [(s.name, getV(s), arm, job_kwargs(arm, s, getV(s), mode), mode)
            for arm in arms for _label, getV, mode in columns for s in sessions]
    results = _run(jobs, args.trace_dir, args.jobs)

    # ---- aggregate -----------------------------------------------------------
    def label_of(r):
        if r['mode'] == 'profile':
            return 'profile'
        return 'deployed' if r['V'] == _SESS_CACHE[r['session']].frames else str(r['V'])

    table = {}   # (arm, label) -> pooled
    per = {}     # (arm, label) -> {session: m/t}
    for r in results:
        per.setdefault((r['arm'], label_of(r)), {})[r['session']] = r['misses_per_token']
    for arm in arms:
        for label, _getV, _mode in columns:
            rows = [r for r in results if r['arm'] == arm and label_of(r) == label]
            table[(arm, label)] = _pooled(rows)

    print(f'\n=== pooled misses per generated token (event-atomic unless stated) ===')
    hdr = 'arm'.ljust(16) + ''.join(f'{l:>12s}' for l, *_ in columns)
    print(hdr)
    for arm in arms:
        ctrl_dev = table[('uniform', 'deployed')]
        ctrl = table[('uniform', 'deployed')]
        cells = ''.join(f'{table[(arm, l)]:>12.3f}' for l, *_ in columns)
        rel = 100 * (table[(arm, 'deployed')] / ctrl_dev - 1)
        print(f'{arm:16s}{cells}   ({rel:+.2f}% at deployed)')
    print('\nper-session deployed misses/token:')
    for arm in arms:
        d = per[(arm, 'deployed')]
        print(f'  {arm:16s} ' + ' '.join(f'{s}={d[s]:.2f}' for s in SESSIONS))

    # ---- capacity vectors ----------------------------------------------------
    rep = deployed_V[0]
    print(f'\nquota K_l at V={rep} (sum {Kvec[rep].sum()}): min {Kvec[rep].min()} max {Kvec[rep].max()} '
          f'mean {Kvec[rep].mean():.1f}')
    print('  K_l = ' + ' '.join(str(x) for x in Kvec[rep]))
    G0, D0 = alloc[rep]['G0'], alloc[rep]['D0']
    print(f'  allocator @V={rep}: dK={alloc[rep]["deltaK"]} eps_drop={alloc[rep]["eps_drop"]:.5f} '
          f'tau_low={alloc[rep]["tau_low"]:.5f}; even-split G+ {G0.min():.5f}..{G0.max():.5f} '
          f'D- {D0.min():.5f}..{D0.max():.5f}')

    # ---- sensitivity of arm (a) ----------------------------------------------
    variants = []
    for name, v in (('dK/2', 'dK2'), ('dK', 'dK'), ('dK*2', 'dKx2')):
        variants.append((f'deltaK {name}', v, None, None))
    for name, v in (('0', 0.0), ('median', 'med'), ('0.01', 0.01)):
        variants.append((f'eps_drop {name}', None, v, None))
    for name, v in (('0', 0.0), ('median', 'med'), ('0.01', 0.01)):
        variants.append((f'tau_low {name}', None, None, v))
    Kvar = {}
    for label, dk_mode, eps, tau in variants:
        Ks = {}
        for V in deployed_V:
            k0 = V // L
            dK = {'dK2': max(1, k0 // 16), 'dKx2': max(1, k0 // 4)}.get(dk_mode or '', None)
            default = allocate_at(V, curves, L)
            e = default['eps_drop'] if eps == 'med' else eps
            t = default['tau_low'] if tau == 'med' else tau
            Ks[V] = allocate_at(V, curves, L, deltaK=dK, eps_drop=e, tau_low=t)['K']
        Kvar[label] = Ks
    sens_jobs = [(s.name, s.frames, 'quota', {'quota': Kvar[label][s.frames]})
                 for label, *_ in variants for s in sessions]
    sens = _run(sens_jobs, args.trace_dir, args.jobs)
    print('\narm (a) threshold sensitivity (pooled deployed misses/token):')
    base = table[('uniform', 'deployed')]
    sens_by = {label: [] for label, *_ in variants}
    idx = 0
    for label, *_ in variants:
        for _s in sessions:
            sens_by[label].append(sens[idx])
            idx += 1
    for label, *_ in variants:
        val = _pooled(sens_by[label])
        print(f'  {label:18s} {val:>10.3f}  ({100 * (val / base - 1):+.2f}%)')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'calibration_note': 'uniform arm matches host_tier --vram-budget recorded',
                       'entropy': {'pooled': H.tolist(), 'weights': w_ent.tolist()},
                       'table': {a: {l: table[(a, l)] for l, *_ in columns} for a in arms},
                       'per_session_deployed': {a: per[(a, 'deployed')] for a in arms},
                       'capacity_vectors': {str(V): Kvec[V].tolist() for V in all_V},
                       'allocator': {str(V): {'deltaK': alloc[V]['deltaK'], 'eps_drop': alloc[V]['eps_drop'],
                                              'tau_low': alloc[V]['tau_low']} for V in all_V},
                       'sensitivity': {label: _pooled(sens_by[label]) for label, *_ in variants}}, f, indent=1)
        print(f'\nwrote {args.json}')
    return 0


# ---------------------------------------------------------------------------------------------- CLI

def cmd_calibrate(args):
    """Calibration gate: uniform-LFRU event-atomic vs host_tier.py --vram-budget recorded."""
    sessions = load_sessions(trace_dir=args.trace_dir)
    rows, ok = [], True
    my_misses = my_tokens = ref_misses = 0
    for s in sessions:
        v = args.vram or s.frames
        mine = simulate(s, vram=v)
        ref = reference_host_tier(s, v)
        rel = abs(mine['misses_per_token'] - ref['misses_per_token']) / ref['misses_per_token']
        rows.append((s.name, v, mine['misses_per_token'], ref['misses_per_token'], rel))
        my_misses += mine['live_vram_misses']; my_tokens += mine['generated_tokens']
        ref_misses += ref['misses_per_token'] * ref['generated_tokens']
        ok &= rel <= 1e-3
    print(f'{"session":9s} {"V":>6s} {"prototype":>12s} {"host_tier":>12s} {"rel":>9s}')
    for name, v, a, b, rel in rows:
        print(f'{name:9s} {v:6d} {a:12.6f} {b:12.6f} {100 * rel:8.4f}%')
    mp, rp = my_misses / my_tokens, ref_misses / my_tokens
    print(f'\npooled prototype {mp:.6f}  host_tier {rp:.6f}  rel {100 * abs(mp - rp) / rp:.4f}%')
    print(f'gate {"PASSED" if ok else "FAILED"} (max per-session rel {100 * max(r[4] for r in rows):.4f}%)')
    return 0 if ok else 1


def reference_host_tier(session, vram):
    """host_tier.TierSim full-RAM at the trace frames, recorded budgets, empty prior."""
    from host_tier import TierSim
    sim = TierSim(session.header, vram=vram, ram_slots=None, vram_budget='recorded', prior='none')
    sim.run(session.rounds)
    gen_tokens = live = hits = 0
    for m in sim.metrics:
        for kind in GENERATION_KINDS:
            gen_tokens += m.by_kind[kind]['tokens']
            live += m.by_kind[kind]['live_keys']
            hits += m.by_kind[kind]['live_vram_hits']
    misses = live - hits
    return {'misses_per_token': misses / gen_tokens if gen_tokens else 0.0,
            'generated_tokens': gen_tokens,
            'hit_rate': hits / live if live else 0.0}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('calibrate', help='run the calibration gate against host_tier.py')
    p.add_argument('--vram', type=int, default=0, help='counts; 0 = each trace header frames')
    p.add_argument('--trace-dir', default=TRACE_DIR)
    p.set_defaults(func=cmd_calibrate)
    a = sub.add_parser('arms', help='run every arm across the deployed capacity and the sweep')
    a.add_argument('--trace-dir', default=TRACE_DIR)
    a.add_argument('--sweep', default='4008,6347,8000,8351,10200,12000')
    a.add_argument('--amp', type=float, default=1.0, help='entropy-weight amplitude (per entropy std)')
    a.add_argument('--seed-share', type=float, default=0.5, help='warm-start seed share of frames')
    a.add_argument('--seed-count', type=int, default=16, help='seeded LFRU count cap')
    a.add_argument('--seed-inclusive', action='store_true', help='seed profile includes the session')
    a.add_argument('--jobs', type=int, default=1)
    a.add_argument('--json', default=None)
    a.set_defaults(func=cmd_arms)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
