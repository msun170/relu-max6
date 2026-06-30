# !!! RETRACTED RESULTS WARNING (2026-07-01) !!! build_blocks() below is an INCOMPLETE family, so this script's
# outputs (single-type culprits, minimal infeasible set, Type-II dim) are NOT valid OUT certificates -- on an
# incomplete family the wall test gives false infeasibility (wall_poscontrol.py shows known-IN max_5,6 come out
# FEASIBLE only on the COMPLETE weight-2 family). Kept as method skeleton only. Do NOT cite these outputs.
#
# WALL-OBSTRUCTION EXTRACTION (user plan, step after the J/NB table). Two questions:
#  (1) WHERE does the obstruction live?  Group NON-braid wall probes by DIRECTION TYPE (S_n-orbit of the primitive
#      normal = sorted coord tuple). Find the MINIMAL set of non-braid direction-types whose constraints, added to J,
#      already force infeasibility. Also test each type alone. This names the wall directions that carry the OUT.
#  (2) Do the SAME culprit types RECUR across weight 2 -> 3 -> 4?  (= a weight-independent obstruction => descent.)
# Plus Type-I (NB c=0 & J c=0, pure cancellation) and Type-II (NB c=0, J c!=0) subspace dimensions.
import sys, itertools, time, os
from math import gcd
from functools import reduce
from collections import Counter
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; t0 = time.time(); P1 = 2147483647
roots = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a != b}
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
braid_type = tuple(sorted(prim(tuple((1 if k==0 else -1 if k==1 else 0) for k in range(n)))))
def dtype(d): return tuple(sorted(prim(d)))
def is_braid(d): return dtype(d) == braid_type
def lattice(W):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; return L, set(L), {p:i for i,p in enumerate(L)}
def gens_of(L):
    g = set()
    for p in L:
        for q in L:
            if p != q: g.add(prim(tuple(q[k]-p[k] for k in range(n))))
    return sorted(g)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def build_blocks(W, complete, n_sample, seed):
    L, Ls, vi = lattice(W); gens = gens_of(L); rng = np.random.default_rng(seed)
    def zono():
        p = L[rng.integers(len(L))]; verts = {p}; used = 0; tries = 0
        while used < 2 and len(verts) < 6 and tries < 30:
            tries += 1; g = gens[rng.integers(len(gens))]; nv = set(); ok = True
            for u in verts:
                w = shift(u, g)
                if w not in Ls: ok = False; break
                nv.add(w)
            if ok and nv - verts: verts |= nv; used += 1
        return verts
    blocks = set()
    if complete:
        zon = set()
        for p in L:
            for g in gens:
                w = shift(p, g)
                if w in Ls: zon.add(frozenset([p, w]))
        for p in L:
            for i in range(len(gens)):
                if shift(p, gens[i]) not in Ls: continue
                for j in range(i+1, len(gens)):
                    a = shift(p, gens[i]); b = shift(p, gens[j]); c = shift(a, gens[j])
                    if b in Ls and c in Ls: zon.add(frozenset([p, a, b, c]))
        zon = list(zon)
        for z in zon: blocks.add(z)
        for i in range(len(zon)):
            for j in range(i, len(zon)):
                u = zon[i] | zon[j]
                if len(u) <= 6: blocks.add(u)
    else:
        seen = set()
        while len(blocks) < n_sample:
            u = zono() | zono()
            if len(u) > 6: continue
            key = tuple(sorted(vi[p] for p in u))
            if key in seen: continue
            seen.add(key); blocks.add(frozenset(u))
    reps = {}
    for b in blocks: reps.setdefault(core.canon(b, n), b)
    return L, vi, list(reps.values())

def rank_modp(A, Pm=P1):
    A = (A % Pm).copy(); rows, ncols = A.shape; r = 0
    for c in range(ncols):
        sub = A[r:, c]; nz = cp.nonzero(sub)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        A[r] = (A[r] * pow(int(A[r, c].item()), Pm-2, Pm)) % Pm
        if r+1 < rows:
            f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= Pm
        r += 1
        if r == rows: break
    return r
def feasible(M, t):
    return rank_modp(M) == rank_modp(cp.concatenate([M, t.reshape(-1,1)], axis=1))

def run(W, complete, n_sample, seed=7, K_J=400, Q_per_type=70, n_types=14):
    L, vi, blocks = build_blocks(W, complete, n_sample, seed); N = len(blocks); NP = len(L)
    Lar = np.array(L, dtype=np.int64)
    perms = list(itertools.permutations(range(n))); G = len(perms)
    pa = np.empty((G, NP), dtype=np.int32)
    for gi, g in enumerate(perms):
        for i, p in enumerate(L): pa[gi, i] = vi[tuple(p[g[k]] for k in range(n))]
    pa = cp.asarray(pa)
    # non-braid direction TYPES present as block edges, by frequency
    cnt = Counter(); rep = {}
    for b in blocks:
        V = list(b)
        for i in range(len(V)):
            for j in range(i+1, len(V)):
                d = prim(tuple(V[i][k]-V[j][k] for k in range(n)))
                if is_braid(d): continue
                t = dtype(d); cnt[t] += 1; rep.setdefault(t, d)
    types = [t for t, _ in cnt.most_common(n_types)]
    rng = np.random.default_rng(123 + W)
    # probes: J (braid tie) + per-type NB groups
    probes = []  # (x0, d, group)  group=-1 for J, else type index
    for _ in range(K_J):
        i, j = rng.choice(n, size=2, replace=False)
        x = rng.integers(-12, 0, size=n); tt = int(rng.integers(3, 12)); x[i] = tt; x[j] = tt
        d = np.zeros(n, dtype=np.int64); d[i] = 1; d[j] = -1
        probes.append((x.astype(np.int64), d, -1))
    for gi, t in enumerate(types):
        d0 = np.array(rep[t], dtype=np.int64)
        for _ in range(Q_per_type):
            x = rng.integers(-6, 7, size=n).astype(np.int64)
            top = int(rng.integers(0, n)); x[top] = int(x.max()) + int(np.abs(d0).sum()) + int(rng.integers(3, 8))
            probes.append((x, d0, gi))
    Ypts = []; trip = []
    for (x, d, gp) in probes:
        i0 = len(Ypts); Ypts.append(x+d); Ypts.append(x-d); Ypts.append(x); trip.append((i0, i0+1, i0+2, gp))
    Y = np.array(Ypts, dtype=np.int64); PY = cp.asarray((Y @ Lar.T).astype(cp.int64))
    def col_at_Y(Vidx):
        Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx); idx = pa[:, Vc]
        return PY[:, idx.reshape(-1)].reshape(len(Y), G, k).max(axis=2).sum(axis=1)
    cols = cp.stack([col_at_Y([vi[p] for p in b]) for b in blocks], axis=1)
    bmaxY = cp.asarray(Y.max(axis=1).astype(cp.int64))
    Jrows = []; jt = []; grp_rows = {gi: [] for gi in range(len(types))}
    for (ip, im, i0, gp) in trip:
        row = cols[ip] + cols[im] - 2*cols[i0]; tgt = int(bmaxY[ip] + bmaxY[im] - 2*bmaxY[i0])
        if gp == -1: Jrows.append(row); jt.append(tgt)
        else:
            assert tgt == 0; grp_rows[gp].append(row)
    Jm = cp.stack(Jrows, axis=0); jt = cp.asarray(jt, dtype=cp.int64)
    NBg = {gi: cp.stack(rows, axis=0) for gi, rows in grp_rows.items() if rows}
    print(f"\n##### weight {W} ({'COMPLETE' if complete else 'sampled '+str(N)} orbits) | {len(types)} non-braid types, {Q_per_type}/type | N={N}", flush=True)
    # sanity: J alone feasible; full infeasible; controls
    allNB = cp.concatenate([NBg[gi] for gi in NBg], axis=0)
    Mfull = cp.concatenate([allNB, Jm], axis=0); tfull = cp.concatenate([cp.zeros(allNB.shape[0], dtype=cp.int64), jt])
    print(f"  J alone feasible={feasible(Jm, jt)} | J+ALL_NB feasible={feasible(Mfull, tfull)} (expect False)", flush=True)
    c0 = cp.asarray(np.random.default_rng(99).integers(-4,5,size=N).astype(np.int64))
    print(f"  CONTROLS: in-family feasible={feasible(Mfull, Mfull@c0)} (exp True) | random feasible={feasible(Mfull, cp.asarray(np.random.default_rng(7).integers(-50,51,size=Mfull.shape[0]).astype(np.int64)))} (exp False)", flush=True)
    # (a) per-type marginal: does a SINGLE type + J already force infeasibility?
    single = []
    for gi in NBg:
        M = cp.concatenate([NBg[gi], Jm], axis=0); t = cp.concatenate([cp.zeros(NBg[gi].shape[0], dtype=cp.int64), jt])
        if not feasible(M, t): single.append(types[gi])
    print(f"  (a) SINGLE non-braid types that ALONE+J force infeasible: {len(single)} of {len(NBg)}", flush=True)
    for t in single[:8]: print(f"        culprit type {t}  (e.g. normal {rep[t]})", flush=True)
    # (b) greedy minimal infeasible type-set
    keep = list(NBg.keys())
    def infeasible_with(groups):
        if not groups:
            return not feasible(Jm, jt)
        NB = cp.concatenate([NBg[g] for g in groups], axis=0); M = cp.concatenate([NB, Jm], axis=0)
        t = cp.concatenate([cp.zeros(NB.shape[0], dtype=cp.int64), jt]); return not feasible(M, t)
    assert infeasible_with(keep)
    for gi in list(keep):
        trial = [g for g in keep if g != gi]
        if infeasible_with(trial): keep = trial
    print(f"  (b) MINIMAL infeasible non-braid type-set: {len(keep)} types -> {[types[g] for g in keep]}", flush=True)
    # Type-I / Type-II dims
    S = cp.concatenate([Jm, allNB], axis=0); rS = rank_modp(S); rNB = rank_modp(allNB)
    typeI = N - rS; typeII = rS - rNB
    print(f"  Type-I (pure cancel, J&NB=0) dim={typeI} | Type-II (NB=0, J!=0) dim={typeII} | rank(J;NB)={rS}", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    return [types[g] for g in keep], single

WHICH = os.environ.get("WX_WHICH", "2")
if WHICH == "2": run(2, True, 0, K_J=400, Q_per_type=80, n_types=14)
if WHICH == "3": run(3, False, 2500, K_J=400, Q_per_type=70, n_types=16)
if WHICH == "4": run(4, False, 2500, K_J=400, Q_per_type=70, n_types=18)
