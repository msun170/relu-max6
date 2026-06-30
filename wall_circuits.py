# !!! RETRACTED RESULTS WARNING (2026-07-01) !!! build_blocks() below generates an INCOMPLETE block family, so the
# multi-weight "J+NB infeasible / Type-II=0" outputs from this script are FALSE NEGATIVES (an incomplete family makes
# even known-IN cases look infeasible -- proven by wall_poscontrol.py: max_5,6 are FEASIBLE on the COMPLETE weight-2
# family but build_blocks reports them INFEASIBLE). This file is kept ONLY as the method skeleton. For a valid (sound)
# wall test use a COMPLETE family (see wall_poscontrol.py). Do NOT cite this script's weight 2/3/4 table.
#
# WALL-CIRCUIT MINING (user plan). Instead of function-value membership, decompose representability into WALL jumps.
# A homogeneous PL function's gradient jumps across hyperplanes; the "jump" of orbit-sum OS_k across a probe (x0,d)
# is the second difference  OS_k(x0+d) + OS_k(x0-d) - 2 OS_k(x0).  Split probes by direction d:
#   J  rows: d ~ e_i - e_j (BRAID).   target = max_n's braid jumps (nonzero).
#   NB rows: d NON-braid.             target = 0  (max_n is smooth across non-braid dirs => must cancel).
# Exact representation in a family  <=>  exists block-coeff vector c with  J c = jt  AND  NB c = 0.
# We compute, per family, the feasibility table: J alone, NB alone, J+NB; and the Type-II circuits (NB c=0, J c!=0)
# and whether the max_n braid target lies in their image. All exact (integer -> mod-p).
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = int(os.environ.get("WC_N", "7")); t0 = time.time()
P1 = 2147483647
roots = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a != b}
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
braidset = {prim(r) for r in roots}
def is_braid(d): return prim(d) in braidset

def lattice(W):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
    return L, set(L), {p: i for i, p in enumerate(L)}
def gens_of(L, Ls):
    g = set()
    for p in L:
        for q in L:
            if p == q: continue
            d = prim(tuple(q[k]-p[k] for k in range(n))); g.add(d)
    return sorted(g)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def build_blocks(W, complete, n_sample, seed):
    L, Ls, vi = lattice(W); gens = gens_of(L, Ls)
    rng = np.random.default_rng(seed)
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
        # complete 1- and 2-gen zonotopes + pairwise joins (<= cap verts)
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
        seen = set(); stall = 0
        while len(blocks) < n_sample and stall < 20000:
            u = zono() | zono()
            if len(u) > 6: stall += 1; continue
            key = tuple(sorted(vi[p] for p in u))
            if key in seen: stall += 1; continue
            seen.add(key); blocks.add(frozenset(u)); stall = 0
    # dedup by S_n canon to get orbit reps (one column per orbit)
    import core
    reps = {}
    for b in blocks: reps.setdefault(core.canon(b, n), b)
    return L, vi, list(reps.values()), gens

def run_family(W, complete=False, n_sample=1500, seed=7, K_J=300, K_NB=400, nb_dirs=120):
    L, vi, blocks, gens = build_blocks(W, complete, n_sample, seed)
    NP = len(L); N = len(blocks)
    Lar = np.array(L, dtype=np.int64)
    perms = list(itertools.permutations(range(n))); G = len(perms)
    pa = np.empty((G, NP), dtype=np.int32)
    for gi, g in enumerate(perms):
        for i, p in enumerate(L): pa[gi, i] = vi[tuple(p[g[k]] for k in range(n))]
    pa = cp.asarray(pa)
    # ---- probes ----
    rng = np.random.default_rng(123 + W)
    nb_pool = [g for g in gens if not is_braid(g)]
    rng.shuffle(nb_pool); nb_pool = nb_pool[:nb_dirs] or [g for g in gens if not is_braid(g)][:nb_dirs]
    probes = []  # (x0, d, is_braid_flag)
    # J probes: two-way tie at the top, d = e_i - e_j
    for _ in range(K_J):
        i, j = rng.choice(n, size=2, replace=False)
        x = rng.integers(-12, 0, size=n); t = int(rng.integers(3, 12)); x[i] = t; x[j] = t
        d = np.zeros(n, dtype=np.int64); d[i] = 1; d[j] = -1
        probes.append((x.astype(np.int64), d, True))
    # NB probes: unique max with big margin so max_n smooth across non-braid d
    for _ in range(K_NB):
        d = np.array(nb_pool[rng.integers(len(nb_pool))], dtype=np.int64)
        x = rng.integers(-6, 7, size=n).astype(np.int64)
        top = int(rng.integers(0, n)); x[top] = int(x.max()) + int(np.abs(d).sum()) + int(rng.integers(3, 8))
        probes.append((x, d, False))
    # build probe point set Y and second-difference operator
    Ypts = []; trip = []
    for (x, d, fl) in probes:
        i0 = len(Ypts); Ypts.append(x + d); Ypts.append(x - d); Ypts.append(x)
        trip.append((i0, i0+1, i0+2, fl))
    Y = np.array(Ypts, dtype=np.int64)
    PY = cp.asarray((Y @ Lar.T).astype(cp.int64))   # |Y| x NP
    # orbit-sum columns at Y
    def col_at_Y(Vidx):
        Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx); idx = pa[:, Vc]
        return PY[:, idx.reshape(-1)].reshape(len(Y), G, k).max(axis=2).sum(axis=1)
    cols = cp.stack([col_at_Y([vi[p] for p in b]) for b in blocks], axis=1)   # |Y| x N
    # wall-jump matrix rows (second differences); split J / NB
    Jrows = []; NBrows = []; jt = []
    bmaxY = cp.asarray(Y.max(axis=1).astype(cp.int64))
    for (ip, im, i0, fl) in trip:
        row = cols[ip] + cols[im] - 2*cols[i0]              # length N
        tgt = int(bmaxY[ip] + bmaxY[im] - 2*bmaxY[i0])
        if fl: Jrows.append(row); jt.append(tgt)
        else:
            assert tgt == 0, f"NB probe has nonzero max_n jump {tgt}"
            NBrows.append(row)
    Jm = cp.stack(Jrows, axis=0); NBm = cp.stack(NBrows, axis=0); jt = cp.asarray(jt, dtype=cp.int64)
    # exact rank / feasibility over GF(p): does target vector t lie in column-space of M (i.e. exists c, M c = t)?
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
    def feasible(M, t):   # exists c with M c = t ?  <=> rank([M|t]) == rank(M)  (t appended as a column)
        rM = rank_modp(M)
        rMt = rank_modp(cp.concatenate([M, t.reshape(-1, 1)], axis=1))
        return rM, rMt, (rM == rMt)
    rJ, rJt, fJ = feasible(Jm, jt)
    Z = cp.zeros(NBm.shape[0], dtype=cp.int64)
    rNB = rank_modp(NBm)
    M = cp.concatenate([NBm, Jm], axis=0); tt = cp.concatenate([Z, jt])
    rM, rMt, fJNB = feasible(M, tt)
    # IN-FAMILY CONTROL: target = wall profile of a random in-family combo c0 -> MUST be feasible (c0 solves it).
    c0 = cp.asarray(np.random.default_rng(99).integers(-4, 5, size=N).astype(np.int64))
    tctrl = M @ c0
    rC, rCt, fCtrl = feasible(M, tctrl)
    # RANDOM control: a generic target should be infeasible (sanity that feasibility isn't vacuous).
    trand = cp.asarray(np.random.default_rng(7).integers(-50, 51, size=M.shape[0]).astype(np.int64))
    _, _, fRand = feasible(M, trand)
    print(f"\n=== weight {W} ({'COMPLETE' if complete else 'sampled '+str(N)+' orbits'}) | probes: {len(jt)} braid, {NBm.shape[0]} non-braid | N={N} blocks ===", flush=True)
    print(f"  J  alone  : rank(J)={rJ}, rank([J|jt])={rJt}  -> braid target {'FEASIBLE' if fJ else 'infeasible'}", flush=True)
    print(f"  NB alone  : rank(NB)={rNB}  (ker dim in block space = {N-rNB}; non-braid-cancelling combos exist)", flush=True)
    print(f"  J + NB    : rank(M)={rM}, rank([M|t])={rMt}  -> {'FEASIBLE (max_n walls representable here!)' if fJNB else 'INFEASIBLE => OUT by wall obstruction'}", flush=True)
    print(f"  Type-II   : dim = rank([NB;J]) - rank(NB) = {rM - rNB}  (>0 needed to build any braid wall non-braid-cleanly)", flush=True)
    print(f"  CONTROLS  : in-family target {'FEASIBLE(ok)' if fCtrl else 'INFEASIBLE -- BUG!!'} | random target {'INFEASIBLE(ok)' if not fRand else 'FEASIBLE(VACUOUS!!)'}", flush=True)
    print(f"  VALID = {fCtrl and (not fRand)} (machinery sound iff in-family feasible & random infeasible)", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    return fJ, fJNB

import os
WHICH = os.environ.get("WC_WHICH", "all")
if WHICH in ("all", "2"): run_family(2, complete=True, K_J=400, K_NB=500, nb_dirs=200)
if WHICH in ("all", "3"): run_family(3, complete=False, n_sample=2500, K_J=400, K_NB=600, nb_dirs=250)
if WHICH in ("all", "4"): run_family(4, complete=False, n_sample=2500, K_J=400, K_NB=700, nb_dirs=300)
print(f"\n[{time.time()-t0:.0f}s] PATTERN to look for: J feasible, J+NB infeasible at EVERY weight => stable wall obstruction.", flush=True)
