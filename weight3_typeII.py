# STEP 3: does the IN mechanism reappear at higher weight for n=7? = is Type-II > 0 (a non-braid-clean combination
# with NONZERO braid effect = a full-support NB-circuit producing braid walls) on the COMPLETE weight-w family?
# Type-II dim = rank([NB; J]) - rank(NB), NB/J = non-braid/braid wall-jump matrices over the orbit-sum blocks.
# We validated this on COMPLETE weight-2 (n=5,6 -> Type-II=1; n=7 -> 0). Here: n=7 complete weight-3 (the test),
# with n=6 complete weight-3 as positive control (expect Type-II>0, since max_6 IS representable).
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
P1 = 2147483647
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)

def build_complete(n, Wt, cap):
    L = [c for c in itertools.product(range(Wt+1), repeat=n) if sum(c) == Wt]; Ls = set(L); vi = {p:i for i,p in enumerate(L)}
    gens = sorted({prim(tuple(q[k]-p[k] for k in range(n))) for p in L for q in L if p != q})
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    def ssz(p, Gs):
        verts = set()
        for s in range(len(Gs)+1):
            for S in itertools.combinations(Gs, s):
                q = p
                for g in S: q = shift(q, g)
                if q not in Ls: return None
                verts.add(q)
        return frozenset(verts)
    zon = set()
    for p in L:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: zon.add(v)
    for p in L:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in Ls: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: zon.add(v)
    for zz in [zz for zz in list(zon) if len(zz) >= 4]:
        for g in gens:
            verts = set(zz); ok = True
            for u in zz:
                wv = shift(u, g)
                if wv not in Ls: ok = False; break
                verts.add(wv)
            if ok and len(verts) > len(zz): zon.add(frozenset(verts))
    zon = list(zon); zorb = core.orbits_of(zon, n); zreps = [zorb[k][0] for k in zorb]
    bl = set(zon)
    for a in zreps:
        for b in zon:
            u = a | b
            if len(u) <= cap: bl.add(u)
    orb = core.orbits_of(list(bl), n); reps = list(orb)
    return L, vi, [orb[k][0] for k in reps]

def typeII(n, Wt, cap, KJ=500, KNB=700, nbtypes=200):
    t0 = time.time()
    L, vi, reps = build_complete(n, Wt, cap); NP = len(L); Nb = len(reps)
    print(f"n={n} weight-{Wt}: {NP} lattice pts, {Nb} orbit blocks  [{time.time()-t0:.0f}s]", flush=True)
    perms = list(itertools.permutations(range(n))); G = len(perms)
    pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in L] for g in perms], dtype=np.int32))
    braid_t = tuple(sorted(prim(tuple((1 if k==0 else -1 if k==1 else 0) for k in range(n)))))
    # probe points
    rng = np.random.default_rng(40+n+Wt); probes = []   # (x0,d,braid?)
    for _ in range(KJ):
        i, j = rng.choice(n, size=2, replace=False)
        x = rng.integers(-12, 0, size=n).astype(np.int64); t = int(rng.integers(3, 12)); x[i]=t; x[j]=t
        d = np.zeros(n, dtype=np.int64); d[i]=1; d[j]=-1; probes.append((x, d, True))
    nbd = sorted({prim(tuple(list(vs)[a][q]-list(vs)[b][q] for q in range(n)))
                  for vs in reps for a in range(len(vs)) for b in range(a+1, len(vs))
                  if tuple(sorted(prim(tuple(list(vs)[a][q]-list(vs)[b][q] for q in range(n))))) != braid_t})
    rng.shuffle(nbd); nbd = nbd[:nbtypes]
    for _ in range(KNB):
        d = np.array(nbd[rng.integers(len(nbd))], dtype=np.int64)
        x = rng.integers(-6, 7, size=n).astype(np.int64); top = int(rng.integers(0, n))
        x[top] = int(x.max()) + int(np.abs(d).sum()) + int(rng.integers(3, 9)); probes.append((x, d, False))
    Yp = []; trip = []
    for (x, d, fl) in probes:
        i0 = len(Yp); Yp += [tuple(int(t) for t in x+d), tuple(int(t) for t in x-d), tuple(int(t) for t in x)]; trip.append((i0, i0+1, i0+2, fl))
    Y = np.array(Yp, dtype=np.int64); PY = cp.asarray((Y @ np.array(L, dtype=np.int64).T).astype(cp.int64))
    def col(vs):
        Vc = cp.asarray([vi[p] for p in vs], dtype=cp.int32); k = len(vs); idx = pa[:, Vc]
        return PY[:, idx.reshape(-1)].reshape(len(Y), G, k).max(axis=2).sum(axis=1)
    cols = cp.stack([col(vs) for vs in reps], axis=1)
    bmaxY = cp.asarray(Y.max(axis=1).astype(cp.int64))
    Jr=[]; NBr=[]; jt=[]
    for (ip, im, i0, fl) in trip:
        row = cols[ip]+cols[im]-2*cols[i0]; tgt = int(bmaxY[ip]+bmaxY[im]-2*bmaxY[i0])
        if fl: Jr.append(row); jt.append(tgt)
        else: NBr.append(row)
    Jm = cp.stack(Jr); NBm = cp.stack(NBr); jt = cp.asarray(jt, dtype=cp.int64)
    def rk(A, Pm=P1):
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
    rNB = rk(NBm); rNBJ = rk(cp.concatenate([NBm, Jm], axis=0))
    typeII_dim = rNBJ - rNB
    # is max_n's braid target reachable by a non-braid-clean combo? (membership of jt in J(ker NB))
    M = cp.concatenate([NBm, Jm], axis=0); tt = cp.concatenate([cp.zeros(NBm.shape[0], dtype=cp.int64), jt])
    feas = rk(M) == rk(cp.concatenate([M, tt.reshape(-1,1)], axis=1))
    print(f"  rank(NB)={rNB} rank([NB;J])={rNBJ} | TYPE-II dim = {typeII_dim}  "
          f"({'MECHANISM PRESENT (non-braid-clean braid circuits exist)' if typeII_dim>0 else 'NO mechanism (Type-II=0)'})", flush=True)
    print(f"  max_{n} braid target reachable non-braid-cleanly: {feas}  "
          f"({'=> could be IN' if feas else '=> OUT (target outside J(ker NB))'})  [{time.time()-t0:.0f}s]", flush=True)
    return typeII_dim

WHICH = os.environ.get("W3", "n7w3")
if WHICH == "n7w2": typeII(7, 2, 12)
if WHICH == "n7w3": typeII(7, 3, 16)
if WHICH == "n6w3": typeII(6, 3, 16)
if WHICH == "n7w4": typeII(7, 4, 8, nbtypes=120)
