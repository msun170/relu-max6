# MAX7 CONSTRUCTION SEARCH (IN direction). Goal: an EXACT rational identity  D*max_7 = sum_t a_t OrbitSum(Q_t) + b*Sx
# with each Q_t a join of two zonotopes (a 2-layer block). Strategy here: test COMPLETE weight-4 P2 sub-families by
# VERTEX CAP (a complete family => a sound finite OUT per the hard rule; an exact IN => a construction to verify).
# For <=k vertices, EVERY k-subset of the weight-4 lattice is a join of two zonotopes (any pair is a segment, any
# point a zonotope), so the complete <=k-vertex P2 family = all <=k-subsets of the 210 weight-4 points (orbit-summed).
# <=2v, <=3v are feasible; <=4v ~ 1.5e4-2.9e4 orbits = the enumeration wall.
import sys, itertools, time, os
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; W = 4; t0 = time.time(); P1 = 2147483647; P2 = 2147483629
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; NP = len(W4)
vi = {p: i for i, p in enumerate(W4)}
perms = list(itertools.permutations(range(n))); G = len(perms)
pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in W4] for g in perms], dtype=np.int32))
m = int(os.environ.get("M", "6000"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
Pi = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(cp.int64))
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
Xs = np.sort(X, axis=1); bsym = cp.asarray((Xs * np.random.default_rng(5).integers(-9, 10, size=n)).sum(axis=1).astype(np.int64))
lin = cp.asarray(X.astype(cp.int64))

def canon_set(S):  # S = tuple of point-index; canonical under S_n via core.canon on the point tuples
    return core.canon(frozenset(W4[i] for i in S), n)
def grp_col(Sidx):
    Vc = cp.asarray(list(Sidx), dtype=cp.int32); k = len(Sidx); idx = pa[:, Vc]
    return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)
def rank_modp(A, Pm):
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

def test_level(maxv, NCAP=7000):
    # enumerate ALL orbit reps of <=maxv-subsets of the 210 weight-4 points
    reps = {}
    for k in range(1, maxv+1):
        cnt = 0
        for S in itertools.combinations(range(NP), k):
            key = canon_set(S)
            if key not in reps: reps[key] = S
            cnt += 1
            if len(reps) > NCAP:
                print(f"  <= {maxv} verts: orbit count exceeds NCAP={NCAP} (enumeration wall) at k={k} after {cnt} subsets, {len(reps)} orbits  [{time.time()-t0:.0f}s]", flush=True)
                return None
    Nrep = len(reps); print(f"  <= {maxv} verts: COMPLETE family = {Nrep} S_7 orbits  [{time.time()-t0:.0f}s]", flush=True)
    cols = cp.stack([grp_col(S) for S in reps.values()], axis=1)
    RAW = cp.concatenate([cols, lin, bmax.reshape(-1,1), bsym.reshape(-1,1)], axis=1)
    L = Nrep + n
    res = {}
    for Pm in (P1, P2):
        A = RAW % Pm; rows = A.shape[0]; r = 0
        for c in range(L):
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
        rank = r
        mi = bool(cp.all(A[rank:, L] % Pm == 0).item()); si = bool(cp.all(A[rank:, L+1] % Pm == 0).item())
        res[Pm] = (rank, mi, si)
    (r1, mi1, si1) = res[P1]; (r2, mi2, si2) = res[P2]
    verdict = "IN => CONSTRUCTION EXISTS!! (extract+verify)" if (mi1 and mi2) else "OUT of this complete family"
    valid = (r1 < m) and (not si1)
    print(f"    rank={r1}/{m} | max_7 {verdict} | sym-control {'IN(VACUOUS!)' if si1 else 'OUT(ok)'} | valid={valid}  [{time.time()-t0:.0f}s]", flush=True)
    return (mi1 and mi2)

NCAP = int(os.environ.get("NCAP", "7000"))
LEVELS = [int(x) for x in os.environ.get("LEVELS", "2,3,4").split(",")]
print(f"weight-4 lattice {NP} pts, m={m}, NCAP={NCAP}, levels={LEVELS}. COMPLETE <=k-vertex P2 families.", flush=True)
for mv in LEVELS:
    if test_level(mv, NCAP) is True:
        print("  *** exact IN -- stop and extract construction ***", flush=True); break
print(f"[{time.time()-t0:.0f}s] done. (<=4v = the point/segment-join family ~2.9e4 orbits = enumeration wall.)", flush=True)
