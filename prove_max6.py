"""Exact computer-assisted proof that max(x1,...,x6) is computable in 2 hidden layers.

The construction C is a signed sum of weight-2 P2 support functions (built and solved via core.py),
hence a 2-hidden-layer ReLU network. We prove C == max6 exactly, with no floating point in the
certificate:

  Reduction. C's gradient on a region is sum_t coef_t * argmax(Q_t); it changes only across the
  "weight-2 difference" hyperplanes g - g' = 0 (g, g' vertices of one term). By S6-symmetry we restrict
  to the chamber x1>=...>=x6 (where max6 = x1). In gap coordinates y_k = x_k - x_{k+1} > 0 most of these
  hyperplanes are forced (constant sign), leaving a small arrangement.

  (1) Complete deterministic enumeration of every cell of that arrangement in y>0, each with an exact
      integer interior witness.
  (2) Exact gradient: grad C = D * e1 on every cell (exact integer arithmetic).
  (3) Exact completeness (adjacency closure): every single-flip neighbor of every cell is either an
      enumerated cell or provably empty via an exact rational Gordan certificate. Regions of an
      arrangement in the connected cone y>0 form a connected adjacency graph, so a nonempty closed set
      is all of it.

Hence C == x1 == max6 on the chamber, and by continuity and S6-symmetry on all of R^6.
"""
import time, sys
import numpy as np
from fractions import Fraction as F
from math import gcd
from scipy.optimize import linprog
import construction

n = 6; t0 = time.time()
sys.setrecursionlimit(10000)

# load the construction (signed sum of weight-2 P2 support functions) and its individual terms
flatcoef, flatV, D, L = construction.flat_terms()
DL = np.array([int(x*D) for x in L], dtype=np.int64)
print(f"construction: {len(construction.load()[0])} orbit terms -> {len(flatV)} P2 terms, D={D} [{time.time()-t0:.0f}s]", flush=True)

# relevant hyperplanes: differences g - g' of co-vertices in some term, reduced to integer normals
def canon_normal(d):
    g = 0
    for x in d: g = gcd(g, abs(x))
    if g == 0: return None
    d = tuple(x//g for x in d)
    for x in d:
        if x != 0: return tuple(-y for y in d) if x < 0 else d
    return None
H = set()
for V in flatV:
    for a in range(len(V)):
        for b in range(a+1, len(V)):
            cn = canon_normal(tuple(int(V[a][k]-V[b][k]) for k in range(n)))
            if cn: H.add(cn)

# gap coordinates y_k = x_k - x_{k+1} > 0. A normal h maps to w_k = sum_{i<=k} h_i (k=1..5); h is
# "forced" (constant sign on y>0) iff all w_k share a sign. Keep the unforced ones.
def canon_w(wv):
    g = 0
    for x in wv: g = gcd(g, abs(x))
    if g == 0: return None
    wv = tuple(x//g for x in wv)
    for x in wv:
        if x != 0: return tuple(-y for y in wv) if x < 0 else wv
    return None
WH = set()
for h in H:
    wv = tuple(sum(h[:k+1]) for k in range(5))
    if any(x > 0 for x in wv) and any(x < 0 for x in wv):
        cw = canon_w(wv)
        if cw: WH.add(cw)
W = np.array(sorted(WH), dtype=np.int64); M = W.shape[0]
T = np.zeros((6, 5), dtype=np.int64)              # y -> x:  x_i = sum_{k>=i} y_k,  x_6 = 0
for i in range(5):
    for k in range(i, 5): T[i, k] = 1
print(f"relevant difference hyperplanes: {len(H)}; unforced in y-space: {M} [{time.time()-t0:.0f}s]", flush=True)

# complete deterministic cell enumeration with exact integer witnesses
BIG = 1.0e7
def lp_point(signs, mins=False):
    Aub = []; bub = []
    for k in range(5):
        r = [0.0]*5; r[k] = -1.0; Aub.append(r); bub.append(-1.0)
    for idx, sg in signs:
        Aub.append([-sg*float(W[idx, k]) for k in range(5)]); bub.append(-1.0)
    res = linprog(c=([1.0]*5 if mins else [0.0]*5), A_ub=Aub, b_ub=bub, bounds=[(1.0, BIG)]*5, method='highs')
    return res.x if res.success else None
def int_witness(signs):
    p = lp_point(signs, mins=True)
    if p is None: return None
    for scale in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576):
        yi = np.round(np.asarray(p)*scale).astype(np.int64)
        if np.all(yi >= 1) and all(sg*int(W[idx] @ yi) > 0 for idx, sg in signs): return T @ yi
    return None
cells = []; witsX = []
def dfs(d, signs):
    if d == M:
        x = int_witness(signs)
        if x is None: raise RuntimeError("cell with no integer witness")
        cells.append(tuple(sg for _, sg in signs)); witsX.append(x); return
    for sg in (1, -1):
        ns = signs + [(d, sg)]
        if lp_point(ns) is not None: dfs(d+1, ns)
dfs(0, [])
NC = len(cells)
print(f"complete cell enumeration: {NC} cells [{time.time()-t0:.0f}s]", flush=True)

# exact gradient check: grad C must equal e1 on every cell, i.e. sum_t coef_t*argmax = D*e1 - D*L
Xw = np.array(witsX, dtype=np.int64)
grad = np.zeros((NC, 6), dtype=np.int64)
for V, c in zip(flatV, flatcoef):
    grad += c * V[np.argmax(Xw @ V.T, axis=1)]
target = np.tile(np.array([D, 0, 0, 0, 0, 0], dtype=np.int64) - DL, (NC, 1))
bad = int(np.count_nonzero(np.any(grad != target, axis=1)))
print(f"exact gradient check: {bad} cells with grad != e1 [{time.time()-t0:.0f}s]", flush=True)

# exact completeness via adjacency closure with rational Gordan certificates
cellset = set(cells)
I5 = [tuple(1 if j == k else 0 for j in range(5)) for k in range(5)]
def verify_zero(cand, gens, k):
    # exact check that sum_j cand_j gens_j == 0 with cand nonneg and nonzero
    if any(c < 0 for c in cand) or all(c == 0 for c in cand): return False
    s = [F(0)]*5
    for j in range(k):
        if cand[j] != 0:
            for rr in range(5): s[rr] += cand[j]*gens[j][rr]
    return all(v == 0 for v in s)

def empty_certificate(gens):
    # True iff there is a nonneg nonzero rational x with sum_j x_j gens_j = 0 (Gordan: the cell is
    # empty). We compute the exact nullspace; a single sign-consistent basis vector is a certificate,
    # otherwise an LP finds a nonneg combination which we then verify exactly (no false positives).
    k = len(gens); Amat = [[F(gens[j][r]) for j in range(k)] for r in range(5)]
    piv = []; r = 0
    for c in range(k):
        if r >= 5: break
        pr = next((rr for rr in range(r, 5) if Amat[rr][c] != 0), None)
        if pr is None: continue
        Amat[r], Amat[pr] = Amat[pr], Amat[r]
        pv = Amat[r][c]; Amat[r] = [v/pv for v in Amat[r]]
        for rr in range(5):
            if rr != r and Amat[rr][c] != 0:
                f = Amat[rr][c]; Amat[rr] = [Amat[rr][j]-f*Amat[r][j] for j in range(k)]
        piv.append(c); r += 1
    free = [j for j in range(k) if j not in set(piv)]
    basis = []
    for fcol in free:
        vec = [F(0)]*k; vec[fcol] = F(1)
        for ri, pc in enumerate(piv): vec[pc] = -Amat[ri][fcol]
        basis.append(vec)
    if not basis: return False
    for vec in basis:
        nz = [v for v in vec if v != 0]
        cand = vec if all(v >= 0 for v in nz) else ([-v for v in vec] if all(v <= 0 for v in nz) else None)
        if cand is not None and verify_zero(cand, gens, k): return True
    B = np.array([[float(x) for x in vec] for vec in basis]).T; nf = len(basis)
    Aub = [[-B[j, a] for a in range(nf)] for j in range(k)]
    Aeq = [[float(sum(B[j, a] for j in range(k))) for a in range(nf)]]
    res = linprog(c=[0.0]*nf, A_ub=Aub, b_ub=[0.0]*k, A_eq=Aeq, b_eq=[1.0], bounds=[(None, None)]*nf, method='highs')
    if res.success:
        cand = [F(0)]*k
        for a in range(nf):
            ta = F(res.x[a]).limit_denominator(10**6)
            for j in range(k): cand[j] += ta*basis[a][j]
        if verify_zero(cand, gens, k): return True
    return False
inset = 0; emptyflip = 0; gaps = 0; t1 = time.time()
for s in cells:
    for i in range(M):
        sp = list(s); sp[i] = -sp[i]; sp = tuple(sp)
        if sp in cellset: inset += 1; continue
        gens = [tuple(v) for v in I5] + [tuple(int(sp[j]*W[j, r]) for r in range(5)) for j in range(M)]
        if empty_certificate(gens): emptyflip += 1
        else: gaps += 1
print(f"adjacency closure: inset={inset} empty={emptyflip} gaps={gaps} [{time.time()-t1:.0f}s]", flush=True)

ok = (bad == 0 and gaps == 0)
print("VERDICT:", "max6 IS computable in 2 hidden layers (exact theorem, no floating point in certificate)."
      if ok else "INCOMPLETE", flush=True)
if ok:
    with open("results/proof_certificate.txt", "w") as fh:
        fh.write("Exact computer-assisted proof: max(x1,...,x6) is computable by a 2-hidden-layer ReLU network.\n\n")
        fh.write(f"Construction: signed sum of weight-2 P2 support functions, denominator D={D}.\n")
        fh.write(f"Gradient-relevant arrangement: {M} unforced hyperplanes in R^5 (gap coordinates).\n")
        fh.write(f"(1) Complete deterministic enumeration: {NC} cells, each with an exact integer witness.\n")
        fh.write(f"(2) Exact gradient = e1 on all {NC} cells.\n")
        fh.write(f"(3) Adjacency closure: {inset} in-set flips + {emptyflip} flips with exact Gordan empty\n")
        fh.write(f"    certificates, 0 gaps => the enumeration is complete.\n\n")
        fh.write("Therefore C == x1 == max6 on the chamber; by continuity and S6-symmetry, == max6 on R^6. QED.\n")
    print("wrote results/proof_certificate.txt", flush=True)
