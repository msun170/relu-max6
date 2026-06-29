"""Shared engine for the max_n / ReLU-depth problem.

A ReLU network with 2 hidden layers computes exactly the signed sums of P2 support functions,
where a P2 polytope is a Minkowski sum of joins of zonotopes. Since the support function of a
Minkowski sum is the sum of support functions, every such function is a signed linear combination
of single building blocks  h_Q(x) = max_{g in Q} <g,x>,  Q = conv(Z1 u Z2) a join of two zonotopes.

For max_n = h_{simplex}, the natural building blocks live on the weight-2 lattice points of the
dilated simplex 2*Delta^{n-1}, i.e. {2 e_i} and {e_i + e_j}. This module generates those blocks,
reduces them to S_n-orbits, and provides exact rational solving. n is a parameter throughout.
"""
import itertools
from fractions import Fraction as F
import numpy as np

# basis vectors and the weight-2 lattice of 2*Delta^{n-1}
def e(i, n):
    a = [0]*n; a[i] = 1; return tuple(a)

def add(*pts):
    n = len(pts[0]); return tuple(sum(p[k] for p in pts) for k in range(n))

def weight2_points(n):
    return [tuple(2*v for v in e(i, n)) for i in range(n)] + \
           [add(e(i, n), e(j, n)) for i, j in itertools.combinations(range(n), 2)]

def difference_generators(n):
    return [tuple(e(a, n)[k]-e(b, n)[k] for k in range(n)) for a in range(n) for b in range(n) if a != b]

# zonotopes over the weight-2 lattice: a base point plus <=R difference generators, every partial
# sum required to stay in the lattice (so the vertex set is a genuine lattice zonotope)
def zonotopes(n, R=3):
    W2 = weight2_points(n); W2set = set(W2); gens = difference_generators(n)
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    out = set()
    for p in W2:
        for r in range(1, R+1):
            for T in itertools.combinations(range(len(gens)), r):
                verts = {p}; ok = True
                for s in range(1, r+1):
                    for S in itertools.combinations(T, s):
                        q = p
                        for k in S: q = shift(q, gens[k])
                        if q not in W2set: ok = False; break
                        verts.add(q)
                    if not ok: break
                if ok and len(verts) >= 2: out.add(frozenset(verts))
    return [frozenset(z) for z in out]

# P2 building blocks: single zonotopes and their joins (union of vertex sets), capped at `cap`
# vertices. Joins of arity >=3 are built incrementally from orbit reps to avoid a blow-up.
def building_blocks(n, R=3, cap=12, join=2):
    zs = zonotopes(n, R)
    terms = set(zs)
    for i in range(len(zs)):
        for j in range(i+1, len(zs)):
            u = zs[i] | zs[j]
            if len(u) <= cap: terms.add(u)
    if join >= 3:
        for _ in range(3, join+1):
            reps = {}
            for vs in terms: reps.setdefault(canon(vs, n), vs)
            nxt = set(terms)
            for vs in reps.values():
                for z in zs:
                    u = vs | z
                    if len(u) <= cap: nxt.add(u)
            terms = nxt
    return list(terms)

# S_n-orbit canonicalization. Only column permutations that preserve each coordinate's value
# profile (its multiset of values across the vertices) can realize the lex-min, so we restrict to
# those; this is exact and far cheaper than iterating all n! permutations.
def canon(vset, n):
    V = list(vset)
    prof = [tuple(sorted(v[c] for v in V)) for c in range(n)]
    classes = {}
    for c in range(n): classes.setdefault(prof[c], []).append(c)
    order = sorted(classes); slots = {}; pos = 0
    for p in order:
        slots[p] = list(range(pos, pos+len(classes[p]))); pos += len(classes[p])
    choices = [[dict(zip(perm, slots[p])) for perm in itertools.permutations(classes[p])] for p in order]
    best = None
    for combo in itertools.product(*choices):
        pm = [0]*n
        for d in combo:
            for s, t in d.items(): pm[s] = t
        key = tuple(sorted(tuple(v[c] for c in sorted(range(n), key=lambda c: pm[c])) for v in V))
        if best is None or key < best: best = key
    return best

def orbits_of(terms, n):
    orb = {}
    for vs in terms: orb.setdefault(canon(vs, n), []).append(vs)
    return orb

# support function of one orbit, summed over its members, evaluated at integer points Pts (P x n).
# orbit_column(...)[r] = sum_{Q in orbit} max_{g in Q} <g, Pts[r]>   (exact integers)
def orbit_column(members, Pts):
    col = np.zeros(Pts.shape[0], dtype=np.int64)
    for vs in members:
        G = np.array(list(vs), dtype=np.int64); col += (Pts @ G.T).max(axis=1)
    return col

# exact rational linear solve A w = b (A: rows of Fraction, b: Fractions). Returns a particular
# solution with free variables set to 0, or None if the system is inconsistent.
def exact_solve(A, b):
    m = len(A); ncol = len(A[0]) if m else 0
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    piv = []; r = 0
    for c in range(ncol):
        pr = next((i for i in range(r, m) if M[i][c] != 0), None)
        if pr is None: continue
        M[r], M[pr] = M[pr], M[r]
        inv = M[r][c]; M[r] = [v/inv for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [a - f*bv for a, bv in zip(M[i], M[r])]
        piv.append((r, c)); r += 1
        if r == m: break
    for i in range(m):
        if all(M[i][c] == 0 for c in range(ncol)) and M[i][ncol] != 0: return None
    w = [F(0)]*ncol
    for rr, cc in piv: w[cc] = M[rr][ncol]
    return w
