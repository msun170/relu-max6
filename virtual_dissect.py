# DISSECT the max_5 / max_6 virtual Minkowski identity  Delta + N = M  (M = positive P2 blocks, N = negative).
# The theorem: for n>=5 every rep is signed, so the simplex appears only AFTER cancellation. This script exposes the
# cancellation: it computes, for each NON-BRAID (bridge) wall direction, the per-block second-difference (wall jump)
# of every orbit-summed block, verifies the signed sum cancels to ZERO on non-braid walls (so M-N has only braid
# walls = Delta), and shows WHICH positive/negative blocks pair up. If the cancellation groups into a few recognizable
# patterns (a high-vertex block vs a sum of lower-vertex ones = a valuation move), that is the descent prototype.
import sys, itertools
from fractions import Fraction as F
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

def family(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zonos = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zonos.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for vv in verts:
                wv = tuple(vv[k]+g[k] for k in range(n))
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    P1 = [frozenset([p]) for p in W2] + list(zonos)
    blocks = set(P1)
    for i in range(len(P1)):
        for j in range(i, len(P1)): blocks.add(P1[i] | P1[j])
    return core.orbits_of(list(blocks), n)
def hQ(members, x): return sum(max(sum(v[k]*x[k] for k in range(len(x))) for v in vs) for vs in members)
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def is_braid(d, n):
    bt = tuple(sorted(prim(tuple((1 if k==0 else -1 if k==1 else 0) for k in range(n)))))
    return tuple(sorted(prim(d))) == bt

def dissect(n):
    print(f"\n===== max_{n} virtual dissection (weight-2) =====")
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 25; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = [[F(hQ(orb[k], x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    b = [F(max(x)) for x in Xs]
    w = core.exact_solve(A, b)
    if w is None: print("  no weight-2 rep (OUT)"); return
    coeffs = w[:N]; lin = w[N:]
    supp = [(okeys[k], coeffs[k]) for k in range(N) if coeffs[k] != 0]
    pos = [(k, c) for k, c in supp if c > 0]; neg = [(k, c) for k, c in supp if c < 0]
    print(f"  {len(supp)} P2 blocks: {len(pos)} positive (M), {len(neg)} negative (N). linear part={[str(t) for t in lin]}")
    for k, c in sorted(supp, key=lambda kc: -abs(kc[1])):
        rep = sorted(next(iter(orb[k]))); nb = sum(not is_braid(tuple(rep[a][i]-rep[b_][i] for i in range(n)), n)
                                                    for a in range(len(rep)) for b_ in range(a+1, len(rep)))
        print(f"    coeff {str(c):>7}  | {len(rep)} verts | {nb} non-braid edge-walls | rep {rep}")
    # collect non-braid edge directions across all support blocks
    nbdirs = set()
    for k, c in supp:
        rep = list(next(iter(orb[k])))
        for a in range(len(rep)):
            for b_ in range(a+1, len(rep)):
                d = prim(tuple(rep[a][i]-rep[b_][i] for i in range(n)))
                if not is_braid(d, n): nbdirs.add(d)
    nbdirs = sorted(nbdirs)
    print(f"  {len(nbdirs)} distinct non-braid wall directions in the support")
    # verify cancellation: for many probes (x0,d), sum_t c_t [OS_t(x0+d)+OS_t(x0-d)-2 OS_t(x0)] == 0 (non-braid),
    # and that the FULL rep equals max_n (sanity). Show one per-block jump table.
    rng2 = np.random.default_rng(11); bad = 0; checks = 0
    for d in nbdirs:
        for _ in range(6):
            x = np.array(rng2.integers(-6, 7, size=n)); top = int(rng2.integers(0, n))
            x[top] = int(x.max()) + int(np.abs(np.array(d)).sum()) + int(rng2.integers(3, 8)); x = tuple(int(t) for t in x)
            xp = tuple(x[i]+d[i] for i in range(n)); xm = tuple(x[i]-d[i] for i in range(n))
            tot = F(0)
            for k, c in supp:
                j = hQ(orb[k], xp) + hQ(orb[k], xm) - 2*hQ(orb[k], x)
                tot += c * j
            checks += 1
            if tot != 0: bad += 1
    print(f"  NON-BRAID CANCELLATION: {checks-bad}/{checks} probes give exactly 0  => bridge walls cancel: {bad==0}")
    # show the per-block jump pattern for the first non-braid direction (the cancellation structure)
    d = nbdirs[0]; x = None
    rng3 = np.random.default_rng(1)
    while x is None:
        xx = np.array(rng3.integers(-6, 7, size=n)); top = int(rng3.integers(0, n))
        xx[top] = int(xx.max()) + int(np.abs(np.array(d)).sum()) + 5; x = tuple(int(t) for t in xx)
    print(f"  per-block jump across non-braid d={d}:")
    s = F(0)
    for k, c in sorted(supp, key=lambda kc: -abs(kc[1])):
        j = hQ(orb[k], tuple(x[i]+d[i] for i in range(n))) + hQ(orb[k], tuple(x[i]-d[i] for i in range(n))) - 2*hQ(orb[k], x)
        s += c*j
        if j != 0: print(f"     coeff {str(c):>7} x jump {j:>4}  = {str(c*j):>8}")
    print(f"     SUM = {s}  (must be 0)")

for n in (5, 6):
    dissect(n)
