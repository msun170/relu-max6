# VALUATION-MOVE FACTORIZATION of the max_6 (and max_5) virtual Minkowski identity Delta + N = M.
# Object (per the precise spec): NB[row, block] = signed second-difference (wall jump) of that block's orbit-sum on a
# canonical NON-BRAID wall-jump probe row; rows grouped by S_n symmetry of the wall direction. The construction
# satisfies NB . c = 0 (exact non-braid cancellation). We (A) for each row list the blocks with nonzero contribution
# (the cancellation group) and classify by size; (B) find the GLOBAL factorization = minimal subsets of blocks whose
# coefficient-weighted jump-columns sum to ZERO over ALL rows (the valuation-move components); (C) for each component
# compute its BRAID effect J.c_G (valuation move = cancels NB AND has braid effect; pure cancellation = no braid
# effect). Two blocks are LINKED only if they cancel the same jump ROW, not merely share a normal.
import sys, itertools, json
from fractions import Fraction as F
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"

def family(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zon = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zon.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for vv in verts:
                wv = tuple(vv[k]+g[k] for k in range(n))
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    P1 = [frozenset([p]) for p in W2] + list(zon)
    bl = set(P1)
    for i in range(len(P1)):
        for j in range(i, len(P1)): bl.add(P1[i] | P1[j])
    return core.orbits_of(list(bl), n)
def hQ(members, x): return sum(max(sum(v[k]*x[k] for k in range(len(x))) for v in vs) for vs in members)
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def is_braid(d, n):
    bt = tuple(sorted(prim(tuple((1 if k==0 else -1 if k==1 else 0) for k in range(n)))))
    return tuple(sorted(prim(d))) == bt

def analyze(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 25; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = [[F(hQ(orb[k], x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    w = core.exact_solve(A, [F(max(x)) for x in Xs])
    supp = [(okeys[k], w[k]) for k in range(N) if w[k] != 0]
    B = len(supp); labels = [f"O{i+1}{'+' if c>0 else '-'}" for i, (k, c) in enumerate(supp)]
    cvec = [c for k, c in supp]
    # non-braid wall directions from support-block edges, grouped by S_n orbit (canonical rep)
    def dcanon(d):
        best = None
        for g in itertools.permutations(range(n)):
            t = prim(tuple(d[g[k]] for k in range(n)))
            if best is None or t < best: best = t
        return best
    nbset = {}
    for k, c in supp:
        rep = list(next(iter(orb[k])))
        for a in range(len(rep)):
            for b_ in range(a+1, len(rep)):
                d = prim(tuple(rep[a][i]-rep[b_][i] for i in range(n)))
                if not is_braid(d, n): nbset.setdefault(dcanon(d), d)
    nbdirs = list(nbset.values())
    # build NB rows (several probes per direction), integer jumps per block
    rng2 = np.random.default_rng(11); rows = []
    for d in nbdirs:
        for _ in range(8):
            x = np.array(rng2.integers(-6, 7, size=n)); top = int(rng2.integers(0, n))
            x[top] = int(x.max()) + int(np.abs(np.array(d)).sum()) + int(rng2.integers(3, 9)); x = tuple(int(t) for t in x)
            xp = tuple(x[i]+d[i] for i in range(n)); xm = tuple(x[i]-d[i] for i in range(n))
            jr = [hQ(orb[k], xp) + hQ(orb[k], xm) - 2*hQ(orb[k], x) for k, c in supp]
            rows.append((d, jr))
    # (A) per-row groups + verify cancellation
    rowinfo = []; sizes = []
    for d, jr in rows:
        grp = [i for i in range(B) if jr[i] != 0]
        tot = sum(cvec[i] * jr[i] for i in range(B))
        rowinfo.append({"dir": list(d), "group": [labels[i] for i in grp], "size": len(grp), "cancels": tot == 0})
        sizes.append(len(grp))
    # (B) global factorization: minimal subsets whose weighted jump-columns sum to 0 over ALL rows
    Vcols = [[cvec[i] * jr[i] for d, jr in rows] for i in range(B)]   # B columns, each over rows (weighted)
    def colsum_zero(S):
        return all(sum(Vcols[i][r] for i in S) == 0 for r in range(len(rows)))
    zero_subsets = [S for r in range(1, B+1) for S in itertools.combinations(range(B), r) if colsum_zero(set(S))]
    # minimal (by inclusion) nonempty zero-sum subsets
    minimal = []
    for S in sorted(zero_subsets, key=len):
        if not any(set(T) < set(S) for T in minimal): minimal.append(S)
    # braid effect of each minimal component: build braid probes
    braidrows = []
    for a in range(n):
        for b_ in range(a+1, n):
            d = tuple((1 if i==a else -1 if i==b_ else 0) for i in range(n))
            for _ in range(2):
                x = np.array(rng2.integers(-10, 1, size=n)); t = int(rng2.integers(3, 10)); x[a]=t; x[b_]=t; x=tuple(int(z) for z in x)
                jr = [hQ(orb[k], tuple(x[i]+d[i] for i in range(n))) + hQ(orb[k], tuple(x[i]-d[i] for i in range(n))) - 2*hQ(orb[k], x) for k, c in supp]
                braidrows.append(jr)
    comps = []
    for S in minimal:
        braideff = any(sum(cvec[i]*jr[i] for i in S) != 0 for jr in braidrows)
        comps.append({"blocks": [labels[i] for i in S], "size": len(S),
                      "type": ("pairwise" if len(S)==2 else "3-group" if len(S)==3 else f"{len(S)}-group"),
                      "braid_effect": braideff,
                      "role": ("valuation move (cancels NB, has braid effect)" if braideff else "pure cancellation (NB=0, no braid)")})
    out = {"n": n, "num_blocks": B, "labels": labels, "coeffs": [str(c) for c in cvec],
           "num_nonbraid_dirs": len(nbdirs), "num_rows": len(rows),
           "rows_pairwise": sum(1 for s in sizes if s == 2), "rows_3group": sum(1 for s in sizes if s == 3),
           "rows_larger": sum(1 for s in sizes if s > 3), "max_row_group_size": max(sizes) if sizes else 0,
           "all_rows_cancel": all(ri["cancels"] for ri in rowinfo),
           "minimal_components": comps, "max_component_size": max((c["size"] for c in comps), default=0)}
    print(f"\n===== max_{n}: {B} blocks, {len(nbdirs)} non-braid dirs, {len(rows)} rows =====")
    print(f"  all rows cancel exactly: {out['all_rows_cancel']}")
    print(f"  per-row group sizes: pairwise={out['rows_pairwise']} 3-group={out['rows_3group']} larger={out['rows_larger']} | MAX row group={out['max_row_group_size']}")
    print(f"  GLOBAL minimal components (max size {out['max_component_size']}):")
    for c in comps: print(f"    {c['blocks']}  [{c['type']}]  {c['role']}")
    return out

results = {}
for n in (5, 6):
    results[f"max{n}"] = analyze(n)
with open(f"{SCR}/valuation_factorization_max6.json", "w") as fh: json.dump(results, fh, indent=2)
print(f"\nwrote {SCR}/valuation_factorization_max6.json")
