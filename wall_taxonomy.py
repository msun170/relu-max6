# WALL TAXONOMY (seed of the descent theorem). A homogeneous support function h_Q is linear on the normal cones of
# the vertices of Q and bends across the hyperplane {<v_a - v_b, x>=0} for each EDGE (v_a,v_b) of conv(Q). The wall
# normals of a block are thus the differences of edge-adjacent vertices. max_n = h_Delta has ONLY braid walls
# (normals ~ e_a - e_b). So any exact rep sum_O a_O OrbitSum_O = max_n must CANCEL every NON-braid wall.
# QUESTION: where do non-braid walls come from? Claim: the JOIN operation itself. A join conv(Z1 u Z2) has BRIDGE
# edges connecting a Z1 vertex to a Z2 vertex, whose normals are generically NON-braid -- EVEN WHEN both generators
# are roots. We verify this exactly on small blocks (<=4 vertices, where convex position => all pairwise diffs are
# edges) and tabulate braid vs non-braid (= bridge) wall normals for root-gen joins vs non-root joins.
import itertools
from math import gcd
from functools import reduce
import numpy as np
n = 7; W = 4
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4)
roots = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a != b}
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def is_braid(d): return prim(d) in {prim(r) for r in roots}
gens = sorted({prim(tuple(q[k]-p[k] for k in range(n))) for p in W4 for q in W4 if p != q})
rootgens = [g for g in gens if is_braid(g)]
nonrootgens = [g for g in gens if not is_braid(g)]
print(f"weight-4 primitive generators: {len(gens)} total = {len(rootgens)} root (braid) + {len(nonrootgens)} non-root")

def seg(p, g):
    q = shift(p, g); return frozenset([p, q]) if q in W4set else None
def affinely_independent(pts):
    P = np.array(pts, dtype=float); P = P[1:] - P[0]
    return np.linalg.matrix_rank(P) == len(pts) - 1
def wall_normals(block):  # for <=4 pts in convex (affinely independent) position, all pairwise diffs are edges
    V = list(block); out = []
    for a, b in itertools.combinations(range(len(V)), 2):
        out.append(prim(tuple(V[a][k]-V[b][k] for k in range(n))))
    return out

rng = np.random.default_rng(0)
def survey(label, gpool):
    nb_join = 0; total = 0; bridge_nonbraid = 0; bridge_total = 0; samples = 0
    examples = []
    tries = 0
    while samples < 4000 and tries < 200000:
        tries += 1
        p = W4[rng.integers(len(W4))]; q = W4[rng.integers(len(W4))]
        g1 = gpool[rng.integers(len(gpool))]; g2 = gpool[rng.integers(len(gpool))]
        s1 = seg(p, g1); s2 = seg(q, g2)
        if s1 is None or s2 is None: continue
        U = s1 | s2
        if len(U) != 4: continue            # want genuine 4-vertex joins of two segments
        if not affinely_independent(list(U)): continue   # convex position: all 6 pairwise diffs are edges
        samples += 1
        Vs = list(U); g1p, g2p = prim(g1), prim(g2)
        # bridge edges = pairs with one vertex from s1 and one from s2
        s1l = list(s1); s2l = list(s2)
        br = [prim(tuple(a[k]-b[k] for k in range(n))) for a in s1l for b in s2l]
        bn = sum(not is_braid(d) for d in br)
        bridge_nonbraid += bn; bridge_total += len(br)
        if bn > 0: nb_join += 1
        total += 1
        if len(examples) < 3 and bn > 0 and is_braid(g1p) and is_braid(g2p):
            examples.append((g1p, g2p, [d for d in br if not is_braid(d)]))
    print(f"\n[{label}] {total} convex 4-vertex segment-joins sampled")
    print(f"  joins with >=1 NON-braid (bridge) wall: {nb_join}/{total} = {100*nb_join/max(total,1):.1f}%")
    print(f"  bridge edges that are NON-braid: {bridge_nonbraid}/{bridge_total} = {100*bridge_nonbraid/max(bridge_total,1):.1f}%")
    for g1p, g2p, nbs in examples:
        print(f"  example ROOT-gen join: g1={g1p} g2={g2p} (both braid) -> non-braid bridge normals e.g. {nbs[:2]}")

survey("ROOT generators only", rootgens)
survey("NON-ROOT generators", nonrootgens)
print("\nCONCLUSION: if root-gen joins still show non-braid bridge walls, then the JOIN operation -- not just")
print("non-root generators -- injects the non-braid walls that any exact max_7 rep must cancel. That makes the")
print("non-braid-cancellation constraint UNIVERSAL across 2-layer blocks, and is the mechanism a descent theorem")
print("must exploit: bridges must pairwise cancel, which should force bounded effective complexity.")
