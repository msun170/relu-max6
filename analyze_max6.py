# Dissect the explicit max6 2-layer construction. For each block: real vertices, edges (LP-exact) with their
# COMPLEXITY (min #roots = sum of positive entries), the join decomposition into two zonotopes Z1,Z2 (= the
# layer-1 ReLU generators), and the BRIDGE edges (cross edges Z1<->Z2 that carry the complexity-2 structure).
import sys, itertools
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 6
BLOCKS = [
 (F(1,360), [(0,0,1,0,0,1),(0,0,2,0,0,0),(0,1,1,0,0,0),(1,0,0,0,0,1),(1,0,1,0,0,0),(2,0,0,0,0,0)]),
 (F(-1,360),[(0,0,1,0,0,1),(0,0,2,0,0,0),(0,1,0,0,0,1),(0,1,0,0,1,0)]),
 (F(1,360), [(0,0,0,0,0,2),(0,0,0,0,1,1),(0,0,0,1,0,1),(0,0,0,2,0,0),(0,1,0,0,0,1),(0,1,0,0,1,0),(0,2,0,0,0,0)]),
 (F(1,90),  [(0,0,0,1,0,1),(0,0,1,0,0,1),(0,0,1,1,0,0),(0,0,2,0,0,0),(0,1,0,0,1,0),(0,1,1,0,0,0),(1,0,0,0,1,0),(1,0,1,0,0,0)]),
 (F(-1,90), [(0,0,0,1,0,1),(0,0,0,1,1,0),(0,0,1,1,0,0),(0,1,0,0,0,1),(0,1,0,0,1,0),(0,1,0,1,0,0),(1,0,1,0,0,0),(1,1,0,0,0,0)]),
 (F(-1,180),[(0,0,0,0,0,2),(0,0,0,0,1,1),(0,0,1,0,0,1),(0,0,1,0,1,0),(0,1,0,1,0,0),(1,0,0,1,0,0),(1,1,0,0,0,0),(2,0,0,0,0,0)]),
]
def complexity(d):  # min #roots summing to d (d in root lattice, sum 0): = sum of positive entries
    return sum(x for x in d if x > 0)
def is_vertex(V, i):
    A = [[float(V[j][k]-V[i][k]) for k in range(n)] for j in range(len(V)) if j != i]
    if not A: return True
    res = linprog(c=[0]*n, A_ub=A, b_ub=[-1.0]*len(A), bounds=[(-10,10)]*n, method='highs')
    return res.success
def is_edge(V, i, j):
    A = [[float(V[k][t]-V[i][t]) for t in range(n)] for k in range(len(V)) if k not in (i,j)]
    Aeq = [[float(V[i][t]-V[j][t]) for t in range(n)]]
    if not A: return True
    res = linprog(c=[0]*n, A_ub=A, b_ub=[-1.0]*len(A), A_eq=Aeq, b_eq=[0.0], bounds=[(-10,10)]*n, method='highs')
    return res.success
def real_vertices(Vraw):
    V = list(dict.fromkeys(Vraw))
    if len(V) <= 2: return V
    return [V[i] for i in range(len(V)) if is_vertex(V, i)]

# weight-2 zonotopes for the join decomposition
W2 = core.weight2_points(n); W2set = set(W2)
gens2 = sorted({d for p in W2 for q in W2 if p!=q for d in [tuple(p[k]-q[k] for k in range(n))] if d>tuple(-x for x in d)})
zonos = set()
def dfs(verts, gi):
    zonos.add(frozenset(verts))
    for j in range(gi+1,len(gens2)):
        g=gens2[j]; new=set(); ok=True
        for v in verts:
            wv=tuple(v[k]+g[k] for k in range(n))
            if wv not in W2set: ok=False;break
            new.add(wv)
        if ok: dfs(verts|new, j)
for p in W2: dfs({p}, -1)
zonos=[frozenset(z) for z in zonos]
def hullset(S):
    V=list(S); return frozenset(real_vertices(V))
def decompose(Vverts):
    target=frozenset(Vverts)
    cand=sorted([z for z in zonos if z<=target], key=lambda z:-len(z))
    for i in range(len(cand)):
        for j in range(i,len(cand)):
            if hullset(cand[i]|cand[j])==target: return cand[i],cand[j]
    return None,None

print("=== max6: layer structure of the 6 construction blocks ===\n", flush=True)
for idx,(c,Vraw) in enumerate(BLOCKS):
    V=real_vertices(Vraw)
    imgs={frozenset(tuple(v[p[k]] for k in range(n)) for v in V) for p in itertools.permutations(range(n))}
    osz=len(imgs)
    E=[(i,j) for i in range(len(V)) for j in range(i+1,len(V)) if is_edge(V,i,j)]
    Ecx={}
    for (i,j) in E:
        d=tuple(V[i][k]-V[j][k] for k in range(n)); cx=min(complexity(d),complexity(tuple(-x for x in d)))
        Ecx[cx]=Ecx.get(cx,0)+1
    Z1,Z2=decompose(V)
    print(f"Block {idx+1}: coeff {c}, orbit {osz}, {len(V)} vertices, {len(E)} edges by complexity {dict(sorted(Ecx.items()))}", flush=True)
    if Z1 is not None:
        z1=sorted(frozenset(real_vertices(list(Z1)))); z2=sorted(frozenset(real_vertices(list(Z2))))
        # bridge edges = edges with one endpoint in Z1, other in Z2
        z1s,z2s=set(z1),set(z2); br=[]
        for (i,j) in E:
            a,b=V[i],V[j]
            if (a in z1s and b in z2s) or (a in z2s and b in z1s):
                d=tuple(a[k]-b[k] for k in range(n)); br.append(min(complexity(d),complexity(tuple(-x for x in d))))
        print(f"  = JOIN of Z1({len(z1)}v) and Z2({len(z2)}v); bridge edges complexity {sorted(br)}", flush=True)
        print(f"    Z1={z1}", flush=True)
        print(f"    Z2={z2}", flush=True)
    else:
        print(f"  (no two-zonotope decomposition found)", flush=True)
    print(flush=True)
