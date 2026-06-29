# Toward a width lower bound via cross-codimensional rigidity, made into a clean COVERING argument.
#
# Reduction (rigorous, real-weight valid): in any depth-2 rep max_n = sum_t c_t h_{Q_t} + linear, look at a
# triangular 2-face corner of Delta -- the normal cone C_{ijk} where coords i,j,k tie for the max. The face
# of Delta there is the triangle T_{ijk}, which is NOT centrally symmetric. In the signed Minkowski sum the
# face at C_{ijk} is sum_t c_t face_{Q_t}(C_{ijk}). Centrally symmetric faces (parallelograms from the
# zonotope parts) contribute ZERO asymmetry. So the triangle's asymmetry must come from blocks with an
# ASYMMETRIC 2-face at C_{ijk} (a bridge face of the join). Hence every one of the C(n,3) corners needs at
# least one block that asymmetrically covers it:  union_t S_t = all triples,  where
#   S_t = { (i,j,k) : Q_t has a non-centrally-symmetric 2-face in normal cone C_{ijk} }.
# Therefore  WIDTH  >=  C(n,3) / max_t |S_t|.  If max_t |S_t| grows slower than C(n,3), width grows.
#
# This script (a) verifies max_n covers all C(n,3) corners asymmetrically, and (b) measures |S_t| for the
# (small) lattice blocks to see whether the covering bound has teeth.
import sys, itertools
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

def face_in_dir(verts, w):
    vals = [sum(v[k]*w[k] for k in range(len(w))) for v in verts]
    M = max(vals)
    return [v for v, val in zip(verts, vals) if val == M]

def centrally_symmetric(face, coords):
    # project face vertices onto the given coords, center, test V == -V
    pts = [tuple(v[c] for c in coords) for v in face]
    pts = list(set(pts))
    d = len(coords)
    # centroid times len (avoid fractions): 2*sum should map each pt to its mirror present in set
    s = [sum(p[c] for p in pts) for c in range(d)]
    L = len(pts); ss = set(pts)
    for p in pts:
        mirror = tuple((2*s[c] - L*p[c]) for c in range(d))   # L*(2*centroid - p)
        # need mirror/L to be a vertex: compare L*q to mirror for some q
        if not any(tuple(L*q[c] for c in range(d)) == mirror for q in ss):
            return False
    return True

def asym_corners(verts, n):
    cov = []
    for tri in itertools.combinations(range(n), 3):
        w = [1 if k in tri else 0 for k in range(n)]
        F = face_in_dir(verts, w)
        if len(F) >= 3 and not centrally_symmetric(F, tri):
            cov.append(tri)
    return cov

def lattice_blocks(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zonos = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zonos.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for v in verts:
                wv = tuple(v[k]+g[k] for k in range(n))
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    P1 = [frozenset([p]) for p in W2] + list(zonos)
    blocks = set(P1)
    for i in range(len(P1)):
        for j in range(i, len(P1)):
            blocks.add(P1[i] | P1[j])
    return blocks

print("asymmetry-covering width bound: WIDTH >= C(n,3) / max_t |S_t|", flush=True)
for n in (5, 6, 7):
    simplex = [tuple(1 if k==i else 0 for k in range(n)) for i in range(n)]
    cn3 = len(list(itertools.combinations(range(n), 3)))
    sc = len(asym_corners(simplex, n))
    blocks = lattice_blocks(n)
    sizes = [len(asym_corners(sorted(Q), n)) for Q in blocks]
    mx = max(sizes); bound = -(-cn3 // mx) if mx else 0   # ceil
    nz = [s for s in sizes if s > 0]
    print(f" n={n}: C(n,3)={cn3}; max_n covers {sc}/{cn3} corners (expect all). "
          f"lattice blocks={len(blocks)}, max|S_t|={mx}, "
          f"blocks with S_t>0: {len(nz)}, mean|S_t| (nz)={np.mean(nz):.2f} => width >= {bound}", flush=True)
print("\nNote: lattice blocks are small (<=8 vertices), so this max|S_t| is a LOWER estimate of what general"
      "\nreal blocks could cover; the bound's strength depends on bounding |S_t| for ALL joins (the open step).", flush=True)
