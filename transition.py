# max6 -> max7 transition, done correctly: gradient matching at GENERIC float points (no argmax ties).
# A weight-2 representation exists iff  grad(max_n) = sum_t c_t grad(h_orbit_t) + const  as fields, i.e. the
# max_n gradient field lies in the span of the block gradient fields plus constants. Report, per n: #orbits
# (DOF), rank of the block-gradient system, rank with the max_n target, defect (over-determination), whether
# feasible, and the solution-space dimension (slack). This shows how the codim-1 system tightens 5 -> 6 -> 7.
import sys, itertools
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
P = 2147483647

def modp_rank(M):
    A = (M % P).astype(np.int64).copy(); rows, cols = A.shape; r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % P != 0), -1)
        if piv < 0: continue
        A[[r, piv]] = A[[piv, r]]; A[r] = (A[r]*pow(int(A[r, c]), P-2, P)) % P
        for i in range(rows):
            if i != r and A[i, c] % P != 0: A[i] = (A[i]-A[i, c]*A[r]) % P
        r += 1
        if r == rows: break
    return r

def weight2_orbits(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zon = set()
    def dfs(v, gi):
        if len(v) >= 2: zon.add(frozenset(v))
        for j in range(gi+1, len(gens)):
            g = gens[j]; nw = set(); ok = True
            for u in v:
                wv = tuple(u[k]+g[k] for k in range(n))
                if wv not in W2set: ok = False; break
                nw.add(wv)
            if ok: dfs(v | nw, j)
    for p in W2: dfs({p}, -1)
    P1 = [frozenset([p]) for p in W2] + list(zon); bl = set(P1)
    for i in range(len(P1)):
        for j in range(i, len(P1)): bl.add(P1[i] | P1[j])
    return core.orbits_of(list(bl), n)

print(f"{'n':>2} {'orbits':>7} {'rankA':>6} {'rank+tgt':>9} {'defect':>7} {'feasible':>9} {'slack(DOF-rank)':>16} {'resid':>8}")
for n in (5, 6, 7):
    orb = weight2_orbits(n); okeys = list(orb); N = len(okeys)
    orbV = [[np.array(sorted(m), dtype=np.float64) for m in orb[k]] for k in okeys]
    M = N + 80
    rng = np.random.default_rng(0)
    X = rng.standard_normal((M, n)) * 7.0           # generic float points -> unique argmaxes
    rows = []; tgt = []
    for xi in range(M):
        x = X[xi]
        # block gradients at x: sum over orbit members of argmax-vertex
        gcols = []
        for mem in orbV:
            g = np.zeros(n)
            for A in mem: g = g + A[int(np.argmax(A @ x))]
            gcols.append(g)
        gmax = np.zeros(n); gmax[int(np.argmax(x))] = 1.0     # grad(max_n)(x) = e_argmax
        # n component-equations: sum_t c_t gcols[t][comp] + const[comp] = gmax[comp]
        for comp in range(n):
            row = [gcols[t][comp] for t in range(N)] + [1.0 if k == comp else 0.0 for k in range(n)]
            rows.append(row); tgt.append(gmax[comp])
    A = np.array(rows); b = np.array(tgt)
    Ai = np.rint(A).astype(np.int64); bi = np.rint(b).astype(np.int64)
    rA = modp_rank(Ai); rAb = modp_rank(np.column_stack([Ai, bi]))
    feasible = (rA == rAb); defect = rAb - rA; dof = N + n
    sol, *_ = np.linalg.lstsq(A, b, rcond=None); resid = np.linalg.norm(A@sol-b)/max(np.linalg.norm(b),1e-9)
    print(f"{n:>2} {N:>7} {rA:>6} {rAb:>9} {defect:>7} {str(feasible):>9} {dof-rA:>16} {resid:>8.4f}", flush=True)
