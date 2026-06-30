# Residual structure analysis (toward a CONTRACTION/closure proof).
# Extract R_w = max7 - F_w (F_w = best weight-<=w approximant) for w=2,3 on a common point set, then probe:
#   (1) contraction ratio rho = ||R_3||/||R_2||  (should match the floor ratio 0.0022/0.0308 = 0.071),
#   (2) even/odd energy split of R_w,
#   (3) CONCENTRATION: bin points by "active-set size at the max" (#coords within tau of max(x)) -- i.e. by how
#       deep a tie-stratum the point sits near -- and measure mean |R_w| per bin. If R_2 and R_3 have the SAME
#       normalized profile (just scaled), that is self-similarity => a recurrence R_{w+1}=T(R_w) is plausible.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7
t0 = time.time()

def enumerate_blocks(W):
    WP = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; WPset = set(WP)
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    gens = set()
    for p in WP:
        for q in WP:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
            if dd > tuple(-x for x in dd): gens.add(dd)
    gens = sorted(gens)
    def ssz(p, G):
        verts = set()
        for s in range(len(G)+1):
            for S in itertools.combinations(G, s):
                q = p
                for g in S: q = shift(q, g)
                if q not in WPset: return None
                verts.add(q)
        return frozenset(verts)
    zonos = set()
    for p in WP:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: zonos.add(v)
    for p in WP:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in WPset: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: zonos.add(v)
    if W >= 3:
        for z in [z for z in zonos if len(z) >= 4]:
            for g in gens:
                verts = set(z); ok = True
                for u in z:
                    w = shift(u, g)
                    if w not in WPset: ok = False; break
                    verts.add(w)
                if ok and len(verts) > len(z): zonos.add(frozenset(verts))
    zonos = list(zonos); zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
    blocks = set(zonos)
    cap = 12 if W == 2 else 16
    for a in zreps:
        for b in zonos:
            u = a | b
            if len(u) <= cap: blocks.add(u)
    orb = core.orbits_of(list(blocks), n)
    return orb, list(orb)

def proj_residual(Acols, b):
    # CGLS on GPU: residual of b projected onto columns of Acols (m x N). Returns (rel_resid, coeffs, residvec).
    Ag = cp.asarray(Acols, dtype=cp.float32); bg = cp.asarray(b, dtype=cp.float32)
    cn = cp.linalg.norm(Ag, axis=0); cn = cp.where(cn > 0, cn, cp.float32(1.0)); Ag2 = Ag / cn
    x = cp.zeros(Ag2.shape[1], dtype=cp.float32); r = bg.copy()
    s = Ag2.T @ r; p = s.copy(); gamma = float(s @ s); bn = float(cp.linalg.norm(bg))
    for it in range(6000):
        q = Ag2 @ p; alpha = gamma/float(q @ q); x += alpha*p; r -= alpha*q
        s = Ag2.T @ r; gnew = float(s @ s); beta = gnew/gamma; p = s + beta*p; gamma = gnew
        if it % 500 == 0 and float(cp.linalg.norm(s))/(float(cp.linalg.norm(Ag2))*float(cp.linalg.norm(r))+1e-30) < 1e-7: break
    return float(cp.linalg.norm(r))/bn, (x/cn), cp.asnumpy(r)

m = 16000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
Xm = -X
bmaxX = X.max(axis=1).astype(np.float32); bmaxXm = Xm.max(axis=1).astype(np.float32)

results = {}
for W in (2, 3):
    orb, okeys = enumerate_blocks(W)
    N = len(okeys)
    CX = np.empty((m, N), dtype=np.float32); CXm = np.empty((m, N), dtype=np.float32)
    for j, k in enumerate(okeys):
        CX[:, j] = core.orbit_column(orb[k], X).astype(np.float32)
        CXm[:, j] = core.orbit_column(orb[k], Xm).astype(np.float32)
    lin = X.astype(np.float32)
    A = np.column_stack([CX, lin])
    rel, coef, _ = proj_residual(A, bmaxX)
    coef = cp.asnumpy(coef)
    FX = A @ coef                                   # F_w at X
    Am = np.column_stack([CXm, Xm.astype(np.float32)])
    FXm = Am @ coef                                  # F_w at -X (same coeffs)
    RX = bmaxX - FX; RXm = bmaxXm - FXm
    results[W] = dict(N=N, rel=rel, RX=RX, RXm=RXm)
    print(f"weight-{W}: {N} orbits, floor f({W})={rel:.5f}  [{time.time()-t0:.0f}s]", flush=True)

R2 = results[2]["RX"]; R3 = results[3]["RX"]
n2 = np.linalg.norm(R2); n3 = np.linalg.norm(R3)
print(f"\n(1) CONTRACTION:  ||R2||={n2:.1f} ||R3||={n3:.1f}  rho_23 = ||R3||/||R2|| = {n3/n2:.4f}")

# (2) even/odd energy (match x <-> -x via the SAME index, since Xm=-X)
for W in (2, 3):
    RX = results[W]["RX"]; RXm = results[W]["RXm"]
    ev = (RX + RXm) / 2; od = (RX - RXm) / 2
    eE = np.linalg.norm(ev)**2; oE = np.linalg.norm(od)**2; tot = eE + oE
    print(f"(2) weight-{W} residual energy: even {eE/tot*100:.1f}%  odd {oE/tot*100:.1f}%")

# (3) concentration by active-set size at the max
Xs = np.sort(X, axis=1)[:, ::-1]                     # descending
mx = Xs[:, 0]
tau = 0.5
active = (np.abs(X - mx[:, None]) <= tau).sum(axis=1)   # #coords within tau of the max
print("\n(3) CONCENTRATION: mean |R_w| by active-set size at max (the tie-stratum depth):")
print(f"  {'k':>3} {'#pts':>7} {'mean|R2|':>10} {'mean|R3|':>10} {'R3/R2':>7}")
for k in range(1, n+1):
    msk = active == k
    if msk.sum() < 5: continue
    a2 = np.abs(R2[msk]).mean(); a3 = np.abs(R3[msk]).mean()
    print(f"  {k:>3} {int(msk.sum()):>7} {a2:>10.4f} {a3:>10.4f} {a3/max(a2,1e-9):>7.3f}")
print(f"\n[{time.time()-t0:.0f}s]  (if R3/R2 ratio is ~constant across strata, residual is SELF-SIMILAR -> contraction plausible)")
