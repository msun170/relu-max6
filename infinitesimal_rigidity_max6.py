# INFINITESIMAL RIGIDITY of the max_5 / max_6 two-layer representations (the deciding test for the rigidity thesis).
# Coefficient rigidity (the NB-circuit) only forces c given fixed blocks. Here we let the BLOCK GEOMETRY move too:
# treat every vertex of every (orbit-rep) block, every coefficient c_t, and the linear term as REAL variables, and
# linearize the identity  sum_t c_t OS_t(x) + d*(sum x_i) = max_n(x)  at the known construction. tangent dim =
# #params - rank(Jacobian). Then subtract the GAUGE (function-preserving) deformations: per-block scaling
# (Q_t -> lambda Q_t, c_t -> c_t/lambda) and per-block translation (absorbed by d). essential = tangent - gauge.
# essential == 0  => representation is ISOLATED up to symmetry/scaling => rigidity thesis REAL.
# essential  > 0  => continuous geometric deformation freedom => "discrete" claim too strong.
import sys, itertools
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

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

def rigidity(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 25; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = [[F(hQ(orb[k], x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    w = core.exact_solve(A, [F(max(x)) for x in Xs])
    supp_idx = [k for k in range(N) if w[k] != 0]
    supp = [okeys[k] for k in supp_idx]; cvec = [float(w[k]) for k in supp_idx]
    # for each support block, list distinct members with a representative permutation sigma (member = sigma . rep)
    perms = list(itertools.permutations(range(n)))
    blocks = []   # each: list of (sigma, vertex_tuple_list) members, and the REP vertex list
    for k in supp:
        rep = sorted(next(iter(orb[k]))); seen = {}; members = []
        for s in perms:
            mv = frozenset(tuple(v[s[i]] for i in range(n)) for v in rep)
            if mv in seen: continue
            seen[mv] = 1; members.append((s, [tuple(v[s[i]] for i in range(n)) for v in rep]))
        blocks.append({"rep": [list(v) for v in rep], "members": members, "nv": len(rep)})
    # parameter layout: [ for each block: nv*n vertex coords ] + [ N_supp coeff ] + [1 linear d]
    offs = []; p = 0
    for b in blocks: offs.append(p); p += b["nv"]*n
    coff = p; p += len(supp); doff = p; p += 1; P = p
    # sample generic points, build Jacobian rows
    rng2 = np.random.default_rng(7); rows = []
    sigxcache = {}
    def sigma_inv_x(s, x):  # u[j] = x[s^{-1}[j]]
        sinv = [0]*n
        for i in range(n): sinv[s[i]] = i
        return [x[sinv[j]] for j in range(n)]
    for _ in range(max(3*P, 600)):
        x = tuple(int(t) for t in rng2.integers(-12, 13, size=n))
        row = np.zeros(P)
        for bi, (k, b) in enumerate(zip(supp, blocks)):
            ct = cvec[bi]
            os = 0.0
            for (s, mverts) in b["members"]:
                vals = [sum(mverts[j][i]*x[i] for i in range(n)) for j in range(b["nv"])]
                jstar = int(np.argmax(vals)); os += vals[jstar]
                # derivative wrt REP vertex jstar (member vertex = sigma . rep vertex jstar): += ct * sigma^{-1} x
                u = sigma_inv_x(s, x)
                base = offs[bi] + jstar*n
                for i in range(n): row[base+i] += ct * u[i]
            row[coff+bi] = os                      # d/dc_t
        row[doff] = sum(x)                          # d/dd
        rows.append(row)
    Jac = np.array(rows)
    # rank / tangent
    sv = np.linalg.svd(Jac, compute_uv=False)
    tol = max(Jac.shape) * sv[0] * 1e-9
    rank = int((sv > tol).sum()); tangent = P - rank
    # gauge vectors (construct dv part; then solve the dc/dd compensation that makes Jac@g ~ 0)
    cold = Jac[:, doff]; cdd = float(cold @ cold)
    def compensate(g):  # add the best delta-d (and per-block delta-c via scaling already set) to minimize Jac@g
        r = Jac @ g
        if cdd > 0: g[doff] += -float(r @ cold) / cdd
        return g
    gauge = []
    for bi, b in enumerate(blocks):
        # scaling: dv = rep vertex coords; dc = -c_t
        g = np.zeros(P)
        for j in range(b["nv"]):
            for i in range(n): g[offs[bi]+j*n+i] = b["rep"][j][i]
        g[coff+bi] = -cvec[bi]; gauge.append(g)
        # translation in each coord dir: dv = e_dir for all vertices of block; dd compensates the induced linear term
        for dirc in range(n):
            g = np.zeros(P)
            for j in range(b["nv"]): g[offs[bi]+j*n+dirc] = 1.0
            gauge.append(compensate(g))
    G = np.array(gauge)
    resid = np.linalg.norm(Jac @ G.T, axis=0)
    gtol = max(1.0, np.linalg.norm(Jac)) * 1e-6
    Gin = G[resid < gtol]
    gauge_dim = int(np.linalg.matrix_rank(Gin, tol=1e-7)) if len(Gin) else 0
    essential = tangent - gauge_dim
    print(f"\n===== max_{n} infinitesimal rigidity =====")
    print(f"  params P={P} (verts {sum(b['nv'] for b in blocks)*n} + coeffs {len(supp)} + d 1) | Jac rows={Jac.shape[0]}")
    print(f"  Jacobian rank={rank} | TANGENT dim={tangent}")
    print(f"  gauge directions in tangent: {len(Gin)}/{len(G)} | GAUGE dim={gauge_dim}")
    print(f"  ESSENTIAL deformation dim = {essential}   => {'RIGID (isolated up to symmetry/scaling)' if essential<=0 else 'has '+str(essential)+' continuous deformation(s) -- NOT rigid'}")
    return essential

for n in (5, 6):
    rigidity(n)
