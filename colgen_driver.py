import sys, time, numpy as np
sys.path.insert(0, "C:/Users/nuswe/dev/active/relu-max-depth/lowerbound")
from pricer import Pricer

n = 7
w = int(sys.argv[1]) if len(sys.argv) > 1 else 4
m = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
NIT = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
pr = Pricer(n, w, m=m)
print(f"weight-{w}: {len(pr.WP)} pts, {len(pr.gens)} gens, m={m}", flush=True)
Q = np.linalg.qr(np.column_stack([pr.X[:, i] for i in range(n)]))[0]
bn = np.linalg.norm(pr.bmax)
t0 = time.time()
for it in range(NIT):
    r = pr.bmax - Q @ (Q.T @ pr.bmax); relres = np.linalg.norm(r)/bn; k = Q.shape[1]-n
    if relres < 1e-9:
        print(f"iter {it}: relres {relres:.2e} -> max{n} IN weight-{w} ({k} blocks, k/m={(k+n)/m:.2f})", flush=True); break
    lam = r
    cs = pr.price_single(lam); single_max = cs[0][0] if cs else 0.0
    pool = pr.pool_zonos(lam)
    jmax, jcol = pr.price_join(lam, pool)
    # choose best violator column
    best_val, best_col = 0.0, None
    if cs:
        gt, p = cs[0][1]; c = pr.zono_col(gt, p); best_val, best_col = abs(float(lam@c)), c
    if jmax > best_val: best_val, best_col = jmax, jcol
    if best_col is None or best_val < 1e-6*np.linalg.norm(lam):
        print(f"iter {it}: NO VIOLATOR. complete single-zono max={single_max:.2e}, join-pool max={jmax:.2e}, relres={relres:.4e}", flush=True)
        print(f"  => lambda certifies max{n} OUT of single-zonotopes (rigorous) and joins (pool); blocks={k}", flush=True); break
    w_ = best_col - Q@(Q.T@best_col); nw = np.linalg.norm(w_)
    if nw < 1e-9*np.linalg.norm(best_col): continue
    Q = np.column_stack([Q, w_/nw])
    if k+n > 0.6*m:
        print(f"iter {it}: basis approaching m; stop. relres={relres:.4e}, single_max={single_max:.2e}, jmax={jmax:.2e}", flush=True); break
    if it % 100 == 0 or it < 3:
        print(f"iter {it}: relres={relres:.4e}, blocks={k}, single_max={single_max:.2e}, jmax={jmax:.2e}  [{time.time()-t0:.0f}s]", flush=True)
print(f"done: blocks={Q.shape[1]-n}, relres={np.linalg.norm(pr.bmax-Q@(Q.T@pr.bmax))/bn:.4e}  [{time.time()-t0:.0f}s]", flush=True)
