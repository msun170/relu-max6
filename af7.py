import sys, numpy as np, time
sys.path.insert(0,"C:/Users/nuswe/relu-max6")
from intersection import project_basis, perm_vertices, verts_of, Bform, Bbilinear, af_defect
import core
for n in (7,):
    d=n-1; Bp=project_basis(n); t0=time.time()
    A=perm_vertices(n,Bp)
    S=verts_of(frozenset(tuple(2 if k==i else 0 for k in range(n)) for i in range(n)),n,Bp)
    print(f"n={n}: vol(A)={np.linalg.norm(0)+__import__('intersection').vol(A):.4f} exact={n**(n-2)*np.sqrt(n):.4f}",flush=True)
    print(f"  AF/Hodge-index defect max_n = {af_defect(S,A,d):.4f}   ({time.time()-t0:.0f}s)",flush=True)
