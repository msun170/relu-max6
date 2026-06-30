# Direct test: can a standard 2-HIDDEN-LAYER ReLU net fit max_7 EXACTLY (real weights, not lattice-restricted)?
# y = w3 . relu(W2 relu(W1 x + b1) + b2) + b3. Adam, manual backprop on GPU. Train on many random points.
# If train+test MSE -> machine-zero at finite width => max_7 IN V_2 (extract+verify). Firm positive plateau => OUT-leaning.
# (Caveat: non-convex, so a plateau is weak evidence; but reaching ~0 is strong and yields a construction.)
import sys, time
import numpy as np
import cupy as cp
n = 7; t0 = time.time()
rng = cp.random.RandomState(0)
def data(m, seed):
    r = cp.random.RandomState(seed); X = r.standard_normal((m, n)).astype(cp.float32) * 3.0
    y = cp.max(X, axis=1, keepdims=True); return X, y
Xtr, ytr = data(20000, 1); Xte, yte = data(8000, 2)
yscale = float(cp.std(ytr)); ytr = ytr / yscale; yte = yte / yscale     # normalize target

for H in (64, 128, 256, 512):                                            # increasing width
    r = cp.random.RandomState(7)
    W1 = (r.standard_normal((n, H)) * (2.0/n)**0.5).astype(cp.float32); b1 = cp.zeros((1, H), cp.float32)
    W2 = (r.standard_normal((H, H)) * (2.0/H)**0.5).astype(cp.float32); b2 = cp.zeros((1, H), cp.float32)
    w3 = (r.standard_normal((H, 1)) * (1.0/H)**0.5).astype(cp.float32); b3 = cp.zeros((1, 1), cp.float32)
    params = [W1, b1, W2, b2, w3, b3]
    mAdam = [cp.zeros_like(p) for p in params]; vAdam = [cp.zeros_like(p) for p in params]
    lr = 3e-3; b1a, b2a, eps = 0.9, 0.999, 1e-8
    M = Xtr.shape[0]; bs = 4000
    best = 1e9
    for it in range(1, 40001):
        idx = rng.randint(0, M, size=bs)
        x = Xtr[idx]; t = ytr[idx]
        z1 = x @ W1 + b1; a1 = cp.maximum(z1, 0)
        z2 = a1 @ W2 + b2; a2 = cp.maximum(z2, 0)
        out = a2 @ w3 + b3
        diff = out - t; loss = float(cp.mean(diff**2))
        g_out = (2.0/bs) * diff
        gw3 = a2.T @ g_out; gb3 = cp.sum(g_out, axis=0, keepdims=True)
        ga2 = g_out @ w3.T; gz2 = ga2 * (z2 > 0)
        gW2 = a1.T @ gz2; gb2 = cp.sum(gz2, axis=0, keepdims=True)
        ga1 = gz2 @ W2.T; gz1 = ga1 * (z1 > 0)
        gW1 = x.T @ gz1; gb1 = cp.sum(gz1, axis=0, keepdims=True)
        grads = [gW1, gb1, gW2, gb2, gw3, gb3]
        for i, (p, g) in enumerate(zip(params, grads)):
            mAdam[i] = b1a*mAdam[i] + (1-b1a)*g; vAdam[i] = b2a*vAdam[i] + (1-b2a)*g*g
            mh = mAdam[i]/(1-b1a**it); vh = vAdam[i]/(1-b2a**it)
            p -= lr * mh/(cp.sqrt(vh)+eps)
        if it % 4000 == 0:
            z1=Xte@W1+b1; a1=cp.maximum(z1,0); z2=a1@W2+b2; a2=cp.maximum(z2,0); o=a2@w3+b3
            te = float(cp.sqrt(cp.mean((o-yte)**2)))                     # test RMSE (relative, since y normalized)
            best = min(best, te)
            print(f"  H={H} it{it}: train_mse={loss:.3e} test_rmse={te:.5f}  [{time.time()-t0:.0f}s]", flush=True)
        if it > 8000 and lr > 3e-4: lr = 1e-3
    print(f"H={H}: best test RMSE = {best:.5f}  ({'~0 => max_7 likely IN V_2' if best<1e-3 else 'plateau => OUT-leaning (or GD stuck)'})", flush=True)
