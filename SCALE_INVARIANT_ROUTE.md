# The scale-invariant certificate route

The two barriers in `PROOFS.md` (counting cannot grow; lattice/parity is scale-sensitive) say a proof of the
real-weight separation must produce a **scale-invariant** obstruction: a single linear functional that
annihilates every join-of-two-zonotopes regardless of generator scale, yet is nonzero on `max_n`. The natural
home for such a functional is a **signed measure** `mu` on input space, paired with functions by
`mu(f) = integral f dmu`. A measure does not refer to any lattice, so it is automatically scale-invariant.
This note develops the route, proves the one-hidden-layer case inside it (scale-invariantly, no lattice), and
states the exact two-hidden-layer gap.

## One hidden layer, scale-invariantly

A one-hidden-layer convex function is a difference of zonotope support functions, and
`h_Z(x) = sum_i (g_i . x)_+` with arbitrary real generators `g_i`. So a measure `mu` annihilates **every**
zonotope support function iff

  `T(v) := integral (v . x)_+ dmu(x) = 0`  for all `v`.

Using `(v.x)_+ = (v.x + |v.x|)/2`, this splits into `integral x dmu = 0` (the linear part) and the **cosine
transform** condition `integral |v.x| dmu(x) = 0` for all `v`.

**Key fact (Aleksandrov):** the cosine transform is injective on *even* measures. Hence a measure with
vanishing cosine transform has zero even part, i.e. `mu` must be **odd** (`mu(-A) = -mu(A)`). Conversely every
odd `mu` has vanishing cosine transform (`|v.x|` is even), and then `T(v) = (1/2) v . integral x dmu`, which
vanishes iff `integral x dmu = 0`. So:

> A measure annihilates all zonotope support functions  iff  it is **odd** with `integral x dmu = 0`.

For such `mu`, only the **odd part** of a test function is seen, and the odd part of `max_n` is
`(max_n(x) - max_n(-x))/2 = (max(x) + min(x))/2`. So `mu(max_n) = integral (max + min) dmu`. Since `max + min`
is **not linear** for `n >= 3` (for `n = 3` it is `sum - median`), there exist odd balanced `mu` with
`mu(max_n) != 0`.

**Theorem 1, scale-invariant form.** For every `n >= 3` there is a finite signed odd measure `mu` with
`integral x dmu = 0` (so `mu` kills every zonotope at every scale) and `mu(max_n) != 0`; hence `max_n` is not
one hidden layer. *Explicit `n = 3` witness* (`measure_cert_k1_exact.py` / direct construction): on the four
pairs `+/-(2,-1,0), +/-(1,1,-2), +/-(0,1,1), +/-(1,0,0)` with weights `a = (-3/7, -1/7, -2/7, 1)`, one has
`sum a_j x_j = 0` exactly and `mu(max_3) = 2 sum a_j (max+min)(x_j) = 6/7 != 0`. No lattice appears anywhere.

This is the clean dual of the summand proof in `PROOFS.md` Section 2: zonotopes are exactly the centrally
symmetric building blocks, and the certificate is precisely a measure that only sees central asymmetry.

(Methodological note, recorded honestly: a first *numerical* attempt that sampled finitely many directions
`v` produced a spurious certificate -- under-sampling the cosine condition. The exact chamber/odd analysis
above corrects it; the lesson mirrors the weight-3 vacuity trap -- continuous conditions must be handled
exactly, not by finite sampling.)

## Two hidden layers: the exact gap

A two-hidden-layer function is a signed sum of **join** support functions `h_Q = max(h_{Z_1}, h_{Z_2})`. The
scale-invariant obstruction we seek is a measure `mu` with

  `integral max(h_{Z_1}, h_{Z_2}) dmu = 0` for all zonotopes `Z_1, Z_2`,  and  `mu(max_n) != 0`.

The one-layer proof used that zonotopes are centrally symmetric, so their certificates are exactly the odd
measures. **Joins are not centrally symmetric**, so the odd-measure trick does not extend: an odd `mu` does
not annihilate `max(h_{Z_1}, h_{Z_2})` in general (the `max` is nonlinear in the generators, and its odd part
is itself nonlinear). This is the precise reason the route does not close at `k = 2`, stated structurally
rather than as a vague gap.

What is needed is the second-order analogue of "odd = cosine-transform kernel": a characterization of the
measures annihilating all `max(h_{Z_1}, h_{Z_2})`. Because `max(a,b) = (a + b + |a - b|)/2`, annihilating all
joins means annihilating all `h_{Z_1} + h_{Z_2}` (already handled: the one-layer condition) **and** all
`|h_{Z_1} - h_{Z_2}|` -- the absolute value of a difference of zonotope support functions. So the open object
is exactly:

> Characterize the signed measures `mu` with `integral |h_{Z_1} - h_{Z_2}| dmu = 0` for all zonotopes
> `Z_1, Z_2`. The separation for `max_n` is then: does some such `mu` (also odd, balanced) have
> `mu(max_n) != 0`?

This is a genuine, scale-invariant, lattice-free reformulation of the open problem: a "second-order cosine
transform" on differences of zonoids. It is the concrete target the two barriers point to, and -- unlike the
fixed-lattice certificate or the volume parity -- it cannot be defeated by rescaling, because measures carry
no scale.
