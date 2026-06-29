# The depth-2 lower bound as a Minkowski-summand problem

This note recasts "does `max_n` need three hidden layers" as a question about Minkowski summands. The
reformulation is exact and real-weight valid. It gives a clean, complete proof of the one-hidden-layer case
and isolates precisely the one structural fact whose analogue is missing at two hidden layers. Everything
marked **Theorem/Proof** is rigorous; everything marked **Open** is not claimed.

Notation: `Delta = conv{e_1, ..., e_n}` is the standard simplex, and `max_n(x) = max_i x_i = h_Delta(x)`
is its support function. `h_P(x) = max_{p in P} <p, x>`. Minkowski sum is `+`, dilation by `t >= 0` is `tP`.

## 1. The support-function dictionary (exact)

A ReLU network with one hidden layer computes, on each linear region, an affine function; the convex ones
are exactly `f = sum_j a_j (w_j . x)_+ + affine`. With `a_j > 0` this is `h_Z` for the zonotope
`Z = sum_j [0, a_j w_j]`. Allowing signs, a one-hidden-layer convex CPWL function is a **difference of two
zonotope support functions**.

A ReLU network with two hidden layers computes signed sums of `ReLU` applied to one-hidden-layer functions.
A single second-layer unit computes `ReLU(h_{Z_1} - h_{Z_2}) = max(h_{Z_1}, h_{Z_2}) - h_{Z_2}`, and
`max(h_{Z_1}, h_{Z_2}) = h_{conv(Z_1 u Z_2)}` is the support function of the **join** of two zonotopes.
Hence a two-hidden-layer CPWL function is a signed combination

  f = sum_t c_t h_{Q_t} + (linear),    c_t in R,   each Q_t = conv(Z_{t,1} u Z_{t,2}) a join of two zonotopes.

Throughout, a **block** is a join of two zonotopes (zonotopes themselves are the degenerate join `Z u Z`).

### The reduction

Split the sum by sign and move negative terms across. Writing `h_{point} = linear`, the identity
`max_n = sum_t c_t h_{Q_t} + linear` becomes an equality of support functions

  h_Delta + sum_{c_t < 0} |c_t| h_{Q_t}  =  sum_{c_t > 0} c_t h_{Q_t} + linear,

i.e. `h_Delta + h_N = h_M` with `M = sum_{c_t>0} c_t Q_t` and `N = sum_{c_t<0} |c_t| Q_t`. Support functions
add under Minkowski sum, and `h_A = h_B` iff `A = B` (up to translation), so this is exactly:

> **Reduction.** `max_n` is computable with two hidden layers iff there exist polytopes `M, N`, each a
> Minkowski sum of (nonnegatively dilated) joins of two zonotopes, with
>
>   **Delta + N = M.**
>
> Equivalently, `Delta` is a Minkowski **summand** of some `M` in the monoid generated under Minkowski
> sum by joins of two zonotopes. (The one-hidden-layer statement is identical with "joins of two
> zonotopes" replaced by "zonotopes".)

This is real-weight valid: `c_t` and the zonotope generators are arbitrary reals.

## 2. One hidden layer: a complete proof

**Theorem 1.** `max_n` is computable with one hidden layer iff `n <= 2`.

*Proof.* For `n <= 2`, `max_2(x) = (x_1+x_2)/2 + |x_1-x_2|/2` is a single absolute value (a zonotope/segment),
and `max_1` is linear. Conversely suppose `max_n` is one-hidden-layer. By the Reduction (zonotope version)
there is a zonotope `M` with `Delta + N = M` where `N` is a Minkowski sum of zonotopes, hence itself a
zonotope. So `Delta` is a Minkowski summand of the zonotope `M`. Now use two classical facts:

  (a) **Every Minkowski summand of a zonotope is a zonotope.** A polytope is a zonotope iff all its
      2-faces are centrally symmetric (Bolker; Schneider, *Convex Bodies*, Thm 3.5.2). If `M = Delta + N`
      and `M` is a zonotope, each 2-face of `Delta` is a Minkowski summand of a 2-face of `M`, which is
      centrally symmetric; a summand of a centrally symmetric polygon is centrally symmetric. So every
      2-face of `Delta` is centrally symmetric, i.e. `Delta` is a zonotope.

  (b) **The simplex `Delta = Delta^{n-1}` is a zonotope iff `n <= 2`.** For `n >= 3` it has triangular
      2-faces `conv{e_i,e_j,e_k}`, which are not centrally symmetric.

Together, `Delta` a summand of a zonotope forces `n <= 2`. ∎

The same Hessian/edge picture proves it concretely: the distributional Hessian of a one-hidden-layer convex
`f` is `sum_a D_a (a a^T) H^{n-1}|_{a^perp}`, a sum of rank-1 measures on the hyperplanes `a^perp`, mutually
singular. `D^2 max_n` is supported on the braid walls `{x_i = x_j}` with eigendirection the root `e_i - e_j`.
Matching forces every generator direction `a` to be a root, and root zonotopes are segments/parallelograms
whose joins we enumerate; one checks `Delta` is absent for `n >= 3`. The summand proof above is the clean
form.

## 3. Two hidden layers: where the proof breaks, exactly

At two hidden layers the blocks are **joins**, and a join is generally **not centrally symmetric**: the
bridge 2-faces of `conv(Z_1 u Z_2)` can be triangles. So `M = sum c_t Q_t` need not be a zonotope, fact (a)
does not apply, and `Delta` is no longer forbidden as a summand on symmetry grounds. Concretely, the
triangular 2-faces of `Delta` -- the obstruction in Theorem 1 -- can now be supplied by bridge faces of
joins. This is not a gap in the write-up; it is the actual mathematical reason the one-layer argument does
not extend, stated precisely.

What survives rigorously at two layers is the **codimension-1 necessary condition** (real-weight valid):

**Lemma (directional density).** Write `D^2 f = sum_d rho_d(x) (d-hat d-hat^T) H^{n-1}|_{d^perp}`, grouping
the rank-1 Hessian contributions of a block sum by edge direction `d` (`rho_d` is the signed,
length-weighted density of `d`-parallel edges of the blocks, a piecewise-constant function on `d^perp`).
If `sum_t c_t h_{Q_t} = max_n` up to linear terms, then:
  - for every non-root direction `d`, `rho_d ≡ 0` on `d^perp` (all off-braid edge contributions cancel);
  - for a root `d = e_i - e_j`, `rho_d` equals the simplex density on the braid wall `{x_i = x_j}`.

This is exactly the constraint our exact computations enforce, and it is what the J/NB split localizes: the
root-wall matching (J) and the non-root cancellation (NB) are individually satisfiable but jointly
over-determined inside the lattice family.

## 4. What is proved, and the exact open lemma

**Theorem 2 (finite, exact; `check_weight2_max7_infeasible.py`).** There is no `Delta_7 + N = M` with `M, N`
built from joins of zonotopes whose vertices lie in the weight-2 lattice `2*Delta^6` (the root model). The
certificate is a rational functional annihilating every block in that family while pairing nontrivially with
`max_7`; it is confirmed mod three large primes.

The full lower bound is Theorem 2 with the lattice restriction removed. By the Reduction, that is:

> **Open lemma (normal form / summand closure).** If `Delta` is a Minkowski summand of a sum of joins of
> two zonotopes, then it is one using only zonotopes with root edges (equivalently, lattice blocks).

Granting the Open lemma, Theorem 2 gives the separation: `max_7` needs three hidden layers, the first CPWL
function proven to need depth 3 at unbounded width. The lemma is the two-hidden-layer analogue of fact (a):
fact (a) said "summands of zonotopes are zonotopes"; we need "summands of sums-of-joins are controlled by
their root part". No such structure theorem is known (confirmed by a 2026 literature review: no a priori
width/complexity bound exists for exact fixed-depth representation).

### Evidence the open lemma is true (not a proof)

1. **Off-lattice probes** (`realweight_test.py`). Augmenting the root family with blocks carrying genuinely
   non-root edge directions -- complexity-2, complexity-3, and off-lattice (coordinate sum != 0) -- never
   brings `max_7` into the span. Positive controls through the same pipeline give `max_5, max_6` IN and
   `max_7` OUT, so the test detects representability. This is the first evidence beyond the lattice.
2. **Bounded complexity = decomposition-polyhedron emptiness** (`decomp_polyhedron.py`). The
   Brandenburg-Grillo-Hertrich decomposition polyhedron of `max_7` is empty at weight-2 complexity and
   nonempty for `max_6`; Theorem 2 is exactly its bounded-complexity emptiness.

### The concrete attack on the open lemma

The triangular 2-faces of `Delta` are what must be supplied by **bridge faces** of joins (Section 3). A
single join of two zonotopes contributes a bounded stock of bridge triangles, while `Delta` has `C(n,3)`
triangular 2-faces, one per triple, each needing the three root edges `e_i-e_j, e_j-e_k, e_k-e_i` to close.
The codimension-2 closing relations (edges around each 2-face sum to zero) couple the directional densities
across triples for a *single* fixed block set. Turning "few bridge triangles per block, many triangles to
cover, closing relations couple them" into a width lower bound that grows with `n` -- and then into the
normal form -- is the open crux. This is the cross-codimensional rigidity: each block is one polytope, so
its bridge faces across all the `C(n,3)` corners are not independent.
