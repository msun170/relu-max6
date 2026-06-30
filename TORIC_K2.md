# Toward a two-hidden-layer criterion: max_n in the permutohedral variety

Groundwork for the toric/intersection-theoretic route to the depth-2 lower bound, following the k=1 criterion
of "Toric geometry of ReLU neural networks" (arXiv:2509.05894) and pushing toward k=2. The key observation:
the relevant toric variety is not generic -- it is the **permutohedral variety**, whose intersection theory is
exceptionally rich (matroids, the Chow ring, combinatorial Hodge theory). That is the promising setting.

## 1. The dictionary for max_n

`max_n(x) = max_i x_i` is the support function of the simplex `Delta_n = conv{e_1,...,e_n}`, and it is linear
on the **braid fan** B_n (the normal fan of the permutohedron / the type-A Coxeter fan): the maximal cone
`sigma_i = {x : x_i >= x_j for all j}` carries the linear piece `m_{sigma_i} = e_i` (since `max_n = x_i` there).

The toric variety `X(B_n)` of the braid fan is the **permutohedral variety** (a smooth projective toric
variety, the iterated blow-up of P^{n-1} along coordinate subspaces). Piecewise-linear functions on B_n =
torus-invariant Q-divisors on `X(B_n)` = **generalized permutohedra** = submodular functions. `max_n = h_{Delta_n}`
is the divisor `D_f` of the simplex (a vertex of the deformation cone).

A `k`-hidden-layer realization corresponds (after symmetrizing) to a signed decomposition
`D_{max_n} = sum_t c_t D_{Q_t}` where each `Q_t` is a join of two zonotopes (`P_2`), each `D_{Q_t}` itself a
divisor / generalized permutohedron. **Zonotopes are exactly the graphical generalized permutohedra** (sums of
root segments `[0, e_a - e_b]`), so the `D_{Q_t}` are joins of graphical pieces. The depth-2 question is thus:

> Is `D_{max_n}` in the signed span (inside `Pic(X(B_n)) ⊗ Q`, equivalently the space of PL functions on B_n)
> of support functions of joins of two graphical zonotopes?

## 2. The k=1 criterion is our gradient-jump obstruction

Intersection number of `D_f` with the torus-invariant curve `V(tau)` dual to a wall `tau` between `sigma_i`
and `sigma_j` is `D_f . V(tau) = <m_{sigma_i} - m_{sigma_j}, u_tau> = <e_i - e_j, u_tau>` -- exactly the
gradient JUMP across the braid wall. The k=1 theorem (2509.05894, Thm 1): a one-hidden-layer function has
**equal** intersection numbers on all walls of each hyperplane. `max_n` violates this: on the hyperplane
`{x_i = x_j}` the jump is `e_i - e_j` where `i,j` are the top two, and `0` elsewhere (other coordinate larger).
This is precisely our distributional-Hessian / J-condition; it reproves `max_n` not 1-layer for `n >= 3`.

So the toric language *is* our gradient-jump analysis. The k=1 criterion lives at **codimension 1**.

## 3. What a k=2 criterion must add, and where the real tool is

Two new features appear at k=2:
1. **Refined fan.** Joins introduce new walls (the join creases `h_{Z_1} = h_{Z_2}`), so `D_f` lives on a
   common refinement of B_n; the relevant intersection numbers move to higher codimension.
2. **Cancellation.** The output layer is a SIGNED sum, so the intersection number at a wall is
   `sum_t c_t (D_{Q_t} . V(tau))` -- a signed combination. This breaks the clean "equal per hyperplane" form;
   it is exactly the (J)+(NB) over-determination we found (creases match, bridges cancel, jointly rigid).
3. **Structure constraint.** The `D_{Q_t}` are not arbitrary divisors -- they are joins of graphical
   zonotopes, a constrained sub-monoid of the nef cone.

The natural home for a k=2 obstruction is therefore the **codimension-2 intersection numbers** -- the
triangular 2-faces of `Delta_n` (the corners `C_{ijk}`), which are torus-invariant **surfaces** in
`X(B_n)`. The restriction of `D_{max_n}` to such a surface has a degree / self-intersection; a join's
restriction is constrained (its bridge faces). This is the same codim-2 triangle structure we studied, now
with intersection-theoretic invariants attached.

## 4. The genuinely new tool the permutohedral variety offers

`X(B_n)` has a **Chow ring** `A^*(X(B_n))` with **Poincare duality** and the **Hodge-Riemann relations**
(this is the combinatorial Hodge theory of Adiprasito-Huh-Katz, proved exactly for the permutohedral /
Bergman-fan setting). Hodge-Riemann gives **inequalities** on intersection numbers (log-concavity of mixed
volumes / the Alexandrov-Fenchel inequalities, sharpened). The concrete hope:

> A Hodge-Riemann / Alexandrov-Fenchel inequality satisfied by every signed combination of join-of-zonotope
> divisors, but VIOLATED by `D_{max_n}` for `n >= 7`, would be a depth-2 lower bound -- and unlike covering
> or the measure route, an inequality is not defeated by cancellation (it is a closed, exact condition on the
> divisor class) and not blocked by density (it is algebraic, not approximate).

This is the first candidate obstruction in this whole investigation that is simultaneously: (a) exact /
algebraic (so not density-blocked), (b) an inequality on intersection numbers (so not cancellation-blind like
counting), and (c) supported by powerful existing machinery (combinatorial Hodge theory on the permutohedral
variety). It is the right place to look.

## 5. Concrete open piece

Express the join-of-zonotope divisor classes in `A^1(X(B_n))` and compute the relevant degree-2 intersection
numbers (mixed volumes) `D_{Q_t}^2`, `D_{Q_t} . D_{max_n}`, etc. Find a quadratic form (from Hodge-Riemann) on
the divisor classes that is `<= 0`-definite on the join sub-monoid but evaluates `> 0` on `D_{max_n}` at
`n = 7`. The mixed-volume computations are finite and checkable for fixed n; this is the next concrete step,
and it connects directly to the mixed-volume tools Bakaev et al already use for the upper bound.

## 6. The matroid refinement (why an inequality can be WEIGHT-INDEPENDENT)

The single hardest obstacle to a depth-2 lower bound is that the network **weight is unbounded**: the rank /
membership test decides `max_n in span(weight-w joins)` for each fixed `w` (we proved OUT for `w = 2, 3`), but
there are infinitely many `w`, and as `w -> infinity` the family becomes dense (`max_7 in closure`). A linear
obstruction must therefore be re-proved at every weight and cannot, alone, close the problem. The reason to
believe Hodge-Riemann is different is that it is **structural** rather than weight-graded, via matroids:

- **The simplex is a matroid polytope.** `Delta_n = conv{e_i}` is the matroid polytope of the rank-1 uniform
  matroid `U_{1,n}` (the hypersimplex `Delta_{1,n}`). So `D_{max_n}` is the divisor class of `U_{1,n}` on the
  permutohedral variety.
- **Zonotopes are matroid polytopes too.** A graphical zonotope (sum of root segments `[0, e_a - e_b]`) is the
  matroid polytope of the corresponding graphic matroid; general zonotopes <-> representable matroids. A join
  of two zonotopes is built from two matroids in complementary coordinate blocks.
- **AHK Hodge theory is matroid-intrinsic.** Adiprasito-Huh-Katz prove Poincare duality, hard Lefschetz, and the
  **Hodge-Riemann relations** for the Chow ring `A^*(M)` of ANY matroid `M` -- a property of the matroid, not of
  any weight grading. The Hodge-Riemann inequalities (log-concavity of the characteristic polynomial, etc.) hold
  for `U_{1,n}` and for every graphic matroid, uniformly.

So the hoped-for obstruction has the right *type* to be weight-independent: a Hodge-Riemann inequality is a fixed
algebraic inequality attached to the *matroid* of each piece, preserved under the join construction at **every**
weight. If joins-of-graphic-matroid divisor classes satisfy an AHK inequality that the `U_{1,n}` class violates
for `n >= 7`, the violation cannot be repaired by raising the weight -- it is a property of `U_{1,n}` itself.
This is precisely the ingredient the linear/rank route lacks. (It does not yet *prove* the bound -- the signed
output sum still has to be controlled -- but it identifies a candidate obstruction of the correct, scale-free
kind, and it ties the problem to a developed theory.)

## 7. Computational infrastructure (built and validated, `intersection.py`)

The degree-2 intersection numbers are mixed volumes, and these are concretely computable for our divisor
classes. `intersection.py` builds them and validates against ground truth:

- **Mixed-volume calculator** via the Minkowski-volume polynomial: `vol(s*D + A)` fit as a degree-`d` polynomial
  in `s`, with `[s^2]` coefficient giving `V(D, D, A^{d-2})` (and polarization for the bilinear form).
- **Validation:** `vol(permutohedron)` reproduces the exact value `n^{n-2} sqrt(n)` to machine precision
  (`32.0000` at `n=4`, `279.5085` at `n=5`), confirming the calculator.
- **Hodge-Riemann verified for our objects:** the degree-1 form `B(D, D') = V(D, D', A^{d-2})` (with `A` the
  ample permutohedron) has signature **(+1, rest -)** -- Lorentzian, one positive eigenvalue -- over
  `{D_{max_n}} u {join blocks}` at `n = 4, 5, 6`. This is the AHK Hodge-Riemann relation holding numerically for
  exactly the classes in play, so the `n = 7` computation is a matter of scale, not of new theory.

**Remaining (the research step):** find the specific AHK inequality on the matroid Chow ring that separates
`U_{1,n}` from the graphic-join sub-monoid at `n = 7`. The infrastructure to test candidate inequalities now
exists; the search for the right one is the open problem.

## 8. NO-GO for the intersection-number route (the screen's real result), and the redirect

Screening the Hodge / Alexandrov-Fenchel inequalities forced a sharp and important conclusion: **no intersection
number can separate `max_n` from the join family.** The argument is one line, and the computation confirms it.

**Lemma (intersection theory factors through `N^1`).** On the permutohedral variety `X(B_n)`, every top
intersection number `D_1 . D_2 ... D_{d}` is a *symmetric multilinear* function of the numerical classes
`[D_i] in N^1(X)_R`. Hence any quantity assembled from intersection numbers of a class `[D]` -- self-
intersections, the Hodge-Riemann Gram matrix, AF defects, all of it -- is a function of the single class
`[D] in N^1(X)_R`.

**Corollary (collapse to the linear test).** Two-hidden-layer realizability of `max_n` is *exactly*
`h_{Delta_n} in span_R{ h_Q : Q a join of two zonotopes }` -- the second-layer weights are unconstrained reals,
so the support-function decomposition `max_n = sum_t c_t h_{Q_t}` is a plain linear relation in `N^1(X)_R`. If
that relation holds, then `[D_{max_n}] = sum_t c_t [Q_t]` *as classes*, and every intersection-theoretic
identity for `max_n` is automatically the same combination of the joins' -- nothing can be violated. So
intersection theory detects non-representability **only** through `[D_{max_n}] notin span_R{[Q_t]}`, which is
precisely the linear membership (rank) test we already run. Hodge-Riemann adds inequalities, but an inequality
on a multilinear quantity still factors through the class, so it cannot see anything the span does not.

**Why that kills the route: the density wall.** The span `{h_Q}` over *all* weights is dense at `max_n`
(established earlier: `max_7` is approximable by join-combinations to distance `<= 0.0017`, the residual driven
toward `0` as weight grows). A *continuous* linear functional vanishing on the span therefore vanishes on its
closure, hence on `max_n`. So no continuous functional -- in particular no AF/Hodge-Riemann inequality on
intersection numbers -- separates `max_n` from the joins.

**The computation confirms it.** The AF / Hodge-index defect `delta(D) = B(A,D)^2 - B(A,A)B(D,D)` (normalized)
for `D = max_n` is positive and **smooth across the representability boundary**: `3.69, 35.3, 539, ...` at
`n = 4,5,6,...` with `n=5,6` representable and `n=7` the candidate non-representable case. There is no sign
flip, no kink, no qualitative change where representability is supposed to fail -- exactly as the Lemma predicts.
The Hodge structure is real (the Gram is Lorentzian at every `n`), but it is blind to the depth boundary.

**Redirect (where the obstruction must live).** Proving `max_n notin span` (exactly, not approximately)
requires a **discontinuous / unbounded** linear functional `lambda` with `lambda(h_Q) = 0` for all joins but
`lambda(h_{Delta_n}) != 0` -- a *singular* measure / distribution certificate. Continuous certificates are ruled
out by density; the certificate must exploit the **exact arithmetic** of `max_n` (rational vertex coordinates,
parity / valuation data), not metric closeness. This is exactly the self-similar measure-annihilator program of
`SCALE_INVARIANT_ROUTE.md`, now *sharpened by a proof that the annihilator is necessarily singular*. The matroid
distinction `U_{1,n}` vs graphic matroids (Sec. 6) may still supply the certificate's exact defining data -- but
it enters as exact algebra on the Chow ring, **not** as an inequality on mixed volumes.

This is the productive outcome of the toric program: it does not yield the bound, and we can now say *precisely
why* (multilinearity + density), which retires an entire elegant-looking family of attempts and concentrates the
remaining effort on the one route the no-go leaves open -- a singular, scale-invariant annihilating functional.

## 9. CAVEAT to the no-go: the density premise is not actually established (and if it fails, the separation is provable by a FINITE certificate)

The no-go's teeth come entirely from the **density premise** `max_7 in closure(V_2)`, i.e. the residual floor
`f_infty = dist(max_7, closure(V_2)) = 0`. A clean re-measurement undercuts that premise:

- **Vacuity-safe column generation** (basis `k+n < 0.6 m`, `m = 9000`) gives the rigorous weight-2 floor
  `f(2) = 0.0308` -- and this is EXACT (all 784 weight-2 blocks exhausted, no violator). The earlier "density
  distance 0.0017 at weight-2" is **inconsistent** with this and was almost certainly a vacuity artifact (the
  weight-3 vacuity traps in this exact pipeline are documented). So the number underpinning `f_infty = 0` is
  unreliable.
- Vacuity-safe floors at higher weight (capped runs, upper bounds): `f(3) <~ 0.0157`, `f(4) <~ 0.0115`. The
  **between-weight gaps shrink fast** (`0.015` then `0.004`), which is the signature of a **positive** limit
  `f_infty ~ 0.006 - 0.010`, not decay to `0`.

**Consequence.** Whether `f_infty = 0` or `f_infty > 0` is genuinely open and is THE decisive question:
  - If `f_infty > 0`: `max_7 notin closure(V_2)`, so `max_7` is **not** two-hidden-layer (the separation HOLDS),
    and a **continuous** separating functional exists -- the no-go does NOT apply, and the bound is provable by a
    finite certificate (the limiting dual of the column generation). This would be the first real-weight depth
    separation for `max_n`.
  - If `f_infty = 0`: density holds, the no-go applies, and only a singular certificate can work.

So the no-go does not close the finite-certificate route after all -- it only closes it *under* the density
premise, which the clean numbers call into question. The honest state: the trajectory leans (weakly) toward
`f_infty > 0` (separation provable), but the true floors are computationally walled (a full-rank ~18866-block
least-squares per weight), so a **reliable `f_infty` estimate is the single most valuable next computation** --
a GPU full-family residual (or stabilized LSMR) at weights 3,4,5,6, watching whether the floor stabilizes
(provable separation) or collapses toward 0 (singular certificate needed).

## 10. RESOLVED: the floor collapses -- density holds (`f_infty = 0`), so the certificate IS singular

The decisive computation ran (`floor_w3_gpu.py`, GPU CGLS against the COMPLETE weight-3 family, all 19219
orbit columns, with a convergence-checked orthogonality gauge):

  `f(2) = 0.0308` (exact)  ->  `f(3) ~= 0.0022` (still descending; true floor lower, but `> 0` by gpu_w3 mod-p).

A ~14x collapse in one weight step. The Sec. 9 hope (`f_infty > 0`, finite certificate) was an artifact of the
**capped** column generation: greedy colgen with 3000 of 19219 blocks plateaued at `0.0157`, but the FULL
weight-3 family reaches `0.0022`. The true floors collapse fast, so **`f_infty = 0` -- density holds, max_7 is
in the closure of `V_2`.** The earlier "0.0017 density" number was right in spirit (small floor) even if its
exact value was unreliable; my Sec. 9 doubt was wrong, and the full solve corrects it.

**What this does and does NOT mean.**
- It does NOT disprove the separation. `f(w) > 0` at every finite weight (proven exactly for `w = 2, 3` via
  gpu_w3) means `max_7 notin span(weight-w joins)` for each `w`. Since any genuine 2-layer net has SOME finite
  weight, `max_7 in V_2` iff `f(w) = 0` for some finite `w`. The data shows `f(w) > 0` but `-> 0`: exactly the
  picture "`max_7 in closure(V_2) \ V_2`", i.e. **the separation can still hold**, with the distance shrinking
  to zero. The strictly-positive finite-weight floors are genuine evidence FOR the separation.
- It DOES kill the finite/continuous certificate (Sec. 8 no-go applies: density => any continuous functional
  vanishing on the joins kills max_7 too). So the separation, if true, requires a **singular** annihilating
  functional, weight-uniform / scale-invariant -- one proving `f(w) > 0` for ALL `w` at once, not weight by
  weight.

**Net.** The honest target is now unambiguous: a single weight-uniform obstruction `lambda` with `lambda(h_Q)=0`
for every join `Q` (all weights) and `lambda(max_7) != 0`, necessarily singular (not a finite measure on
points). This is precisely `SCALE_INVARIANT_ROUTE.md`: an odd/scale-invariant functional whose annihilation of
all joins is structural (central symmetry of zonotope pieces, the `U_{1,n}` vs graphic-matroid distinction),
not weight-graded. The whole investigation has converged to that one object.
