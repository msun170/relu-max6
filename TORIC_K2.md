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
