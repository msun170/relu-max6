# The non-linear obstruction: what must survive signed cancellation

Every LINEAR/continuous obstruction collapses with weight (iters 25-28: mixed-volume/Hodge multilinear =>
span test; finite measure => density; odd-part/central-symmetry => collapses 0.119->0.0049). The reason is
uniform: all are continuous functionals, and `max_7 in closure(V_2)` (the floor f(w) -> 0). A real obstruction
must be **non-linear** and **insensitive to signed cancellation**. This note records the one candidate class
with that property and why it is the right shape.

## The cancellation-free reformulation

`max_n` is 2-layer  iff  `h_{Delta_n} = sum_t c_t h_{Q_t}` (signed, `Q_t` joins of zonotopes). Split by sign:

  `Delta_n + N = M`,   `N = sum_{c_t<0} |c_t| Q_t`,  `M = sum_{c_t>0} c_t Q_t`   (Minkowski sums; both EFFECTIVE).

Now there is **no cancellation**: `M, N` are genuine polytopes, Minkowski sums of joins-of-zonotopes. The
question becomes a **Minkowski-summand** question: can the simplex be a Minkowski summand of a sum-of-joins,
with the complementary summand also a sum-of-joins? Any invariant that is *monotone or structured under
Minkowski sum* (not multilinear) avoids the collapse, because the multilinear collapse used the signed/linear
structure that this reformulation removes.

## Why multilinear invariants still fail here (and what doesn't)

Mixed volumes, surface area measures, all intersection numbers: multilinear in the summands => `V(Delta_n + N)`
expands into the SAME data as `V(M)` once the linear relation holds. They cannot see the summand structure.
**Non-multilinear** invariants of the face lattice CAN:

- **2-face combinatorial type.** A 2-face of a Minkowski sum in a fixed normal direction is the Minkowski sum
  of the summands' faces there. Zonotope 2-faces are **centrally symmetric** (2k-gons). Simplex 2-faces are
  **triangles** (the unique non-centrally-symmetric minimal polygon). A join makes a triangle only at a
  **bridge** (vertex-of-`Z_1` + edge-of-`Z_2`). So in `Delta_n + N = M`: every direction where `Delta_n`
  contributes a triangle 2-face forces `M` (a sum of joins) to carry a triangle there -- which the joins can
  only supply from bridges. The simplex has `C(n,3)` triangular 2-faces; they must ALL be realized by bridge
  triangles of `M`, simultaneously and consistently.

- **The bridge cancellation as a chain condition.** From `STRUCTURE_MAX6.md`: the bridges are the only
  asymmetric content, and in the working `max_6` construction they **cancel at codimension 1** (NB) while the
  root parts build the simplex (J). At `n=7` (J)+(NB) are individually satisfiable but **jointly
  over-determined** (transition.py: defect 1 at weight-2; f(3)=0.0022>0 so still infeasible at weight-3). The
  bridges form a chain that must (i) generate the simplex's `C(n,3)` triangle 2-faces and (ii) self-annihilate
  at codim 1. This is a **cycle/boundary** condition -- the signature of a homological obstruction.

## The candidate obstruction (homology of the matroid, not its intersection ring)

AHK gives the permutohedral / matroid Chow ring two kinds of structure: the **multilinear intersection form**
(which we showed collapses) and the **homology / module structure** (cycles mod boundaries). Signed cancellation
is exactly "mod boundaries". So the right object is a **homology class**, not an intersection number:

> Is there a class in `H_*` of the (Bergman fan of the) matroid `U_{1,n}` -- carried by the simplex's triangle
> 2-faces -- that cannot be represented by chains supported on centrally-symmetric (graphic-matroid / zonotopal)
> pieces, for `n >= 7`?

This is non-linear (homology, not a functional), cancellation-proof (boundaries are quotiented out, so the
signed bridge-cancellation is built in), and weight-uniform (a matroid-intrinsic class, independent of zonotope
size = weight). It is the homological shadow of the (J)+(NB) over-determination. `U_{1,n}` (simplex) vs graphic
matroids (zonotopes) is precisely the distinction AHK homology can see.

## Status and the concrete test

Unproven -- this identifies the correct *type* of object (a relative homology / cycle obstruction on the matroid
fan), not the obstruction itself. The concrete, decidable proxy is **persistence of the (J)+(NB) defect with
weight**: defect 1 at weight-2, infeasible at weight-3 (f(3)=0.0022>0). If the IN search certifies `max_7` OUT
of weight-4 as well (defect persists), the over-determination is weight-stable and the homological obstruction
is the right target. If the IN search finds `max_7 IN` at weight-4 (residual -> 0), there is no obstruction --
`max_7` is 2-layer over the reals and the separation fails at `n=7`. The two branches are decided by the same
experiment.
