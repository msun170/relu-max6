# A self-similar measure-annihilator hierarchy for ReLU depth

This note records a clean structural fact about the depth hierarchy, derived while pursuing the
scale-invariant certificate of `SCALE_INVARIANT_ROUTE.md`. It recasts "is `f` computable in `k` hidden
layers" as an iterated version of the one-layer (cosine-transform) condition, and explains structurally why
two layers are so much harder to rule out than one.

All functions are homogeneous degree-1 CPWL (support functions). Let `V_0` = linear functions,
`V_1 = span{(v.x)_+}` = one-hidden-layer functions, and in general `V_k` = `k`-hidden-layer functions (a
linear space: signed sums of `k`-layer functions are `k`-layer). For a signed measure `mu`, write
`mu \perp V_k` if `\int f dmu = 0` for all `f in V_k`. A measure with `mu \perp V_{k}` and `mu(max_n) != 0`
is a **certificate that `max_n` needs more than `k` layers** (assuming `V_k` is closed; see the caveat).

## The ReLU-lift lemma

`V_k = V_{k-1} + span{ReLU(g) : g in V_{k-1}}` (a `k`-layer net applies one ReLU layer to `(k-1)`-layer
functions, plus pass-through). Hence

> `mu \perp V_k`  iff  `\int ReLU(g) dmu = 0` for every `g in V_{k-1}`.

(The forward direction is clear; conversely `g = ReLU(g) - ReLU(-g)` shows the ReLU condition already forces
`mu \perp V_{k-1}`.)

## The self-similar reduction (the point)

Fix support points `p_1,...,p_m`. Map each point to its **`V_{k-1}`-feature vector** `b_i in R^d`: the values
at `p_i` of a basis of `V_{k-1}` (so every `g in V_{k-1}` has `g(p_i) = b_i . alpha` for its coordinate
vector `alpha in R^d`, `d = dim V_{k-1}` at the points). Then

  `\int ReLU(g) dmu = sum_i mu_i (b_i . alpha)_+`,

so `mu \perp V_k` becomes `sum_i mu_i (b_i . alpha)_+ = 0` for all `alpha` -- which is **exactly the
one-hidden-layer (k=1) condition, but in the feature space** `R^d`. By the k=1 lemma there, this holds iff the
lifted measure `nu = sum_i mu_i delta_{b_i}` is **odd and balanced** (`sum_i mu_i b_i = 0` and the cosine
transform of `nu` vanishes; equivalently, per realizable positive-set `S` of the features,
`sum_{i in S} mu_i b_i = 0`).

> **Hierarchy.** `mu \perp V_k`  iff  the lift of `mu` to `V_{k-1}`-feature space is odd-balanced. The depth
> hierarchy is self-similar: depth-`k` annihilation is depth-1 (odd-balanced) annihilation in the
> depth-`(k-1)` feature space.

For `k = 1` the feature map is the identity (`V_0` = linear), recovering the clean result of
`SCALE_INVARIANT_ROUTE.md`: `mu \perp V_1` iff `mu` itself is odd-balanced, which is easy to satisfy with
`+/-` point pairs, so one-layer lower bounds are easy (`max_n` is not 1-layer for `n >= 3`).

## Why two layers are hard, made precise

For `k = 2` the feature map `p -> b_p = ((v.p)_+ values)` is **nonlinear**. So `-b_p` is generically **not**
the feature of any support point. The odd-balanced condition (cosine transform of `nu` vanishes => `nu` has
zero even part) wants the feature cloud to have `+/-` symmetry, which the nonlinear lift destroys. Odd-balanced
lifted measures are therefore rare, certificates are rare, and `max_n` can stay two-layer for a long range --
precisely the phenomenon we see (`max_n` is 2-layer through `n = 6`, and `max_7` is OUT only of the lattice
models we can compute). The difficulty is not incidental; it is the nonlinearity of the feature lift.

## Validation

`feature_cert.py` sets up the exact feature-lift conditions and asks whether a certificate exists. For
`n = 3` (where `max_3` IS two-layer): 24 support points, `dim V_1 = 12`, 7585 feature-chambers enumerated,
and the system `{sum_{i in S} mu_i b_i = 0 for all S} + {mu(max_3) = 1}` is **inconsistent -> no
certificate**, exactly as it must be. The reduction correctly certifies two-layer-ness.

## Honest status

This is a structural reframing, not a proof of the separation. Two caveats: (i) it does not by itself produce
a certificate for `max_7` -- the `V_1`-feature space and its chamber count grow fast, and the genuinely-open
question is whether an odd-balanced lifted measure with `mu(max_n) != 0` exists for `n >= 7`; (ii) soundness
of the certificate route needs `V_k` to be closed (whether the exact `k`-hidden-layer class is topologically
closed is itself subtle). What the hierarchy buys is the precise object to study -- an odd-balanced measure in
a nonlinear feature space -- and a clean explanation of why depth-2 resists the tools that settle depth-1.
