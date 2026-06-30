# A research program on the exact depth of max_n (and where max_6 fits)

Framing for a possible collaboration (Hertrich group). Separates what is PROVEN, what is CONJECTURED with
strong support, and what is OPEN.

## The object

`max_n(x) = max(x_1,...,x_n)`, exact representation by ReLU networks. `k` hidden layers compute exactly the
CPWL functions of "depth k". In the polyhedral language: `max_n = h_{Delta_{n-1}}` (support function of the
simplex), and a 2-hidden-layer function is a signed sum of support functions of joins of two zonotopes.

## Proven

- **n <= 6: `max_n` is 2-hidden-layer.** n=4,5 known (BBHSY, STOC 2026, via polyhedral subdivision of the
  simplex). **n=6 is ours** (WRITEUP.md): an explicit 6-orbit weight-2 signed representation, verified two ways
  -- exact-rational (0 error over 400 fresh rational points) and a rigorous arrangement-cell proof (0 of 2608
  cells have a wrong gradient, check_max6.py). BBHSY explicitly left n=6 OPEN, so this resolves their question.
- **`max_7` is OUT of the weight-2 and the complete weight-3 lattice models** (exact mod-p, two primes,
  gpu_w3.py). So any 2-layer representation of `max_7` must have edge complexity >= 4.
- **Lower bounds (prior work):** integer weights need `ceil(log2(n+1))` (Haase-Hertrich-Loho, via 2-adic
  normalized-volume parity); decimal-fraction weights need `ceil(log3 n)` (Averkov-Hojny-Merkert). Both are
  base-`b` volume arguments and force 3 layers only at `n >= 9-10`.

## Conjecture (strong support): `max_n` is 2-hidden-layer iff `n <= 9`

- The only scale-invariant lower bound (decimal, `ceil(log3 n)`) forces 3 layers first at **n=10**, and is
  matched there by the upper bound -- so `max_10` needs exactly 3 (decimal). For `n <= 9` the lower bound
  permits 2. **No technique gives a real/rational lower bound > 2 at n=7,8,9.**
- Constructions exist for n <= 6 (now including our n=6). If the pattern continues, `max_7, max_8, max_9` are
  2-layer and the threshold is exactly **n=9 / n=10**, making the decimal bound tight across the board.
- This would also explain why every depth-2 lower-bound obstruction we tried (mixed-volume/Hodge-Riemann,
  finite signed measures, odd-part/central-symmetry, homological) provably collapses: there may be nothing to
  obstruct below n=10.

## Open (the frontier)

**Is `max_7` 2-hidden-layer?** Resolving it either way is a real result:
- **IN** (construct it): needs an edge-complexity-4 construction -- a polyhedral subdivision of `Delta_6` into
  join pieces with join intersections (the valuation method that built max_5/max_6), now at higher weight. We
  could not derive it by hand, and the complete weight-4 family is computationally out of reach (its join span
  is near-full-dimensional, rank > 23000 at m=24000 and still growing; membership is not testable non-vacuously
  with available tooling).
- **OUT** (prove >= 3 layers over the reals): would be the FIRST nontrivial real-weight depth lower bound for any
  `max_n`. Beyond all known techniques (the base-`b` volume arguments bottom out at n~10).

## Honest assessment

The direct evidence is genuinely split. `max_4,5,6` are all clean LOW-complexity (weight-2); `max_7` is OUT of
weights 2 and 3 and of every structured weight-4 subfamily we could test, and its best 2-layer approximation
floor is small but positive (~0.0006, decreasing with weight) -- consistent BOTH with "in the closure, not in the
set" (OUT/3-layer) and with "IN at high weight-4 we cannot reach". Net `~50/50`.

## Concrete deliverables and next steps

1. **max_6 (done):** a clean, verified, novel theorem resolving BBHSY's open n=6 case. Publishable as-is.
2. **The n<=9 program:** construct max_7, max_8, max_9 (IN) -> with the n=10 decimal lower bound this gives the
   exact threshold "2-layer iff n<=9". The construction is the bottleneck; the valuation/subdivision method is
   the right tool and is exactly the kind of thing the BBHSY techniques could push.
3. **OR the separation:** if max_7 is genuinely 3-layer, a new scale-invariant lower-bound technique is needed --
   a high-risk, high-reward open problem.
