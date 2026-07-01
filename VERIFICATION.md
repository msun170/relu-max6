# Adversarial verification of the max_6 paper

Each numbered claim in `paper/max6.tex` was checked by an independent adversarial agent with a clean
context, instructed to break the proof (find gaps, unjustified steps, hidden assumptions, or
counterexamples). The computational claims were audited against the actual code, with independent
recomputation. Summary of the round and the fixes applied.

| # | Claim | Verdict (pre-fix) | Action |
|---|-------|-------------------|--------|
| Lemma 1 | Two-layer dictionary (2-layer = signed sum of block support functions) | CORRECT | none |
| Lemma 2 | Symmetrization (WLOG S_n-symmetric representation) | MINOR (exposition) | fixed |
| Prop 1 | Necessity of cancellation (n>=5 needs a negative coefficient) | MINOR (one sub-proof wrong) | fixed |
| Lemma 3 | Gradient reduction | MINOR (two under-justified steps) | fixed |
| Lemma 4 | Forced directions (85 forced / 35 unforced) | CORRECT (counts independently reproduced) | none |
| Lemma 5 | Completeness by adjacency closure | CORRECT (Gordan impl. audited) | none |
| Theorem | max_6 in two hidden layers | MINOR (packaging only) | fixed |
| Prop 2 | max_7 OUT of complete weight-2 family | CORRECT (family completeness + certificate reverified) | none |
| Lemma 6 | Single-chamber invisibility | MINOR (wording) | fixed |

No claim was found false. Every result stands; all issues were expository, one sub-proof, or packaging.

## Substantive fixes

- **Prop 1 (necessity of cancellation), Step 3.** The original argument ("if conv(Z_1 u Z_2) is a simplex
  then Z_1,Z_2 are faces in complementary skew position, so dimension <= 3") is FALSE: a triangle is
  conv(interior chord u opposite edge), where the chord is a segment (zonotope) but not a face. Replaced by
  a correct central-symmetry argument: a zonotope Z inside a simplex S carries at most two of S's vertices
  (if vertex v=e_j of S is a vertex of Z with center c, then 2c-v in Z subset S forces barycentric
  coordinate c_j >= 1/2, and three such would sum to > 1). Hence the n vertices of S = conv(Z_1 u Z_2) split
  as <= 2+2, so n <= 4. The proposition and the n>=5 threshold were always correct; only the sub-proof was
  replaced.

- **Prop 1 / Lemma 2, the affine constant.** Both proofs now note explicitly that the constant term b of
  the affine part vanishes (evaluate the identity at x=0; support functions vanish at the origin).

- **Lemma 3 (gradient reduction).** The step "zero gradient on the cells => constant on the chamber" is now
  justified via: affine-on-each-cell + continuity across shared walls + connectedness of the cell-adjacency
  graph; and "the constant is 0" is now derived from degree-1 positive homogeneity (kappa = lambda*kappa
  for all lambda>0) rather than a loose "evaluate at x=0".

- **Lemma 6 (single-chamber invisibility).** Wording tightened: the conclusion is "the certificate cannot
  have all evaluation points in a single open chamber" (the exact contrapositive), and the proof is a
  single global contradiction, not a per-chamber decomposition.

- **Theorem (packaging).** `construction.py` (a dependency of `check_max6.py`) had been dropped in the
  repo split and is restored. The paper's "independent CPWL verifier" phrase is replaced by the accurate
  statement that an independent exact-rational evaluation of C and max_6 at random points (including
  tie/boundary cases) confirms equality with zero error.

## What the code audits confirmed (independently reproduced)

- The construction's six orbit blocks are genuine two-way joins conv(Z_1 u Z_2) of real zonotopes (not
  3-way joins); expanding over S_6 gives 1530 P_2 terms.
- The chamber proof is float-free where it matters: integer witnesses, int64 gradient equality, and
  exact-Fraction Gordan emptiness certificates. Floating point appears only in proposal/pruning steps, each
  followed by exact re-verification. Reproduced tallies: 35 unforced hyperplanes, 2608 cells, 0 bad
  gradients, 13756 in-set + 77524 Gordan-empty flips, 0 gaps. Minimum argmax margin over all
  1530 x 2608 evaluations is 1 (no ties).
- The Gordan emptiness system correctly includes the cone constraints y_k > 0 among its generators
  (omitting them would be a soundness bug; they are present), and the error orientation is fail-safe.
- max_7 OUT of the complete weight-2 lattice family is exact and non-vacuous: the family enumeration is
  complete (brute-forced against all lattice zonotopes; the only extra sets are support-redundant
  midpoints), the dual certificate lambda satisfies lambda.block=0 for every block, lambda.linear=0,
  lambda.max_7=1 over exact Q, and the same pipeline gives max_5, max_6 IN as controls.
