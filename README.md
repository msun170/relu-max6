# max6 in two hidden layers

Two results about computing the maximum with shallow ReLU networks, kept as separate layers:

- **Theorem 1.** `max(x1,...,x6)` is computable by a ReLU network with two hidden layers (exact proof).
- **Theorem 2.** `max7` (and `max8`) has no two-hidden-layer representation whose building blocks have
  weight-2 vertices (exact, with a checkable dual certificate). This is a finite-family lower bound, not
  the full separation.
- **Open conjecture.** Every two-hidden-layer `max_n` can be normalized into the weight-2 model. If true,
  Theorem 2 gives the first two-versus-three hidden-layer separation over the reals.

Theorem 1 resolves an open problem of Bakaev, Brunck, Hertrich, Stade, Yehudayoff (STOC 2026), who proved
the `max5` case and left `max6` open. See `WRITEUP.md` for the readable account.

## Check the theorems

Requires Python 3 with `numpy` and `scipy`.

```
python check_max6.py                      # Theorem 1: exact proof (the adjacency-closure step takes ~10 min)
python check_weight2_max7_infeasible.py   # Theorem 2: verifies the exact dual certificate (~20 s)
```

Expected output of the first: `VERDICT: max6 IS computable in 2 hidden layers ...`.
Expected output of the second: `RESULT: PASS -- max7 is not in the complete weight-2 two-layer span (exact).`

## Files

| file | role |
|------|------|
| `core.py` | shared engine: weight-2 lattice, zonotopes, S_n-orbit canonicalization, support functions, exact rational solve |
| `construction.py` | the six-orbit max6 construction, expanded to its building blocks |
| `check_max6.py` | Theorem 1: cell enumeration, exact gradient, exact adjacency closure |
| `verify.py` | an independent exact verifier (separate code path) |
| `build_max7_certificate.py` | builds the exact dual certificate for Theorem 2 |
| `check_weight2_max7_infeasible.py` | Theorem 2: regenerates the family and verifies the certificate |
| `lower_bound.py` | general-n weight-2 enumeration and exact membership test |
| `search.py` | general-n search for a weight-2 construction |
| `minimize.py` | minimize the construction (fewest orbit terms) |
| `results/` | construction, proof certificate, max7 dual certificate |
| `WRITEUP.md` | the readable account; `SEPARATION_PROGRAM.md` the research program for the conjecture |

## Notes

The construction uses rational weights, which is necessary: integer-weight ReLU networks provably need
`ceil(log2 n)` hidden layers for `max_n` (Haase, Hertrich, Loho). Only two-way joins are valid
two-hidden-layer building blocks; a three-way join is three layers.
