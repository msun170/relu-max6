# max6 in two hidden layers

An exact, computer-assisted proof that `max(x1, ..., x6)` is computable by a ReLU neural network with
**two hidden layers** (arbitrary width, rational weights). This resolves an open problem of Bakaev,
Brunck, Hertrich, Stade, and Yehudayoff (*Better Neural Network Expressivity: Subdividing the Simplex*,
STOC 2026), who proved the same for `max5` and explicitly left `max6` open.

## The result

There is an explicit signed combination of **6 S6-orbits of P2 support functions** (denominator 360)
equal to `max6`. A P2 support function is the support function of a join of two zonotopes, which a
2-hidden-layer ReLU network computes exactly; a signed sum of them is again 2 hidden layers. So the
construction *is* a 2-hidden-layer network, and we prove it equals `max6` everywhere, with no floating
point in the certificate.

The proof reduces, by S6-symmetry and a gradient argument, to checking that a construction-dependent
gradient equals `e1` on every cell of a 35-hyperplane arrangement in R^5. That arrangement has exactly
**2608 cells**; we enumerate all of them deterministically (each with an exact integer witness), verify
the gradient exactly, and prove the enumeration complete by an exact rational adjacency-closure argument.

## Layout

| file | what it is |
|------|------------|
| `core.py` | shared engine: weight-2 lattice, zonotopes, S_n-orbit canonicalization, support functions, exact rational solve (parameterized by n) |
| `construction.py` | loads `results/construction_max6.txt` and expands it to its individual P2 terms |
| `prove_max6.py` | the exact theorem: cell enumeration + exact gradient check + adjacency-closure completeness |
| `verify.py` | a fully independent exact CPWL verifier (separate code path, for cross-checking) |
| `search.py` | general-n search: is `maxn` a signed sum of weight-2 P2 support functions? |
| `minimize.py` | minimize the construction (fewest orbit terms, smallest denominator) |
| `lower_bound.py` | complete weight-2 enumeration: proves `max7`/`max8` have no weight-2 two-layer representation |
| `results/` | the construction and the proof certificate |

## Reproduce

Requires Python 3 with `numpy` and `scipy`.

```
python prove_max6.py     # regenerates results/proof_certificate.txt   (~10 min)
python minimize.py       # regenerates results/construction_max6.txt   (minimal 6-orbit form)
python search.py 6       # find a 2-layer construction for max6 and verify it
python search.py 7 R4 C16 J3   # max7: reports inconsistent even in a rich class
python verify.py         # independent verifier self-checks
```

## Status and scope

- `max6`: **proven** in two hidden layers (exact, this repo).
- `max7`, `max8`: **proven** to have no weight-2 two-hidden-layer representation (`lower_bound.py`, exact
  over Q, complete enumeration of the finite weight-2 building-block family with all generators). This is
  a conditional lower bound; the full two-vs-three layer separation needs only the normal-form conjecture
  that a two-layer `maxn` may be assumed weight-2.
- Note: only TWO-way joins are valid two-layer building blocks (a three-way join is three layers). Earlier
  exploratory searches that included higher joins were testing a larger, non-two-layer space.
- The construction uses rational (non-integer) weights, which is necessary: integer-weight ReLU nets
  provably need `ceil(log2 n)` layers for `maxn` (Haase, Hertrich, Loho, ICLR 2023).
