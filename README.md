# max_6 in two hidden layers

**Theorem.** `max(x_1,...,x_6)` is exactly computable by a ReLU network with two hidden layers.

This resolves the `n = 6` case left open by Bakaev, Brunck, Hertrich, Stade, and Yehudayoff (*Better Neural
Network Expressivity: Subdividing the Simplex*, STOC 2026), who proved the `max_5` case. The construction is
a signed combination of six `S_6`-orbits of join-of-two-zonotope support functions (denominator 360, zero
linear part), and correctness is established by an exact, floating-point-free computer-assisted chamber
proof.

The paper is in [`paper/max6.tex`](paper/max6.tex). It also contains:
- a **necessity-of-cancellation** theorem (for `n >= 5` every two-layer representation needs a negative
  coefficient: the simplex appears only after cancellation in a signed Minkowski identity), and
- a **single-chamber invisibility lemma** locating the obstruction to a `max_7` separation as a global,
  cross-chamber phenomenon, with the reduction of a genuine two-vs-three-layer separation to a lattice
  normal-form conjecture.

The `max_7` frontier investigation (OUT results, construction searches, obstruction probes) lives in the
separate `relu-max7` repository.

## Reproduce

Requires Python 3 with `numpy` and `scipy`.

```
python verify_2layer.py        # exact rational construction + fresh-point check for max_4, max_5, max_6
python check_max6.py           # the exact floating-point-free chamber proof of the max_6 theorem
python min_virtual_decomp.py   # the minimal virtual Minkowski decomposition (dyadic 4:2:1 for max_5, max_6)
```

`verify_2layer.py` prints, for each n, `EXACT MATCH => max_n IS 2 hidden ReLU layers`. `check_max6.py`
produces the chamber certificate (the adjacency-closure step is the slow part).

## Files

| file | role |
|------|------|
| `core.py` | shared engine: weight-2 lattice, zonotopes, `S_n`-orbit canonicalization, support functions, exact rational solve |
| `verify_2layer.py` | exact rational construction of `max_4,5,6` as signed sums of `P_2` support functions, verified at fresh rational points |
| `check_max6.py` | the exact, floating-point-free chamber proof of the `max_6` theorem |
| `min_virtual_decomp.py` | minimal virtual Minkowski decomposition (decomposition-polyhedron min-mass vertex); dyadic 4:2:1 for `max_5,6` |
| `paper/max6.tex` | the paper |
| `results/construction_max6.txt` | the six orbit representatives (gradient/vertex sets) and coefficients |
| `results/proof_certificate.txt` | the chamber-proof certificate |
| `results/p2_decompositions_max6.json` | each orbit's explicit `conv(Z1 u Z2)` decomposition |

## Notes

The construction uses rational weights, which is necessary: integer-weight ReLU networks provably need
`ceil(log2 n)` hidden layers for `max_n` (Haase, Hertrich, Loho, ICLR 2023). Only two-way joins of zonotopes
are valid two-hidden-layer building blocks; a three-way join is three layers.
