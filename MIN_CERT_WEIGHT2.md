# Minimal-support weight-2 dual certificate (route C, bounded experiment) -- DECISIVE

Goal (per the strategy fork): before scaling the exact dual certificate to weight-3, determine at weight-2
whether the certificate has ANY clean/sparse/structured representative, or whether it is inherently a
high-dimensional basis artifact. Decision rule: small+structured => possible named invariant, keep the
certificate route; large+diffuse => stop, pivot to route A (decomposition polyhedron) or B (construction).

## Setup
- Complete weight-2 P2 orbit family: 121 blocks. Design M = [121 block columns | 7 linear columns] = 128
  columns, evaluated at m random integer points in [-16,16]^7. Target = max_7 values.
- Certificate lambda over points: lambda.h_Q = 0 (all 121), lambda.x_i = 0 (all 7), lambda.max_7 = 1.

## The generic lower bound (proved)
A support-s certificate = s points whose block+linear rows are linearly DEPENDENT (that is the
annihilation) while max_7 stays independent. For points in general position, rank(M restricted to k rows)
= min(k, rank(M)), so max_7 is trivially in the column span until k > rank(M). Hence
```
min-support = rank(M) + 1   (girth of the row matroid),   rank(M) = functional dim of {weight-2 P2 span} + linear.
```
Structured (near-wall / symmetric) points could in principle create a smaller circuit; the experiment tests
whether they actually do.

## Results (exact, mod-p rank + greedy removal + exact rational reconstruction)
- rank(M) = 125 (of 128 columns) => generic min-support = 126.
- Greedy removal from a 350-point working certificate, 4 independent random orders: EVERY pass bottomed out
  at exactly 126 points. No structured/near-wall shortcut below rank+1 exists.
- Exact rational certificate on the 126-point minimal support: all 126 coefficients nonzero;
  integer-normalized (cleared denominators, gcd 1) max |coefficient| has ~200 DIGITS.
- Geometry of the minimal support: 126 points touching 123 DISTINCT braid chambers (about one witness per
  chamber), with min coordinate-gap 0 (some support points lie ON braid walls). Maximally diffuse, and
  cross-chamber -- a direct realization of the single-chamber invisibility lemma (GLOBAL_GLUE_OBSTRUCTION.md).

## Interpretation (decisive)
| representative                    | support | max |coeff| |
|-----------------------------------|---------|-------------|
| raw nullspace particular solution | dense   | ~200 digits |
| min-support (irreducible)         | 126     | ~200 digits |
| min-denominator                   | (bounded below by the same rank+1 support; coefficients stay ~200 digits) |

The certificate is INHERENTLY large (support = rank+1 = 126, forced), diffuse (one witness per braid
chamber), and arithmetically opaque (~200-digit integer coefficients even at minimal support). It is a
basis artifact of a high-dimensional nullspace, NOT a named/local invariant. This is fully consistent with,
and explained by, the cross-chamber (global-gluing) nature of the obstruction: there is no sparse local
object to find.

## Decision
STOP the raw-dual-certificate route; do not scale this object to weight-3 (~19k columns would be strictly
worse). The certificate exists and proves OUT exactly, but it is not human-theorem material. Pivot to:
- route A: the P2-restricted decomposition polyhedron (Brandenburg-Grillo-Hertrich) -- the correct global
  object, and the frame the single-chamber lemma says is necessary; PRIMARY theory route.
- route B: a disciplined construction / IN search (propose finite support by approximation, then solve
  exactly, then chamber-verify) -- highest upside (max_7 IN would be the bigger prize); run opportunistically.

The durable keep from route C is the single-chamber invisibility lemma + this quantitative confirmation
(min-support = rank+1, ~200-digit coeffs, 123 chambers) that the certificate is global/diffuse by nature.
