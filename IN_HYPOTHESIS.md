# Working hypothesis (2026-06-30): max_7, max_8, max_9 ARE 2-layer; n=10 is the first 3-layer case

A literature-informed reassessment that flips the earlier OUT-leaning intuition.

## The argument that max_7 is IN

**All known depth lower-bound techniques bottom out around n = 9-10.**
- Haase-Hertrich-Loho (integer weights): `ceil(log2(n+1))` via normalized-volume PARITY (2-adic, base 2).
  Scale-sensitive => integer-only; rational weights escape by clearing denominators.
- Averkov-Hojny-Merkert (decimal weights): `ceil(log3 n)` -- a base-3 volume argument. Forces 3 layers only
  at `n >= 10` (`ceil(log3 9)=2`, `ceil(log3 10)=3`).
- A volume/lattice argument with base `b` forces `>=3` layers only at `n >= b^2`. To force 3 at `n=7` needs
  `b < 7^{1/2} ~ 2.65` for the relevant counting... and the genuinely scale-INVARIANT versions need `b>=3`
  (decimal), giving the threshold at `n=10`. **No known technique gives a real/rational lower bound > 2 at n=7.**

**Constructions exist for n=4,5,6** (max_5: Bakaev et al STOC 2026; max_6: ours, novel). The pattern of "2-layer
keeps working" plus "no lower bound technique reaches n=7" points to: max_7,8,9 are 2-layer, threshold at n=10
(matching the decimal `ceil(log3 n)` bound, which is then TIGHT).

This also explains cleanly why every OUT obstruction we built collapsed (mixed-volume/Hodge, finite measure,
odd-part, homological): **there may be no obstruction because max_7 is 2-layer.** The floor collapse
(f(2)=0.032 -> f(3)~0.0006 -> f(4) lower) is the approach-to-zero of a function that IS in the closure AND
(hypothesis) in the set.

## What's needed to confirm

- `max_7` is OUT of weight-2 and complete weight-3 (proven). So a 2-layer rep needs edge complexity >= 4.
- Find an explicit WEIGHT-4 representation (structured family mod-p membership test, weight4_modp.py; expand
  generator complexity until IN, then verify exactly with Fractions). A positive result = novel theorem
  "max_7 is 2-layer", extending the frontier and supporting the n=10 threshold.
- Then max_8, max_9 similarly; and n=10 OUT is already the AHM decimal bound. Complete story: 2-layer iff n<=9.

## Caveat (honest)

This is a hypothesis from the SHAPE of known results, not a proof. It is possible max_7 is genuinely 3-layer and
the lower-bound techniques are just not yet sharp enough -- in which case proving it needs a fundamentally new
(scale-invariant) technique, which is the hard open problem. The construction search is the decisive test: find
a weight-4 rep => IN settled; exhaust reasonable structured families without finding one => OUT-leaning but
inconclusive (the complete weight-4 family is too large to enumerate).
