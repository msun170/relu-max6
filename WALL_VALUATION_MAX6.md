# Valuation-move factorization of the max_5 / max_6 virtual identity -- NEGATIVE result (honest)

Goal (descent prototype): does the non-braid bridge-wall cancellation in the known max_5, max_6 reps factor into
BOUNDED valuation moves (pairwise / small-group cancelling subsets)? If yes -> prototype for a descent theorem
bounding bridge complexity -> bounded weight.

## Method (valuation_factorization_max6.py)
NB[row, block] = exact signed second-difference (wall jump) of each support block's orbit-sum on a non-braid wall
probe; rows grouped by S_n symmetry of the wall direction. The construction satisfies NB . c = 0 (verified exactly,
all rows). Two blocks are LINKED only if they cancel the SAME jump ROW (not merely share a normal). FACTORIZATION =
minimal subsets of blocks whose COEFFICIENT-WEIGHTED jump-columns sum to ZERO over ALL rows (= valuation-move
components). Soundness: a subset nonzero over the probes is nonzero function-level, so "no factorization" is robust
to adding rows (more rows cannot create a factorization).

## RESULT (negative)
    max_5: 5 blocks (2 pos, 3 neg). all rows cancel exactly. GLOBAL minimal zero-sum component = ALL 5 blocks.
           No proper sub-collection cancels. max component size = 5.
    max_6: 6 blocks (3 pos, 3 neg). all rows cancel exactly. GLOBAL minimal zero-sum component = ALL 6 blocks.
           No proper sub-collection cancels. max component size = 6. Per-row: only 2 pairwise rows; 14 rows couple
           up to all 6 blocks.
So the bridge-wall cancellation is GLOBALLY COUPLED / IRREDUCIBLE -- it does NOT decompose into bounded valuation
moves. The earlier "pairwise +1/90 vs -1/90" observation was a SINGLE-ROW artifact, not a global factorization.

## Honest conclusion
- The descent-via-BOUNDED-LOCAL-valuation-moves hypothesis is NOT supported by the known max_5, max_6 constructions.
  Their cancellation is one irreducible global circuit (the single dependency sum_t V_t = 0 has FULL support).
- This does NOT disprove the bounded-weight normal form; it RULES OUT the simplest proof route for it (local move
  descent). A proof, if one exists, must handle GLOBAL cancellation, not reduce it to bounded local moves.
- Caveat (as warned): we only analyzed the KNOWN constructions. "Globally coupled" is a statement about THESE reps,
  not a proof that all reps are. But it removes the optimistic prototype.

## Where this leaves the program
The forced-negativity theorem and the virtual-Minkowski framing stand. The clean facts:
  (1) for n>=5 every rep is signed (THEOREM); (2) the cancellation is globally coupled (this result). Together they
  say: max_n's representation is an IRREDUCIBLE virtual Minkowski identity Delta + N = M with global cancellation --
  no bounded-move structure to exploit. So the bounded-weight bound (if true) needs a genuinely global argument; the
  local-descent route is closed. This is a real (negative) narrowing of the search, consistent with the problem being
  genuinely hard. Outputs: valuation_factorization_max6.json.
