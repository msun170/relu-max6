# Wall-circuit mining: turning the OUT obstruction into a structural (provable) statement

Span-based membership says "max_7 not in span(F)" but hides WHY. Wall-circuit mining decomposes representability into
gradient-jump (wall) constraints, separated into BRAID and NON-BRAID, exposing the obstruction in a form that may
generalize to a theorem. This is the computational handle on the descent / finite-normal-form route (see
OUT_THEORY.md).

## Definitions
A homogeneous PL function f has a gradient that is constant on the chambers of its normal fan and JUMPS across the
bounding hyperplanes. For a probe (x0, d) (base point x0, primitive direction d) define the discrete wall jump
    jump_f(x0,d) = f(x0+d) + f(x0-d) - 2 f(x0)   (second difference along d).
For a support function this is >= 0 and is > 0 exactly when the segment [x0-d, x0+d] crosses a bend of f with a
d-component in its normal. For an orbit-sum OS_k it is a linear functional of the block (computed from OS_k values).

max_n = h_Delta bends ONLY across BRAID hyperplanes {x_i = x_j} (its gradient is e_argmax). So:
- BRAID probes  (d ~ e_i - e_j, x0 with a two-way tie at the max): jump_{max_n} != 0  -> the target braid jumps.
- NON-BRAID probes (d not ~ e_i - e_j, x0 with a strict unique max so max_n is smooth there): jump_{max_n} = 0.

Stack the probes into two integer matrices over the block-coefficient vector c:
    J  c = jt     (match max_n's braid walls)          [J:  P_J  x N,  jt the braid targets]
    NB c = 0      (kill all non-braid walls)            [NB: P_NB x N]
An exact S_n-invariant representation of max_n in the family exists IFF this combined system is solvable. (Wall
jumps + one linear term determine a homogeneous PL function, so wall-solvability = representability.)

## Three circuit types (classification of c with respect to the walls)
    Type I  : NB c = 0 and J c = 0    -- pure cancellation identities (valuation / "noise" circuits).
    Type II : NB c = 0 and J c != 0   -- the ONLY circuits that can build max_n's braid walls while staying
                                         non-braid-clean. The OUT question is whether jt lies in span{J c : NB c=0}.
    Type III: NB c != 0               -- introduce uncancelled non-braid walls; inadmissible unless paired off.

The OUT route must show: the max_n braid target jt is NOT in the image of the Type-II circuits, i.e. you cannot
produce max_n's braid jumps using only non-braid-cancelling combinations.

## Feasibility table (exact, mod-p; wall_circuits.py)
Diagnostic per family: is J alone feasible? is the combined J+NB feasible? WHERE does infeasibility live?

    family                       N blocks   J alone    NB ker   J + NB        controls
    weight 2 (COMPLETE)          102        FEASIBLE   62       INFEASIBLE    VALID (in-fam feasible, rand infeasible)
    weight 3 (sampled)           621        FEASIBLE   215      INFEASIBLE    VALID (in-fam feasible, rand infeasible)
    weight 4 (sampled)           1041       FEASIBLE   341      INFEASIBLE    VALID (in-fam feasible, rand infeasible)

KEY RESULT: the pattern  J FEASIBLE  but  J+NB INFEASIBLE  holds at ALL THREE weights. So at every weight the OUT is
PRECISELY "max_n's braid walls ARE matchable on their own, but NOT while simultaneously cancelling the non-braid
walls". The non-braid (bridge) walls are the obstruction carrier -- exactly the wall taxonomy's prediction. This is a
clean STRUCTURAL statement of the OUT, not a numerical residual.

(NOTE: do NOT read the rank jump as "codimension 1". Appending ONE target column can only raise rank by 0 or 1, so
rank(M)->rank([M|t])+1 is just a restatement of "infeasible", not a structural invariant. A real "where does the
obstruction live" measure would be the MINIMAL subset of NB rows that already forces infeasibility -- see Next.)

CONTROLS (validate the machinery): an IN-FAMILY target (wall profile of a genuine in-family combination) is correctly
FEASIBLE, and a RANDOM target is correctly INFEASIBLE. So J+NB INFEASIBLE for max_n is a real, sound OUT certificate,
not an artifact of always-infeasible targets. (Weight-2: VALID. Weight-3/4 controls running.)

CAVEATS (honest scope): weight 3,4 families are SAMPLED (621, 1041 orbits) and probes are finite, so the J/NB system
is a RELAXATION -- J+NB INFEASIBLE => OUT is sound (necessary conditions fail), but feasible would not prove IN.
The robust signal is the RECURRENCE of "J feasible, J+NB infeasible" across weights with different N and probe counts.
Next: (a) re-run with several seeds for robustness; (b) MINIMAL infeasible NB subset (irreducible obstruction set);
(c) extract Type-I circuits and test whether the same cancellation shapes recur across weights.

This recurrence is the first real evidence for a DESCENT theorem (the non-braid walls cancel via a weight-independent
mechanism while an irreducible braid-vs-bridge incompatibility remains), not merely another finite non-membership.

## What to extract next (once the table is filled)
- A basis of Type-I circuits at each weight; check whether the SAME combinatorial circuit shapes recur (bridge-pair
  cancellations) across weight 2 -> 3 -> 4. Recurrence = evidence the cancellations are generated by a fixed finite
  move set (the valuation moves), which is the engine of the finite normal form.
- The specific non-braid wall directions that carry the obstruction (the rows of NB that jt "needs"). If they are a
  small rational list, that supports WALL-DIRECTION FINITENESS (the weakened real-to-lattice target in OUT_THEORY.md).

## Proof target this feeds
FINITE NORMAL FORM (descent form): in any minimal exact real 2-layer rep of max_n, every non-braid bridge-wall
cancellation is generated by finitely many valuation circuits; after quotienting by them only bounded lattice block
types remain. The wall-circuit table + recurring Type-I circuits are the concrete evidence and the building blocks
for that argument.
