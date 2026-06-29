"""Write the explicit P2 decomposition of each max6 orbit representative as a join of two zonotopes.

Each building block in the construction was generated as a union of two weight-2 zonotopes. We recover, for
each of the six orbit representatives R, a pair (Z1, Z2) of zonotopes with Z1.vertices u Z2.vertices = R,
recording each zonotope's base point and braid generators. Saved to results/p2_decompositions_max6.json
and checked by check_max6.py.
"""
import itertools, json
import construction
import core

n = 6
W2 = core.weight2_points(n); W2set = set(W2)
gens = core.difference_generators(n)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

# zonotopes with provenance: vertex set -> (base point, generator list)
prov = {}
for p in W2:
    for r in range(1, 4):
        for T in itertools.combinations(range(len(gens)), r):
            verts = {p}; ok = True
            for s in range(1, r+1):
                for S in itertools.combinations(T, s):
                    q = p
                    for k in S: q = shift(q, gens[k])
                    if q not in W2set: ok = False; break
                    verts.add(q)
                if not ok: break
            if ok and len(verts) >= 2:
                fs = frozenset(verts)
                if fs not in prov: prov[fs] = (p, [gens[k] for k in T])
zonos = list(prov.keys())

coeffs, reps, D, L = construction.load()
out = []
for idx, rep in enumerate(reps):
    R = frozenset(tuple(v) for v in rep)
    sub = [z for z in zonos if z <= R]
    pair = None
    for i in range(len(sub)):
        for j in range(i, len(sub)):
            if sub[i] | sub[j] == R: pair = (sub[i], sub[j]); break
        if pair: break
    assert pair is not None, f"no two-zonotope decomposition for orbit {idx}"
    Z1, Z2 = pair; b1, g1 = prov[Z1]; b2, g2 = prov[Z2]
    out.append({
        "orbit_id": idx,
        "coeff": str(coeffs[idx]),
        "Z1_base": list(b1), "Z1_generators": [list(g) for g in g1], "Z1_vertices": sorted(map(list, Z1)),
        "Z2_base": list(b2), "Z2_generators": [list(g) for g in g2], "Z2_vertices": sorted(map(list, Z2)),
        "join_vertices": sorted(map(list, R)),
    })
json.dump(out, open("results/p2_decompositions_max6.json", "w"), indent=1)
print(f"wrote results/p2_decompositions_max6.json ({len(out)} orbit decompositions)")
