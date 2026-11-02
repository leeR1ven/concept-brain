"""Analytic: how many random conjunction units does 'depth by width' need?
randmix c_flat geometry: an item = 16-of-512 lines per bundle, a unit draws fan=3 lines per branch,
a branch fires iff >=1 of its lines is active (theta 1, weights 1). A unit with m branches (one per group)
responds to a specific m-tuple of items with probability p1^m, p1 = P(branch hits the item's code).
Exact hypergeometric p1; then units needed so that a specific m-tuple has >= c responsive units (expected),
and the total over all C(G, m) group combinations (G = 14 direct groups as in randmix)."""
from math import comb
n_c, k_c, fan, G, c = 512, 16, 3, 14, 10
p1 = 1 - comb(n_c - k_c, fan) / comb(n_c, fan)
print(f"p1 (one branch hits a given item) = {p1:.4f}")
for m in (1, 2, 3, 4, 5):
    pm = p1 ** m
    per_type = c / pm
    n_types = comb(G, m)
    print(f"m={m}: p={pm:.2e}  units per type for {c} responders = {per_type:,.0f}  "
          f"types C({G},{m})={n_types}  total = {per_type * n_types:,.0f}")
# today's budget: ~48k units over 45 populations (14 direct + 31 pair types)
print("today: ~47.9k units / 45 populations = ~1,060 per type ->", f"{1060 * p1**2:.1f} responders per specific pair")
