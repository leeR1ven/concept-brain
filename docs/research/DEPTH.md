# Depth requirements (read before starting)

The designer explicitly asked for DEEP research, not a quick survey. A shallow result based on a few papers is
considered a failure. Requirements:

1. Breadth: search widely (many queries, several phrasings, both reviews and primary experimental papers, recent
   work 2015-2026 as well as classics). Aim for 40+ distinct sources actually consulted. Follow citation trails:
   when a review cites a key experiment, look the experiment up.
2. Read, don't guess: base claims on sources you actually opened (abstracts at minimum, full text/figures where the
   detail matters). Never cite from memory without checking. Mark each claim's evidence strength
   (strong / moderate / weak / speculative) and note contradicting results and open controversies.
3. Mechanism, not labels: for every mechanism explain HOW it works step by step (which cells, which connections,
   which time scales, what numbers: firing fractions, time constants, connection probabilities, learning windows),
   so it could be implemented.
4. Compare competing theories (e.g. persistent-activity vs activity-silent working memory) and say which the
   evidence favours and why.
5. Read our actual code before writing implications: /root/work/concept-brain (brain/net.py = neurons, connections,
   learning rule; brain/net3d.py = 3D sensory network; brain/areas.py = areas; brain/brain.py = whole brain wiring;
   talk.py = dialogue training and checks; DESIGN.md). Point out concretely where our design CONTRADICTS the biology
   or misses something essential, not only where it matches.
6. Implications must be concrete and minimal: which region/connection/rule to change, what it replaces, expected
   effect, how we would test it, and what could go wrong. Rank them by expected impact on the open problems.
   The designer prefers a few simple rules that follow from structure; avoid ad-hoc patches, but do not
   over-simplify away something the evidence says is essential — say so explicitly if it is.
7. Think hard before writing conclusions: cross-check, look for counter-evidence, and revise.

Output: the full report (Chinese plain language; technical terms may keep English in parentheses) with a source list
(URLs). It can be long; structure it with headings and a short ranked recommendation section at the top.
