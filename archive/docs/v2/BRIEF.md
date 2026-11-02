# ARCH_v2 design phase — brief for every agent

All paths are relative to the scratchpad root `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/`.

## Mandate
The designer (user) has handed the whole design to us: "you decide; I'll look at what changed once it works. Test first
with a small model and short training; only when that clearly works, use the large model and long training. You can
keep discussing and testing continuously." The final goal is an AI architecture that **learns in real time**, aiming at
general intelligence. The current instance hears speech and answers through muscles.

## Read first
- `arch/review/SYNTHESIS.md` — the consensus of five whole-architecture reviews. **This is the starting point.**
- The five reports: `arch/review/{skeptic,learning,neuro,develop,compute}/report.md`. Their scripts and outputs sit in the
  same folders and are reusable: episode store, rule-v2 tests, assembly/attractor sims, cds_sim curriculum, babbling,
  error-driven readout.
- `arch/synth/ARCH_v1.md` §1 and `arch/synth/archv1.py` — the verified "body": F/GO/CD turn units, the 3-tick beat,
  program WTA, END/counter, D/I/M, with 0 errors in the integrated toy. Also `arch/content/fe.py` — the exact ear layer-0
  front end (FEAT0, T, Z mirrors).
- `research/FRAMEWORK.md` — the designer's original framework. It is guidance; the final goal wins.
- Code (read only): `/root/work/concept-brain/brain/net.py` (the current rule), `net3d.py`, `world/voice.py` (sound and
  mouth physics), `talk.py`.

## Decisions already taken (the five reviews agree; the designer delegated)
1. **Keep the body** as an innate reflex scaffold. Later, learned gates may tune or override it.
2. **The learning rule may be extended**, but it must stay local: each synapse sees only its own pre/post activity, its
   own traces, and at most a few broadcast signals. Allowed extensions:
   - an onset-based back term;
   - a prediction-error form: predicted → no more strengthening; predicted but absent → weaken;
   - learned inhibition through inhibitory units;
   - co-activity learning in association/assembly areas;
   - eligibility traces plus broadcast third factors (teacher/partner reaction, surprise/novelty, reward);
   - normalisation applied at learning time.
   State every rule exactly.
3. **Every learned part must reach behaviour**, proven by ablation. Do not keep learners that are only passengers.
4. **The hippocampus becomes a one-shot, long-lasting episodic index with offline replay** ("sleep") into slow cortex.
5. **Teaching:**
   - no muscle puppeting in the main regime. Instead: babbling → own-voice inverse map → imitation → learning by listening
     (predict what will be heard next, from every speaker, including overheard dialogues) → contingent partner feedback.
   - Teacher-forcing may remain only as an optional bootstrap or baseline.
6. **A staged, grounded micro-world comes first**: a scene the model sees, child-directed routines, held-out
   combinations, and tasks where echo scores 0. The adult corpus comes last, interleaved.
7. **Memory and association areas use sparse codes** (≤ 5 % active) with high fan-in (~1,000), where they store
   sequences, episodes or assemblies.
8. **The PFC is built on learned inputs**:
   - lexical/meaning layers and hippocampal recall;
   - per-layer persistence (leaky units, which kill synchronous period-2 alternation);
   - BG-like gating that updates while the partner speaks and holds during the reply;
   - assemblies;
   - a readout at every boundary;
   - slow context layers (≈ line gist, ≈ dialogue topic).
9. **Few general mechanisms.** Every new fixed, task-specific mechanism needs a justification and a plan for how it
   could later be learned.

## The shared bench
- An agent is building `arch/v2/bench/` to the contract in `arch/v2/bench/API.md`. When it works, it writes the file
  `arch/v2/bench/READY`.
- Until then, unit-test your mechanisms on your own toys. Afterwards, run your prototype on the bench so that results
  are comparable.
- You may copy bench code into your folder and extend it there. Report any bench bug to the coordinator in your final
  reply; don't silently fork the metrics.

## Rules for all agents
- Write only inside your own folder `arch/v2/<your_folder>/`. Read anything under `research/` and `arch/` and the code
  (read only). Never modify `/root/work` or other agents' folders. Read no other file in the scratchpad root.
- No SSH, no remote servers, no git. Web search and fetch for papers are allowed.
- Compute: `export OMP_NUM_THREADS=1`; at most one heavy process at a time; keep each run under about 30 min (shared
  4-CPU box). Put long runs in the background with `nice`.
- Label claims PROVED (argument with numbers), SIMULATED (numpy run; file and numbers), CITED (paper and finding) or
  ASSUMED.
- The harness may block writing `report.md`. Try writing it anyway, and **always return the full report inline in your
  final reply**. Write in English, and end with a short plain Chinese summary (summary_zh).
