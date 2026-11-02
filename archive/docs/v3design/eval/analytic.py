"""Analytic checks for the evaluation design (no training).  1 process, tiny memory.
(a) config-key census of the v2.2 agent stack (AST), (b) specificity index from ARCH_v2 sec 8.3 lesions,
(c) power for a family-level gain, (d) compute budget of the minimal version."""
import ast, math, os, re, itertools
import numpy as np

V22 = 'arch/v2/v22/'
EXTRA = ['v2/hearing/agent_h.py', 'v2/hearing/hear.py',
         'v2/meaning/mcore.py', 'v2/meaning/meanlayer.py']
FILES = [V22 + f for f in ('core.py', 'core_think.py', 'core_synth.py', 'core_v21.py', 'core_v22.py', 'agent.py',
                           'agent_think.py', 'agent_synth.py', 'agent_v21.py', 'agent_v22.py', 'rules.py',
                           'randlayer.py')] + EXTRA

# ---- (a) census: config keys = self.X = literal inside Cfg-like __init__, dict(...) class attrs, keyword defaults
n_bool = n_num = n_str = 0; keys = set(); lits = 0; lit_files = {}
for f in FILES:
    src = open(f, encoding='utf-8').read(); tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and ('Cfg' in node.name):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant):
                    for t in sub.targets:
                        if isinstance(t, ast.Attribute):
                            keys.add(t.attr); v = sub.value.value
                            n_bool += isinstance(v, bool); n_num += (isinstance(v, (int, float)) and not isinstance(v, bool)); n_str += isinstance(v, str)
                if isinstance(sub, ast.Call) and getattr(sub.func, 'id', '') == 'dict':
                    for kw in sub.keywords:
                        if isinstance(kw.value, ast.Constant):
                            keys.add(kw.arg); v = kw.value.value
                            n_bool += isinstance(v, bool); n_num += (isinstance(v, (int, float)) and not isinstance(v, bool)); n_str += isinstance(v, str)
        if isinstance(node, (ast.FunctionDef,)) and node.name == '__init__':
            a = node.args
            for arg, d in zip(a.args[len(a.args) - len(a.defaults):], a.defaults):
                if isinstance(d, ast.Constant) and arg.arg not in ('self',):
                    keys.add(arg.arg)
            for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                if isinstance(d, ast.Constant):
                    keys.add(arg.arg)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if node.value not in (0, 1, -1):
                lits += 1; lit_files[os.path.basename(f)] = lit_files.get(os.path.basename(f), 0) + 1
print(f'(a) config keys (distinct names) {len(keys)}; Cfg-class literal values: bool {n_bool}, numeric {n_num}, '
      f'string-mode {n_str}; numeric literals outside {{0,1,-1}} in code: {lits}')
print('    per file:', dict(sorted(lit_files.items(), key=lambda x: -x[1])))

# ---- (b) specificity index SI = max_task drop / sum drops (ARCH_v2 sec 8.3; only the tasks the report lists)
base = dict(what=.62, where=.69, colour=.90, count=.83, ritual=.91, check=.66, name=.45)
les = {
    'scene focus F': dict(what=0.00, colour=0.00, count=0.08, check=0.12),
    'learned attention attD': dict(where=0.01, colour=0.00, count=0.00),
    'comparator/primes': dict(where=0.01, colour=0.00, count=0.00, check=0.33, what=0.40),
    'BG copy gate (prompt)': dict(ritual=0.14, what=0.15),
    'hippocampal recall routes': dict(name=0.10),
    'visual recognition VR': dict(what=0.52),
}
for m, d in les.items():
    drops = {t: max(base[t] - v, 0) for t, v in d.items()}
    tot = sum(drops.values()); si = max(drops.values()) / tot
    print(f'(b) {m:28s} drops {drops}  sum {tot:.2f}  SI {si:.2f}  tasks hit {len(drops)}')
# v2.1 gateR by stage: B .884->.427, C .971->.936, D unchanged
dB, dC = .884 - .427, .971 - .936
print(f'(b) gateR (say-what-I-recognise) by stage: SI {dB / (dB + dC):.2f}')

# ---- (c) power: per seed n held-out probes, paired gain over a same-item baseline, 5 seeds, one-sided t (df 4)
t95 = 2.132
for n in (30, 60, 100, 200):
    for sd_seed in (0.05, 0.08):
        se_seed = math.sqrt(2 * 0.25 / n + sd_seed ** 2)        # p~0.5 worst case, paired difference per seed
        mde = t95 * se_seed / math.sqrt(5)                       # minimal detectable mean gain (alpha .05, ~50% power)
        mde80 = (t95 + 0.941) * se_seed / math.sqrt(5)           # ~80% power (t_.20,4 = 0.941)
        print(f'(c) n {n:3d} per family per seed, seed sd {sd_seed}: detectable gain {mde:.3f} (50% power), {mde80:.3f} (80%)')

# ---- (d) compute: base life cost on the A800 CPU (ASSUMED inputs from reports), grid of the minimal version
life_local = 565 * 1.3          # ARCH_v2 10^6-tick life 565 s local; x1.3 for randmix + meaning (randmix staged 1270 vs 1030 s)
a800 = 0.75                     # A800 core speed relative to the local core (v21: staged 19-30 min vs ~17 min local)
probe = 1.25                    # + frozen-copy probes at 5 checkpoints (6 families x 100 items x ~90 ticks ~ 27% of a life)
base = life_local / a800 * probe
print(f'(d) base life (s=1, L=1e6 ticks) on A800 ~ {base:.0f} s; s=2 {2 * base:.0f} s; s=4 {4 * base:.0f} s (job limit 5400 s)')
cells = [(s, L) for s in (0.5, 1, 2) for L in (0.5, 1, 2)]
cpu = sum(s * L * base for s, L in cells)
print(f'(d) scaling grid 3x3 (s x L) at W=1, 1 agent: {cpu:.0f} CPU-s per seed; 3 seeds {3 * cpu:.0f} s = {3 * cpu / (4 * 5400):.1f} jobs')
print(f'    + W=2 at (1,1),(2,2),(1,2): {3 * (1 + 4 + 2) * base:.0f} s for 3 seeds = {3 * 7 * base / 21600:.1f} jobs')
print(f'    multiple-baseline main runs, 2 agents (v3, v2.2) x 5 seeds at (1,1): {10 * base:.0f} s = {10 * base / 21600:.1f} jobs')
print(f'    repair twins 2 agents x 5 seeds x 2 arms: {20 * base:.0f} s = {20 * base / 21600:.1f} jobs')
k = 6
print(f'    ablations {k} mechanisms x 5 seeds at (1,1): {k * 5 * base:.0f} s = {k * 5 * base / 21600:.1f} jobs')
print(f'    coverage (staged J2 5 seeds ~1600 s each, 2 lives, world2 2 agents 10 days ~1500 s): '
      f'{(5 * 1600 + 2 * base + 1500) * 2:.0f} s for 2 agents = {(5 * 1600 + 2 * base + 1500) * 2 / 21600:.1f} jobs')
# RNN baseline cost: GRU H=256, tokens per life
tok = 1e6 / 3 * 0.20                                    # 3 ticks per syllable, ~20 % of ticks with heard speech (ASSUMED)
H, E, EYE, Vh = 256, 64, 1024, 600
mac = 3 * H * (H + E + H) + H * Vh + EYE * H           # gates + output + eye projection per token (forward)
print(f'(d) RNN baseline: ~{tok:.0f} tokens per life, ~{3 * mac / 1e6:.1f} M MAC per token (fwd+bwd) -> '
      f'{tok * 3 * mac / 1e9:.0f} G MAC per life (~1-3 min numpy/torch CPU)')
