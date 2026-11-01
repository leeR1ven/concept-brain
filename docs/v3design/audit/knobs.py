# Count the configuration knobs (named constructor/config parameters with defaults) per version layer.
# Read-only: imports the repo modules, constructs nothing heavy.
import sys, inspect, os
R = 'v2'
for d in ('synth', 'v21', 'hearing', 'meaning', 'randmix'):
    sys.path.insert(0, os.path.join(R, d))
sys.path.insert(0, os.path.join(R, 'synth', 'bench_copy'))
import core, core_think, core_synth, core_v21, agent_think, agent_synth, agent_v21, agent_h, meanlayer, randlayer, rules, hear
from body import Body

def sig(f, skip=('self', 'world', 'kw', 'cfgkw', 'a')):
    return {k: v.default for k, v in inspect.signature(f).parameters.items()
            if k not in skip and v.default is not inspect._empty}

layers = {}
layers['core Cfg (develop+tune)'] = set(vars(core.Cfg()).keys())
layers['TCfg (think core)'] = set(vars(core_think.TCfg()).keys()) - layers['core Cfg (develop+tune)']
layers['SCfg (synth core)'] = set(core_synth.SCfg.EXTRA)
layers['RCfg (v2.1 core)'] = set(core_v21.RCfg.RAL)
layers['DevAgent'] = set(sig(__import__('agent').DevAgent.__init__))
layers['ThinkAgent'] = set(sig(agent_think.ThinkAgent.__init__)) - layers['DevAgent']
layers['SynthAgent'] = set(agent_synth.SYNTH_AGENT_KEYS)
layers['AgentV21'] = set(agent_v21.V21_AGENT_KEYS)
layers['AgentH (hearing)'] = set(agent_h.H_KEYS)
layers['MeanCodes (meaning)'] = set(sig(meanlayer.MeanCodes.__init__)) - {'seed', 'cls'}
layers['RandMix (randmix)'] = set(sig(randlayer.RandMix.__init__)) - {'seed'}
layers['Body (ARCH_v1, bench)'] = set(sig(Body.__init__)) - {'seed'}
layers['SoundGate'] = set(sig(hear.SoundGate.__init__))
try:
    sys.path.insert(0, 'arch/v2/v22')
    src = open('arch/v2/v22/agent_v22.py').read()
    import re
    m = re.search(r"V22_AGENT_KEYS = dict\(V21_AGENT_KEYS, (.*?)\)\n", src)
    layers['AgentV22 (in progress)'] = set(re.findall(r"(\w+)=", m.group(1)))
except Exception as e:
    print('v22 parse failed', e)
tot = 0
for k, v in layers.items():
    tot += len(v)
    print(f'{k:28s} {len(v):3d}  {sorted(v)[:14]}{" ..." if len(v) > 14 else ""}')
print('total named knobs', tot)
# hard-coded literals in code paths (not exposed as knobs): count numeric literals in the agent/core step files
import re, io, tokenize
def lits(path):
    n = 0
    with open(path, 'rb') as f:
        for t in tokenize.tokenize(f.readline):
            if t.type == tokenize.NUMBER and t.string not in ('0', '1', '0.0', '1.0', '2', '-1'):
                n += 1
    return n
for p in ['synth/agent.py', 'synth/agent_think.py', 'synth/core_think.py', 'synth/core.py', 'synth/core_synth.py',
          'synth/agent_synth.py', 'v21/core_v21.py', 'v21/agent_v21.py', 'hearing/agent_h.py', 'hearing/hear.py',
          'meaning/meanlayer.py', 'randmix/randlayer.py', 'synth/rules.py']:
    print(f'{p:26s} numeric literals (excl. 0/1/2/-1): {lits(os.path.join(R, p))}')
