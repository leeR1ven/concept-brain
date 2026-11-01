# Read-only census: list every named knob (with default) per layer, for the v3 gene table.
# Run: python3 -B knob_list.py   (no .pyc written into the repo)
import os, sys, inspect, os, re
R = 'v2'
for d in ('synth', 'v21', 'hearing', 'meaning', 'randmix'):
    sys.path.insert(0, os.path.join(R, d))
sys.path.insert(0, os.path.join(R, 'synth', 'bench_copy'))
import core, core_think, core_synth, core_v21, agent_think, agent_synth, agent_v21, agent_h, meanlayer, randlayer, hear
from body import Body

def sig(f, skip=('self', 'world', 'kw', 'cfgkw', 'a')):
    return {k: v.default for k, v in inspect.signature(f).parameters.items()
            if k not in skip and v.default is not inspect._empty}

L = {}
L['Cfg'] = dict(vars(core.Cfg()))
L['TCfg'] = {k: v for k, v in vars(core_think.TCfg()).items() if k not in L['Cfg']}
L['SCfg'] = dict(core_synth.SCfg.EXTRA) if isinstance(core_synth.SCfg.EXTRA, dict) else {k: None for k in core_synth.SCfg.EXTRA}
L['RCfg'] = dict(core_v21.RCfg.RAL) if isinstance(core_v21.RCfg.RAL, dict) else {k: None for k in core_v21.RCfg.RAL}
L['DevAgent'] = sig(__import__('agent').DevAgent.__init__)
L['ThinkAgent'] = {k: v for k, v in sig(agent_think.ThinkAgent.__init__).items() if k not in L['DevAgent']}
L['SynthAgent'] = dict(agent_synth.SYNTH_AGENT_KEYS) if isinstance(agent_synth.SYNTH_AGENT_KEYS, dict) else {k: None for k in agent_synth.SYNTH_AGENT_KEYS}
L['AgentV21'] = dict(agent_v21.V21_AGENT_KEYS) if isinstance(agent_v21.V21_AGENT_KEYS, dict) else {k: None for k in agent_v21.V21_AGENT_KEYS}
L['AgentH'] = dict(agent_h.H_KEYS) if isinstance(agent_h.H_KEYS, dict) else {k: None for k in agent_h.H_KEYS}
L['MeanCodes'] = {k: v for k, v in sig(meanlayer.MeanCodes.__init__).items() if k not in ('seed', 'cls')}
L['RandMix'] = {k: v for k, v in sig(randlayer.RandMix.__init__).items() if k != 'seed'}
L['Body'] = {k: v for k, v in sig(Body.__init__).items() if k != 'seed'}
L['SoundGate'] = sig(hear.SoundGate.__init__)
src = open('arch/v2/v22/agent_v22.py').read()
m = re.search(r"V22_AGENT_KEYS = dict\(V21_AGENT_KEYS, (.*?)\)\n", src, re.S)
L['AgentV22'] = dict(re.findall(r"(\w+)=([^,]+)", m.group(1))) if m else {}
tot = 0
for k, v in L.items():
    tot += len(v)
    print(f'== {k} ({len(v)})')
    print('   ' + ', '.join(f'{a}={b!r}' for a, b in v.items()))
print('total', tot)
