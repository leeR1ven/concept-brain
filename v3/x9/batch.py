"""run a list of concept.py configs in parallel: python3 -I batch.py specs.txt [n_proc]
specs.txt: one line per run: <name> key=value key=value ..."""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
specs = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
npr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
def one(sp):
    name, args = sp[0], sp[1:]
    t = time.time()
    with open(os.path.join(HERE, 'out', name + '.log'), 'w') as lg:
        r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'concept.py')] + args +
                           ['out=' + os.path.join(HERE, 'out', name + '.json')], stdout=lg, stderr=subprocess.STDOUT,
                           env=dict(os.environ, BENCH=HERE))
    print(f'{name} exit {r.returncode} {time.time() - t:.0f} s', flush=True)
with ThreadPoolExecutor(npr) as ex:
    list(ex.map(one, specs))
print('ALLDONE', flush=True)
