# quick stage-A check: listen to stage-B model dialogues (no programs yet -> silent), then babble, then report.
import sys, json, time
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from run_lib import DiagWorld, new_agent, z_report, lexicon_chars, Timer
T = Timer()
w = DiagWorld('B', seed=0, tutor='model')
a = new_agent(w)
a.log_events = []
print('FEAT0', a.NF, T(), flush=True)
w.run(a, n_dialogues=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
print('after listening', z_report(a, w, lexicon_chars()), a.stats, 't', w.t, T(), flush=True)
w.babble(a, int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
print('after babbling', z_report(a, w, lexicon_chars()), a.stats, 't', w.t, T(), flush=True)
