# per-layer breakdown of unit firing rates in the X10 sound area (s=1): dead / rare / mid / always-on
import runpy, sys, numpy as np
sys.argv = ['sound_stats.py', '6000', '1']
g = runpy.run_path(sys.argv[0] if False else 'sound_stats.py')
