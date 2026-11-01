# -*- coding: utf-8 -*-
"""Repo layout: v2.2 runs on ../v21 (ARCH_v2.1) and ../synth (the ARCH_v2 core and its bench copy), but four of its own
files replace the repo copies: agent_v21.py (dip_scope, core factory), rules.py (canonical DG tie rule), randlayer.py
(pow:p allocation) and submit.py.  Importing this module first puts this folder ahead of ../v21 and ../synth and loads
the replaced modules from here, so the path inserts that ../v21 files make later cannot pull in the unpatched copies."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(HERE, '..', 'synth'), os.path.join(HERE, '..', 'v21')):
    if _d not in sys.path:
        sys.path.insert(0, _d)
while HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
import rules       # noqa: E402,F401  (v2.2's copy: canonical tie rule)
import randlayer   # noqa: E402,F401  (v2.2's copy: pow:p allocation)
import agent_v21   # noqa: E402,F401  (v2.2's copy: dip_scope='all' default, core factory)
for _m in ('rules', 'randlayer', 'agent_v21'):
    assert os.path.dirname(os.path.abspath(sys.modules[_m].__file__)) == HERE, _m
