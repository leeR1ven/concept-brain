# -*- coding: utf-8 -*-
"""world2: an open house + garden where N models live continuously with scripted caregivers (see README.md).

    import sys; sys.path.insert(0, '<scratchpad>/arch/v2/bench')
    from world2 import World2, evaluate, headline, PerAgent
    w = World2(n_agents=4, seed=0)
    log = w.run(agents, ticks=100000)        # agents.step(obs) -> (N, 41)
    print(headline(evaluate(log)))
"""
from .core import World2, DEFAULTS, MOTOR, N_ACT, N_MOTOR, N_MOUTH, phys  # noqa: F401
from .agents import (PerAgent, MouthOnly, Explore, Silent, RandomActor, Echo, Lookup, Oracle, Namer,  # noqa: F401
                     Looker, REFERENCES)
from .metrics import evaluate, headline  # noqa: F401
from . import lex  # noqa: F401
