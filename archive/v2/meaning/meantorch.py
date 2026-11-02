# -*- coding: utf-8 -*-
"""MeanMix with the random layer's activity on CUDA (randmix's validated TorchRandMix)."""
from meanlayer import MeanMix
from randtorch import TorchRandMix


class TorchMeanMix(MeanMix, TorchRandMix):
    pass
