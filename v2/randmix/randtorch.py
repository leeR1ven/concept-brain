# -*- coding: utf-8 -*-
"""RandMix with its activity computed in torch on CUDA (the GPU form of the random layer).

State and learning stay in numpy (randlayer.RandMix: lines, weights, instar batches, pruning); after every learning
batch the weights (and, after pruning, the lines) are mirrored to the device.  activate(X) on the device:
  x = dense 0/1 input (D + 1 lines; line D is the empty slot of a pruned synapse)
  s = (x[lines] * w).sum(-1)                        (N, 2) branch sums: one gather over the fixed fan-in table
  eligible = s_a >= theta and (s_b >= theta or direct)
  key = population * 1000 + (eligible ? 100 - (s_a + s_b) - tie : 500)   (float64)
  one sort of the N keys; units are stored population by population, so a unit's rank in its population is its sorted
  position minus the population's offset; it fires iff eligible and rank < k_t.
Identical to RandMix.activate (same tie-break vector); with trained float weights a near-tie can flip in the last bit
(checked by test_torch.py).  The device tensors are not pickled (rebuilt on first use)."""
from __future__ import annotations

import numpy as np
import torch

from randlayer import RandMix

_T_KEYS = ('t_lines', 't_w', 't_type', 't_direct', 't_tie', 't_kt', 't_toff', 't_ar')


class TorchRandMix(RandMix):
    def __init__(self, device='cuda', mem_frac=0.0, **kw):
        self.device = device
        self.mem_frac = mem_frac
        super().__init__(**kw)
        self._dev_ok = False

    def _to_dev(self):
        d = self.device
        if self.mem_frac > 0 and d.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(self.mem_frac)
        self.t_lines = torch.from_numpy(np.where(self.lines >= 0, self.lines, self.D).astype(np.int64)).to(d)
        self.t_w = torch.from_numpy(self.w.astype(np.float32)).to(d)
        self.t_type = torch.from_numpy(self.type_of.astype(np.int64)).to(d)
        self.t_direct = torch.from_numpy(self.direct).to(d)
        self.t_tie = torch.from_numpy(self.tie.astype(np.float64)).to(d)
        self.t_kt = torch.from_numpy(self.kt.astype(np.int64)).to(d)
        self.t_toff = torch.from_numpy(self.toff.astype(np.int64)).to(d)
        self.t_ar = torch.arange(self.N, device=d)
        self._dev_ok = True

    def activate(self, X):
        if len(X) == 0:
            return np.zeros(0, np.int64)
        if not self._dev_ok:
            self._to_dev()
        d = self.device
        x = torch.zeros(self.D + 1, dtype=torch.float32, device=d)
        x[torch.from_numpy(np.asarray(X, np.int64)).to(d)] = 1.0
        s = (x[self.t_lines] * self.t_w).sum(2)                         # (N, 2)
        el = (s[:, 0] >= self.theta) & ((s[:, 1] >= self.theta) | self.t_direct)
        sc = (s[:, 0] + s[:, 1]).double()
        key = self.t_type.double() * 1000.0 + torch.where(el, 100.0 - sc - self.t_tie, torch.full_like(sc, 500.0))
        o = torch.argsort(key)
        pos = torch.empty_like(o)
        pos[o] = self.t_ar
        rank = pos - self.t_toff[self.t_type]
        keep = el & (rank < self.t_kt[self.t_type])
        return torch.nonzero(keep).squeeze(1).cpu().numpy().astype(np.int64)

    def _apply(self):
        super()._apply()
        if self._dev_ok:
            self.t_w = torch.from_numpy(self.w.astype(np.float32)).to(self.device)
            if self.prune_on:
                self.t_lines = torch.from_numpy(np.where(self.lines >= 0, self.lines, self.D).astype(np.int64)).to(self.device)

    def __getstate__(self):
        d = RandMix.__getstate__(self)
        for k in _T_KEYS:
            d.pop(k, None)
        d['_dev_ok'] = False
        return d
