"""Render the style test frame by frame to EXR. Timeline and rule live in shot_test1.py.
usage: python3 render_test1.py <meshdir> <outdir> <first> <last> <step> <w> <h> <spp>"""
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grammar as g  # noqa: E402
import shot_test1 as s  # noqa: E402

A_POS, B_POS = np.array((-3.5, 0.0, 0.0)), np.array((3.5, 0.0, 0.0))


def scene(meshdir, w, h, spp):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    g.setup_render(w, h, spp)
    g.world()
    mem = g.membrane()
    a = g.Neuron(g.mesh_from_npz(f'{meshdir}/a.npz', 'a'), mem, A_POS, rot=(0.3, 0.2, 0.4), name='A', light=120.0)
    b = g.Neuron(g.mesh_from_npz(f'{meshdir}/b.npz', 'b'), mem, B_POS, rot=(-0.2, 0.5, 2.1), name='B', light=120.0)
    gold, cyan = g.fiber_material('ab', g.GOLD_CORE), g.fiber_material('in', g.CYAN)
    fib = {
        'AB': g.Fiber('AB', [A_POS + (0.36, 0, 0.02), (-1.6, 0.5, 0.42), (0.7, -0.4, -0.26), B_POS - (0.36, 0, 0.02)],
                      gold, wobble=0.05, seed=2),
        'inA': g.Fiber('inA', [(-15, 3.2, 4.2), (-9.5, 1.2, 2.6), (-6, -0.4, 1.0), A_POS + (-0.34, -0.05, 0.12)],
                       cyan, wobble=0.06, seed=5),
        'inB': g.Fiber('inB', [(15, 3.0, -4.0), (9.5, 1.0, -2.4), (6, -0.5, -1.0), B_POS + (0.34, -0.05, -0.1)],
                       cyan, wobble=0.06, seed=7),
    }
    g.area_light('key', (-5, -9, 8), (0, 0, 0), 9, (0.55, 0.72, 1.0), 170)
    g.area_light('fill', (8, 7, -5), (0, 0, 0), 12, (0.34, 0.42, 0.9), 70)
    g.dust()
    cam = g.Camera()
    rides = {k: g.TravelLight(k + '_l', g.GOLD_CORE if k == 'AB' else g.CYAN, 10.0) for k in fib}
    return a, b, fib, cam, rides


def frame(t, a, b, fib, cam, rides):
    st = s.state(t)
    a.set(*st['lit_A'])
    b.set(*st['lit_B'])
    head = {k: [] for k in fib}
    for name, x in st['pulses']:
        head[name].append(x)
    for k, f in fib.items():
        r = np.full(f.rings, 0.03 if k != 'AB' else s.radius_of(st['w']))
        glow = np.zeros(f.rings)
        for x in head[k]:
            glow = np.maximum(glow, s.comet(f.u, x))
        taper = 1.0 + 1.1 * np.exp(-f.u / 0.03) + 1.1 * np.exp(-(1 - f.u) / 0.03) if k == 'AB' else 1.0
        f.set(r * taper, glow.astype(np.float32))
        x = max(head[k], default=None)
        ok = x is not None and x <= 1.0
        rides[k].set(f.at(x) if ok else None, 1.0 if ok else 0.0)
    c = s.catmull_time(s.CAM_KEYS, t)
    az, el, dist, tx, tz = np.radians(c[0]), np.radians(c[1]), c[2], c[3], c[4]
    target = np.array((tx, 0.0, tz))
    cam.set(target + dist * np.array((np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el))), target)


def main():
    meshdir, out, first, last, step, w, h, spp = sys.argv[1:9]
    first, last, step, w, h, spp = map(int, (first, last, step, w, h, spp))
    os.makedirs(out, exist_ok=True)
    a, b, fib, cam, rides = scene(meshdir, w, h, spp)
    sc = bpy.context.scene
    for i in range(first, last + 1, step):
        p = f'{out}/f{i:05d}.exr'
        if os.path.exists(p):
            continue
        t0 = time.time()
        frame(i / s.FPS, a, b, fib, cam, rides)
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        print(f'FRAME {i} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
