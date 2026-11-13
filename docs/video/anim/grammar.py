"""Visual grammar of the series, as Blender/Cycles building blocks.

One definition of how things look, so every episode says the same thing the same way:
  lit neuron      = fixed warm emission; there is no in-between level and no brighter level
  dark neuron     = wet, dark, cool membrane, seen only by its rim
  signal          = a comet of light travelling along a connection
  world input     = the same comet in cyan, arriving from outside the frame
  thickness       = connection strength; learning changes thickness and nothing else
"""
import math

import bpy
import numpy as np

GOLD_CORE = (1.0, 0.60, 0.22)
GOLD_RIM = (1.0, 0.34, 0.06)
CYAN = (0.16, 0.72, 1.0)
FOG = (0.0075, 0.0165, 0.034)
LIT_LEVEL = 1.05  # the one and only brightness of a lit neuron
RIM_DARK = (0.30, 0.62, 1.0)


def setup_render(w, h, spp, denoise=True, threshold=0.012):
    sc = bpy.context.scene
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "CUDA"
    p.get_devices()
    for d in p.devices:
        d.use = d.type == "CUDA"
    sc.render.engine = "CYCLES"
    c = sc.cycles
    c.device = "GPU"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    c.samples, c.adaptive_threshold, c.use_adaptive_sampling = spp, threshold, True
    c.adaptive_min_samples = max(16, spp // 8)
    c.seed, c.use_animated_seed = 11, False
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 5, 2, 3
    c.transmission_bounces, c.volume_bounces, c.transparent_max_bounces = 4, 0, 8
    c.caustics_reflective, c.caustics_refractive, c.blur_glossy = False, False, 1.0
    c.sample_clamp_direct, c.sample_clamp_indirect = 0.0, 5.0
    c.filter_width = 1.4
    c.use_auto_tile = False
    c.use_denoising = denoise
    if denoise:
        c.denoiser = "OPENIMAGEDENOISE"
        c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        c.denoising_prefilter = "ACCURATE"
    sc.render.use_persistent_data = True
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    im = sc.render.image_settings
    im.file_format, im.color_mode, im.color_depth, im.exr_codec = "OPEN_EXR", "RGB", "16", "ZIP"
    sc.render.use_compositing = False
    sc.render.use_sequencer = False
    return sc


def world(color=FOG):
    w = bpy.data.worlds.new("void")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*color, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    w.cycles.sampling_method = "NONE"
    bpy.context.scene.world = w


def _n(nt, kind, **kw):
    node = nt.nodes.new(kind)
    for k, v in kw.items():
        setattr(node, k, v)
    return node


def _principled(nt, base, rough, coat=0.6):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.16
    b.inputs["Specular IOR Level"].default_value = 0.55
    return b


def _bump(nt, b, scale, strength):
    """Fine, irregular membrane detail, so surfaces never look like clean CG plastic."""
    tc = _n(nt, "ShaderNodeTexCoord")
    no = _n(nt, "ShaderNodeTexNoise")
    no.inputs["Scale"].default_value = scale
    no.inputs["Detail"].default_value = 5.0
    no.inputs["Roughness"].default_value = 0.55
    bp = _n(nt, "ShaderNodeBump")
    bp.inputs["Strength"].default_value = strength
    bp.inputs["Distance"].default_value = 0.04
    nt.links.new(tc.outputs["Object"], no.inputs["Vector"])
    nt.links.new(no.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Coat Normal"])
    return no


def _math(nt, op, a, b=None, clamp=False):
    m = _n(nt, "ShaderNodeMath", operation=op, use_clamp=clamp)
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _rim(nt):
    lw = _n(nt, "ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42
    return lw.outputs["Facing"]


def membrane(name="membrane", rim_gain=0.34, fog_len=0.0):
    """Neuron body. Object colour: R = lit (0..1), G = ignition front (path length)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.cycles.emission_sampling = "NONE"  # lit neurons light the scene through point lights
    nt = m.node_tree
    nt.nodes.clear()
    out = _n(nt, "ShaderNodeOutputMaterial")
    b = _principled(nt, (0.018, 0.032, 0.062), 0.34)
    no = _bump(nt, b, 7.0, 0.22)
    rim = _rim(nt)

    rimE = _n(nt, "ShaderNodeEmission")
    rimE.inputs["Color"].default_value = (*RIM_DARK, 1)
    nt.links.new(_math(nt, "MULTIPLY", _math(nt, "POWER", rim, 2.2), rim_gain), rimE.inputs["Strength"])
    add = _n(nt, "ShaderNodeAddShader")
    nt.links.new(b.outputs[0], add.inputs[0])
    nt.links.new(rimE.outputs[0], add.inputs[1])
    surf = add.outputs[0]

    oi = _n(nt, "ShaderNodeObjectInfo")
    sep = _n(nt, "ShaderNodeSeparateColor")
    nt.links.new(oi.outputs["Color"], sep.inputs[0])
    if fog_len == 0.0:
        at = _n(nt, "ShaderNodeAttribute", attribute_name="d", attribute_type="GEOMETRY")
        front = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", sep.outputs[1], at.outputs["Fac"]), 0.22, clamp=True)
        mask = _math(nt, "MULTIPLY", sep.outputs[0], front)
        # hottest at the cell body, still clearly lit at the branch tips
        fall = _math(nt, "ADD", 0.34, _math(nt, "MULTIPLY", 1.5, _math(nt, "EXPONENT", _math(nt, "MULTIPLY", at.outputs["Fac"], -1.5))))
        mix = _n(nt, "ShaderNodeMix", data_type="RGBA")
        mix.inputs["A"].default_value = (*GOLD_CORE, 1)
        mix.inputs["B"].default_value = (*GOLD_RIM, 1)
        nt.links.new(_math(nt, "POWER", rim, 0.8), mix.inputs["Factor"])
        e = _n(nt, "ShaderNodeEmission")
        nt.links.new(mix.outputs["Result"], e.inputs["Color"])
        grain = _math(nt, "ADD", 0.62, _math(nt, "MULTIPLY", no.outputs["Fac"], 0.76))
        shade = _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", _math(nt, "POWER", rim, 1.3), 0.8))  # edges dimmer, so it stays round
        lvl = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", mask, fall), grain), shade), LIT_LEVEL)
        nt.links.new(lvl, e.inputs["Strength"])
        add2 = _n(nt, "ShaderNodeAddShader")
        nt.links.new(surf, add2.inputs[0])
        nt.links.new(e.outputs[0], add2.inputs[1])
        surf = add2.outputs[0]
    else:
        # distant tissue dissolves into the fog
        cam = _n(nt, "ShaderNodeCameraData")
        f = _math(nt, "SUBTRACT", 1.0, _math(nt, "EXPONENT", _math(nt, "DIVIDE", cam.outputs["View Distance"], -fog_len)))
        fogE = _n(nt, "ShaderNodeEmission")
        fogE.inputs["Color"].default_value = (*FOG, 1)
        mx = _n(nt, "ShaderNodeMixShader")
        nt.links.new(f, mx.inputs[0])
        nt.links.new(surf, mx.inputs[1])
        nt.links.new(fogE.outputs[0], mx.inputs[2])
        surf = mx.outputs[0]
    nt.links.new(surf, out.inputs["Surface"])
    return m


def fiber_material(name, color, rim_gain=0.42):
    """Connection. Per-vertex attribute `glow` carries the travelling light."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.cycles.emission_sampling = "NONE"
    nt = m.node_tree
    nt.nodes.clear()
    out = _n(nt, "ShaderNodeOutputMaterial")
    b = _principled(nt, (0.02, 0.036, 0.07), 0.3)
    _bump(nt, b, 18.0, 0.1)
    rim = _rim(nt)
    at = _n(nt, "ShaderNodeAttribute", attribute_name="glow", attribute_type="GEOMETRY")
    rimE = _n(nt, "ShaderNodeEmission")
    rimE.inputs["Color"].default_value = (*RIM_DARK, 1)
    nt.links.new(_math(nt, "MULTIPLY", _math(nt, "POWER", rim, 1.5), rim_gain), rimE.inputs["Strength"])
    # the core of a pulse burns towards white, the skin keeps the hue
    mix = _n(nt, "ShaderNodeMix", data_type="RGBA")
    mix.inputs["A"].default_value = (*color, 1)
    mix.inputs["B"].default_value = (1.0, 0.95, 0.86, 1)
    nt.links.new(_math(nt, "POWER", at.outputs["Fac"], 3.0), mix.inputs["Factor"])
    e = _n(nt, "ShaderNodeEmission")
    nt.links.new(mix.outputs["Result"], e.inputs["Color"])
    nt.links.new(_math(nt, "MULTIPLY", at.outputs["Fac"], 26.0), e.inputs["Strength"])
    a1 = _n(nt, "ShaderNodeAddShader")
    a2 = _n(nt, "ShaderNodeAddShader")
    nt.links.new(b.outputs[0], a1.inputs[0])
    nt.links.new(rimE.outputs[0], a1.inputs[1])
    nt.links.new(a1.outputs[0], a2.inputs[0])
    nt.links.new(e.outputs[0], a2.inputs[1])
    nt.links.new(a2.outputs[0], out.inputs["Surface"])
    return m


def mesh_from_npz(path, name):
    z = np.load(path)
    v, f, d = z["v"].astype(np.float32), z["f"].astype(np.int32), z["d"].astype(np.float32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.ravel())
    me.loops.add(f.size)
    me.loops.foreach_set("vertex_index", f.ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3))
    me.update()
    me.validate(verbose=False)
    me.polygons.foreach_set("use_smooth", np.ones(len(f), bool))
    me.attributes.new("d", "FLOAT", "POINT").data.foreach_set("value", d)
    me.update()
    return me


class Neuron:
    def __init__(self, mesh, mat, loc, rot=(0, 0, 0), scale=1.0, name="neuron", light=None):
        o = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(o)
        o.location, o.rotation_euler = loc, rot
        o.scale = (scale, scale, scale)
        o.data.materials.append(mat) if not o.data.materials else None
        o.color = (0, 0, 0, 1)
        o.visible_shadow = False  # the cell-body light sits inside this mesh
        self.obj = o
        self.light = None
        if light is not None:
            L = bpy.data.lights.new(name + "_glow", "POINT")
            L.color, L.shadow_soft_size, L.energy = GOLD_CORE, 0.34, 0.0
            lo = bpy.data.objects.new(name + "_glow", L)
            bpy.context.collection.objects.link(lo)
            lo.location = loc
            self.light, self.light_power = L, light

    def set(self, lit, front=9.0):
        self.obj.color = (float(lit), float(front), 0, 1)
        if self.light is not None:
            self.light.energy = float(lit) * self.light_power


def catmull(P, n):
    P = np.asarray(P, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    seg = len(P) - 3
    t = np.linspace(0, seg, n, endpoint=True)
    i = np.minimum(t.astype(int), seg - 1)
    u = (t - i)[:, None]
    a, b, c, d = P[i], P[i + 1], P[i + 2], P[i + 3]
    return 0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u**2 + (3 * b - a - 3 * c + d) * u**3)


class Fiber:
    """A tube whose radius and travelling glow are written every frame."""

    def __init__(self, name, ctrl, mat, rings=520, segs=16, wobble=0.0, seed=1):
        C = catmull(ctrl, rings)
        if wobble:
            rng = np.random.default_rng(seed)
            s = np.linspace(0, 1, rings)
            for ax in range(3):
                for k in (5, 9, 14):
                    C[:, ax] += wobble / k * 5 * np.sin(s * k * 3.1 + rng.uniform(0, 6.28)) * np.sin(np.pi * s) ** 0.5
        L = np.linalg.norm(np.diff(C, axis=0), axis=1)
        self.u = np.concatenate([[0], np.cumsum(L)]) / L.sum()
        self.length = float(L.sum())
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.zeros_like(C)
        N[0] = np.cross(T[0], [0.31, 0.27, 0.91])
        N[0] /= np.linalg.norm(N[0])
        for i in range(1, rings):  # parallel transport, so the tube never twists
            n = N[i - 1] - T[i] * (N[i - 1] @ T[i])
            N[i] = n / np.linalg.norm(n)
        self.C, self.N, self.B = C, N, np.cross(T, N)
        self.rings, self.segs = rings, segs
        a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        self.ca, self.sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
        i, j = np.meshgrid(np.arange(rings - 1), np.arange(segs), indexing="ij")
        j2 = (j + 1) % segs
        q = np.stack([i * segs + j, (i + 1) * segs + j, (i + 1) * segs + j2, i * segs + j2], -1).reshape(-1, 4)
        caps = [list(range(segs))[::-1], list(range((rings - 1) * segs, rings * segs))]
        me = bpy.data.meshes.new(name)
        v0 = np.repeat(C, segs, axis=0)
        me.from_pydata(v0.tolist(), [], q.tolist() + caps)
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
        me.attributes.new("glow", "FLOAT", "POINT")
        me.materials.append(mat)
        me.update()
        self.me = me
        self.obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(self.obj)
        self._last = None

    def set(self, radius, glow):
        key = (radius.tobytes(), glow.tobytes())
        if key == self._last:
            return
        self._last = key
        r = radius[:, None, None]
        p = self.C[:, None, :] + r * (self.ca * self.N[:, None, :] + self.sa * self.B[:, None, :])
        self.me.vertices.foreach_set("co", p.astype(np.float32).ravel())
        self.me.attributes["glow"].data.foreach_set("value", np.repeat(glow, self.segs).astype(np.float32))
        self.me.update()

    def at(self, u):
        i = int(np.clip(np.searchsorted(self.u, u), 0, self.rings - 1))
        return self.C[i]


class TravelLight:
    """A small light riding with a pulse, so nearby membranes catch it as it passes."""

    def __init__(self, name, color, power):
        L = bpy.data.lights.new(name, "POINT")
        L.color, L.shadow_soft_size, L.energy = color, 0.08, 0.0
        self.o = bpy.data.objects.new(name, L)
        bpy.context.collection.objects.link(self.o)
        self.L, self.power = L, power

    def set(self, pos, level):
        if pos is not None:
            self.o.location = pos
        self.L.energy = float(level) * self.power


def area_light(name, loc, target, size, color, power):
    L = bpy.data.lights.new(name, "AREA")
    L.shape, L.size, L.color, L.energy = "DISK", size, color, power
    L.cycles.is_portal = False
    o = bpy.data.objects.new(name, L)
    bpy.context.collection.objects.link(o)
    o.location = loc
    d = np.array(target, float) - np.array(loc, float)
    az = math.atan2(d[0], d[1])
    el = math.atan2(d[2], math.hypot(d[0], d[1]))
    o.rotation_mode = "XYZ"
    o.rotation_euler = (math.pi / 2 + el, 0, -az)
    return o


def dust(n=520, box=((-11, 11), (-8, 12), (-6.5, 6.5)), seed=3, strength=1.4):
    """Slow specks in the tissue. They give depth through parallax and soft bokeh."""
    rng = np.random.default_rng(seed)
    ico_v = np.array([[0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618], [1, 1.618, 0], [-1, 1.618, 0],
                      [1, -1.618, 0], [-1, -1.618, 0], [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1]], float)
    ico_v /= np.linalg.norm(ico_v[0])
    ico_f = [(0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1), (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
             (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6), (3, 6, 7), (3, 10, 6), (3, 2, 10), (3, 11, 2), (3, 7, 11)]
    V, F = [], []
    for i in range(n):
        c = [rng.uniform(*box[k]) for k in range(3)]
        r = rng.uniform(0.006, 0.02)
        V.append(ico_v * r + c)
        F += [(a + 12 * i, b + 12 * i, cc + 12 * i) for a, b, cc in ico_f]
    me = bpy.data.meshes.new("dust")
    me.from_pydata(np.concatenate(V).tolist(), [], F)
    m = bpy.data.materials.new("dust")
    m.use_nodes = True
    m.cycles.emission_sampling = "NONE"
    nt = m.node_tree
    nt.nodes.clear()
    e = _n(nt, "ShaderNodeEmission")
    e.inputs["Color"].default_value = (0.42, 0.66, 1.0, 1)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], _n(nt, "ShaderNodeOutputMaterial").inputs["Surface"])
    me.materials.append(m)
    o = bpy.data.objects.new("dust", me)
    bpy.context.collection.objects.link(o)
    o.visible_shadow = False
    return o


class Camera:
    def __init__(self, lens=38.0, fstop=0.16):
        c = bpy.data.cameras.new("cam")
        c.lens, c.sensor_width = lens, 36.0
        c.dof.use_dof = True
        c.dof.aperture_fstop = fstop
        c.dof.aperture_blades = 0
        c.clip_start, c.clip_end = 0.1, 300
        self.o = bpy.data.objects.new("cam", c)
        bpy.context.collection.objects.link(self.o)
        bpy.context.scene.camera = self.o
        self.c = c

    def set(self, pos, target, focus=None):
        pos, target = np.asarray(pos, float), np.asarray(target, float)
        d = target - pos
        self.o.location = pos
        az = math.atan2(d[0], d[1])
        el = math.atan2(d[2], math.hypot(d[0], d[1]))
        self.o.rotation_mode = "XYZ"
        self.o.rotation_euler = (math.pi / 2 + el, 0, -az)
        self.c.dof.focus_distance = float(np.linalg.norm(d)) if focus is None else float(focus)
