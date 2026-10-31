# summarise life_run.py outputs: overall, groups, stability (accuracy per tenth of life), retention by encounters
import sys, json, glob
f = lambda x: '-' if x is None else f'{x:.2f}'
for p in sys.argv[1:]:
    for fn in sorted(glob.glob(p)):
        R = json.load(open(fn)); S = R['summary']; g = S['groups']
        print(f"== {R['tag']} seed {R['seed']} ticks {S['ticks']} ({S['hours']:.1f} h) secs {R['secs']} cfg {R['cfg']}")
        print(f"   free replies {S['n_free']} acc {f(S['acc'])} | scene(new) {f(g['scene']['acc'])} (n {g['scene']['n']}) "
              f"repeat {f(g['scene_repeat']['acc'])} held-out {f(g['held_out']['acc'])} (n {g['held_out']['n']}) memory "
              f"{f(g['memory']['acc'])} (n {g['memory']['n']}) social {f(g['social']['acc'])} (n {g['social']['n']})")
        st = S['stability']
        print("   acc per tenth of life: " + " ".join(f(s['acc']) for s in st) + "   answer rate: " + " ".join(f(s['spoke']) for s in st))
        print("   scene(new) curve: " + " ".join(f(x) for x in g['scene']['curve']) + "   held-out curve: " + " ".join(f(x) for x in g['held_out']['curve']))
        qa = S['retention'].get('qa_by_encounters', [])
        print("   B-D accuracy by encounters: " + "  ".join(f"{d['encounters']}x {f(d['acc'])} (n {d['n']})" for d in qa))
        for t in ('fact_name', 'fact_person', 'fact_place', 'novel', 'before'):
            r = S['retention'].get(t) or {}
            if r.get('n'):
                print(f"   {t}: acc {f(r['acc'])} n {r['n']} | by encounters " + "  ".join(f"{d['encounters']}x {f(d['acc'])}/{d['n']}" for d in r['by_encounters']))
        tk = S['tasks']
        print("   tasks: " + " ".join(f"{k} {f(v['acc'])}" for k, v in sorted(tk.items()) if isinstance(v, dict) and v.get('n')))
        o = S['overlap']; m = S['mouth']
        print(f"   overlap ticks {o['overlap_ticks']} interruptions {o['interruptions']} | garble {m['garble']} of {m['syllables']} syl | "
              f"latency median {S['latency']['median']} | longest monologue {S['monologue']['longest_syllables']} | flags {S['flags'] or 'none'}")
