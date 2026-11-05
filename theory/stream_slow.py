"""stream_record.py with slower speech: each syllable lasts SLOW x 3 ticks, so the last layer has time to answer before the next
syllable begins. The cochlea plays each of a syllable's 3 frames SLOW times. Labels keep the true tick, so onset is tick 0."""
import sys

SLOW = int(sys.argv.pop(1))
import core.world.corpus as cw  # noqa: E402

orig = cw.script_dialogue


def slow(*a, **k):
    out = []
    for x in orig(*a, **k):
        if x is None:
            out.append(None)
        else:
            out += [(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], i) for i in range(SLOW)]
    return out


cw.script_dialogue = slow
src = open('stream_record.py').read()
src = src.replace("keep((-1, -1, -1, -1, -1) if c is None else (c[0], c[1], c[5], c[6], c[7]))",
                  "keep((-1, -1, -1, -1, -1) if c is None else (c[0], 3 * c[1] + c[8] if SLOW > 1 else c[1], c[5], c[6], c[7]))")
src = src.replace("[(x, k, 'C1', 0.0, 0.0) for x in ln for k in range(3)]", "[(x, k, 'C1', 0.0, 0.0) for x in ln for k in range(3) for _ in range(SLOW)]")
src = src.replace("[j for j in range(len(ln)) for _ in range(3)]", "[j for j in range(len(ln)) for _ in range(3 * SLOW)]")
src = src.replace("(s[0], s[1], tag, -1, pos[t])", "(s[0], s[1] * SLOW + (t - 15) % SLOW if SLOW > 1 else s[1], tag, -1, pos[t])")
exec(compile(src.replace("if __name__ == '__main__':\n    main()", ""), 'slow', 'exec'))
main()
