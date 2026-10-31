# -*- coding: utf-8 -*-
"""R1 check: my batched RWKV-7 vs the official `rwkv` pip package; Triton vs torch loop; speed; tokenisation."""
import os, sys, time, glob
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_CUDA_ON'] = '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                       # noqa: E402
from rwkv7 import RWKV7, HAVE_TRITON               # noqa: E402
import rwkv7                                       # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.3)
print('triton', HAVE_TRITON, getattr(rwkv7, 'TRITON_ERR', ''))
for path in sys.argv[1:]:
    print('==', path)
    from rwkv.model import RWKV as REF
    from rwkv.utils import PIPELINE
    ref = REF(model=path[:-4], strategy='cuda fp32')
    pipe = PIPELINE(ref, 'rwkv_vocab_v20230424')
    text = '我的小狗叫豆米。今天天气很好，我们去公园玩了。\n我的小狗叫'
    ids = pipe.encode(text)
    print('tokens', [pipe.decode([i]) for i in ids])
    out_ref, _ = ref.forward(ids, None)
    out_ref = out_ref.float().cuda()
    m = RWKV7(path)
    print('L C H N', m.L, m.C, m.H, m.N)
    for tri in (False, True):
        m.use_triton = tri
        st = m.zero_state(1)
        lg = m.forward(torch.tensor([ids]).cuda(), st)
        print('triton' if tri else 'loop', 'max|diff| vs official', float((lg[0, -1] - out_ref).abs().max()),
              'argmax', pipe.decode([int(lg[0, -1].argmax())]), pipe.decode([int(out_ref.argmax())]))
        st2 = m.zero_state(1)
        m.forward(torch.tensor([ids[:7]]).cuda(), st2)
        b = m.forward(torch.tensor([ids[7:]]).cuda(), st2)
        print('  chunked vs whole', float((b[0, -1] - lg[0, -1]).abs().max()))
    # in-context recall sanity at short delay, two query styles
    for q in ['我的小狗叫豆米。\n问：我的小狗叫什么名字？\n答：你的小狗叫', '我的小狗叫豆米。\n今天天气很好。\n我的小狗叫',
              'User: 我的小狗叫豆米。\n\nAssistant: 好的。\n\nUser: 我的小狗叫什么名字？\n\nAssistant: 你的小狗叫']:
        st = m.zero_state(1)
        x = pipe.encode(q)
        out = []
        for _ in range(4):
            lg = m.forward(torch.tensor([x]).cuda(), st)
            t = int(lg[0, -1].argmax()); out.append(t); x = [t]
        print('  recall', repr(q[-12:]), '->', repr(pipe.decode(out)))
    for nm in ['豆米', '琥茉', '洛琪', '栗蔻']:
        print('  name tokens', nm, pipe.encode(nm))
    # speed: 4096-token chunk, batch 1 and 8
    for B in (1, 8):
        st = m.zero_state(B)
        x = torch.randint(100, 60000, (B, 4096)).cuda()
        torch.cuda.synchronize(); t0 = time.time()
        m.forward(x, st, need_logits=False)
        torch.cuda.synchronize()
        print(f'  speed B={B}: {B * 4096 / (time.time() - t0):.0f} tokens/s')
    del m, ref
    torch.cuda.empty_cache()
print('mem GB', torch.cuda.max_memory_allocated() / 2 ** 30)
