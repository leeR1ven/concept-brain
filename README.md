# concept-brain

模仿真实大脑、边运行边学的 AI 模型（研究原型）。设计见 [DESIGN.md](DESIGN.md)。

```bash
pip install numpy
python tests/test_parts.py sense     # 听觉区单独验证：听得清 / 记住连续的声音（约 2 分钟）
```

- `brain/net.py`   二值神经元的区和连接（学习：先亮 → 后亮；整体降权）
- `brain/net3d.py` 感觉区的立体局部网络（每层亮的比例固定）
- `brain/areas.py` 感觉区 / 前额叶·海马体 / 运动区 各部分，和把它们按同一拍串起来的 Circuit
- `world/voice.py` 声音世界（每个字一个声音，自己 / 别人略有差异；嘴部肌肉）
