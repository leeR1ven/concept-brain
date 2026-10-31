# concept-brain

**中文** | [English](README.en.md)

模仿真实大脑、边运行边学的 AI 模型（研究原型）。设计见 [DESIGN.md](DESIGN.md)。

```bash
pip install numpy
python tests/test_parts.py sense     # 听觉区单独验证：听得清 / 记住连续的声音（约 2 分钟）
```

- `brain/net.py`   二值神经元的区和连接（学习：先亮 → 后亮；整体降权）
- `brain/net3d.py` 感觉区的立体局部网络（每层亮的比例固定）
- `brain/areas.py` 感觉区 / 前额叶·海马体 / 运动区 各部分，和把它们按同一拍串起来的 Circuit
- `world/voice.py` 声音世界（每个字一个声音，自己 / 别人略有差异；嘴部肌肉）

## v2（进行中）
新一代架构的整体审查、共享测试台、三个独立原型和合成后的统一架构 ARCH_v2 在 [v2/](v2/README.md)
（统一架构综合分 0.42→0.83，在新种子上 0.82；详见 [v2/synth/ARCH_v2.md](v2/synth/ARCH_v2.md)）；五份审查报告和共识在 [docs/review/](docs/review/SYNTHESIS.md)。
设计者的核心思想及验证情况见 [docs/IDEAS.md](docs/IDEAS.md)。

许可证：MIT（见 [LICENSE](LICENSE)）。
