# Review round 2: ARCH_v2

Five independent reviewers answered the same macro questions about the whole ARCH_v2 architecture, each from a
different lens: neuroscience, learning theory, development, embodied / open world, and skeptic.
- Their brief is in `BRIEF.md`.
- The coordinator's synthesis and ranked plan are in **`SYNTHESIS.md`**.
- Each lens folder holds its report and probe scripts.

The scripts were run in the working layout, where `../../synth` is the ARCH_v2 folder. Here they point at `v2/synth/`.
Some of them load the saved agent `out/agent_final_s0.pkl` or the life log `out/life/*_log.pkl`. Those pickles are not
kept in the repo; rerun `v2/synth/run_j.py` / `life_synth.py` to regenerate them.

## 中文摘要
五位审查者的结论高度一致：ARCH_v2 有"一岁孩子"的基本配置，练过的题几乎全对。但它边听边学时不分"谁在回答谁"：自己说的话不进入想法和记忆，于是把"自己回答后对方说的下一句"当成答案来学。这一个原因造成了以下问题：
- 没人教时越学越偏；
- 主动说的话多是对方的台词；
- 大部分"没见过的组合答不出"。

两个局部修正都有明显效果：
- 符号测试中，没见过的组合 0 → 0.72–0.80；
- 完整测试台的一个种子上，综合分 0.70 → 0.93，无人教时的下降 0.43 → 0.03。

另外，几个展示性数字要打折：一生模式的自由回答不如"重复对方 / 按题型固定答"的简单办法；"最爱话题"其实是虚字。

下一步按顺序做：
1. 诚实测量；
2. 按角色学习；
3. 把听懂的说出来；
4. 有正负号的价值回路；
5. 学出来的词义层加先天随机连接；
6. 慢速巩固；
7. 适应开放世界。
