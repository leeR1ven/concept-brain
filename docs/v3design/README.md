# Design round v3: a fixed set of general mechanisms

The designer's request (2026-11-01): "最好想个办法不用为了让模型实现某项能力而不停的加机制，而是固定数量的机制就能学会任何事情。"
In English: find a way to stop adding a mechanism for every new ability, so that a fixed number of mechanisms can learn
anything.

Five members answered the same question from different angles: audit, neuroscience, learning theory, development and
evaluation. They worked from one shared brief, `BRIEF.md`. This round is design only, with no training. Each member's
folder holds its report and its small analytic scripts.
- **`audit/`:** an inventory of all 83 mechanisms and 265 knobs of today's agent, each classed as generic, learnable,
  innate or scaffolding.
- **`neuro/`:** the brain's repeated toolkit.
- **`learning/`:** what "learn anything" can mean, local credit assignment, the cost of depth, and the global scale
  factor.
- **`develop/`:** innate drives, the world and a curriculum.
- **`eval/`:** a sealed test that cannot be passed by designing on it.

**The synthesis is `synth/ARCH_v3_core.md`:** one fixed core of 4 mechanisms, 2 broadcasts, 2 objectives and 5 innate priors (17 sub-items), a 30-gene genome, lineage inheritance, the maturity rule, a neuron-level form for every part, a 13-step migration ladder from v2.2 and the first five experiments. **Adversarial review (`review/`):** four reviewers (development, ML, neuroscience, skeptic engineer) answered one question, "is it sound and buildable?", from their backgrounds, with small toy checks. All four say "fix, then build" and name 5 FATAL-as-written problems: babbling and speech release that cannot reach zero errors, replay that collides with the held context, per-tick prediction that collapses, and a world seed per generation that invalidates inheritance. `review/REVISION_BRIEF.md` lists them together with the designer's own specification, which binds revision 2. Paths of the form `arch/v2/...` refer to
the coordinator's working tree; the published parts live in `v2/` (v2.2 is `v2/v22/`).

## 中文摘要
五位成员从不同角度回答同一个问题：怎样用一套固定的机制学会新本事，而不是每种能力加一个机制。

- **盘点：** 现在的模型有 83 个机制、265 个手调参数。ARCH_v2 之后新加的 21 个机制里，只有 3 个是通用的；很多时间常数是照着测试世界的节奏定的。
- **大脑：** 大脑用的是一小套反复使用的零件：
  - 同一种皮层回路；
  - 基底节"选择并放行"（说话和思考都由它选）；
  - 海马体一次记住加回放；
  - 几种全脑广播信号；
  - 一族可塑性规则。
  - 再加少数先天部分：连线蓝图、脑干节律、朝向人脸和声音、驱力和社交奖励。
- **学习理论：** "学会任何事"字面上做不到，能做到的是"机制数量不随能力增加"。最大的风险是：不用反向传播，在一生连续学习中能否学到抽象的中间表示。
- **发育：** 大人纠正孩子却学不进去，主要是世界的问题。现在的世界从不示范"问题后面紧跟答案"（自问自答、旁听别人问答、把问答重说一遍）。只要世界有这些，一个不加任何角色规则的学习者就能学会，而且不忘。
- **综合方案（`synth/ARCH_v3_core.md`）：** 4 个通用机制（各脑区同一种电路、一条三因子学习规则、海马一次记忆加回放、基底节闸门）、2 个广播信号、5 项先天部分；关键新结构是前额叶记"最近一件事是谁说的、过去了多久"；30 个基因，265 个手调参数删掉 189 个；同一血统长大后融合生成后代；最终成品为纯神经元；从 v2.2 出发分 13 步迁移。
- **测试：** 先冻结代码，再出题；新题型连设计者都没见过，只开封一次；模型只能看到感官输入；放大必须是整个模型按同一个倍数放大。
- **审查（`review/`）：** 四位审查者（发育、机器学习、神经科学、工程）都认为"先修再建"，指出 5 个按原文会失败的问题：咿呀和说话闸门做不到长大后 0 错误、回放和手里的问题冲突、每一拍都学"预测下一刻"会崩、每一代换世界种子会让继承作废；还有很多尺寸和计算量估计偏乐观。修改要求连同设计者原来的设计说明一起写在 `review/REVISION_BRIEF.md`。
- **设计者机制核对表（`DESIGNER_SPEC.md`）：** 逐条核对设计者说过的 40 条机制与现在的架构。被丢掉的最重要几条是：学习规则本身（时序赫布、凹形增长、统一整体降权、两层权重）、链式神经元和回程、文字网络式的前额叶、可变激活个数、运动区"抑制默认占上风"。这张表是 v3 第二版的硬性要求。
- **第二版 r2b（`synth/ARCH_v3_core_r2.md`）：** 完全以设计者的整体架构为准（立体局部随机感觉区、链式神经元、文字网络式前额叶、设计者的海马体、回程、肌肉对抗抑制神经元的运动区），每个部件都标明出处：设计者的 / 设计者的但定了参数 / 新加的。学习只有一条时序赫布规则，奖励只是工具。小实验：设计者的规则自然做到"最近一件事"（3,000 段后 1.00）；字面上"还亮几个表示过了多久"在真实大人行为下会崩（0.01），"疲劳快慢不一轮流占上风"是 0.75（唯一新加的东西，待设计者决定）；奖励要有正负，否则空房间里打招呼会越来越多。
