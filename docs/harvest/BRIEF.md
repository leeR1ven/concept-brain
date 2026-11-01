# Harvest brief: what from our earlier brain research can improve the DESIGNER'S architecture (not replace it)

## Why this round exists
The designer asked many agents to study the brain and then *optimise his architecture*. Instead, the v3 round replaced it with a learning-theory core ("ARCH_v3_core"). His words (2026-11-01): "之前我让你找很多智能体了解大脑相关知识，然后优化我们的架构，结果你直接把整个架构换了，我想的是在我的架构基础上优化的更好。你们之前的讨论成果你觉得还有哪些可以拿来优化我们的架构？"

## The frame (binding)
`docs/ARCHITECTURE.md` is the designer's architecture and the single authority. Read ALL of it first, especially §二 (what is right), §三 (wrong rules: never bring these back) and §四 (open items, with the latest experiment evidence from X1c, X3, X4, X4b).
Binding designer decisions are also in `docs/IDEAS.md` #19–#24.

## What to extract
From your assigned documents, list every idea that would make an EXISTING part of the designer's architecture work better, i.e. a refinement of a part he already has:
- sensory areas, the concept (association) area, chain neurons (paired forward/return), the layered recurrent PFC, return chains (imagery / inner voice), the hippocampus (own population, chain-fed, fast Hebbian), the motor area (muscle vs inhibitory neurons, inhibition dominant), the one temporal-Hebbian rule (concave growth, ~1 % floor, two-layer weights, one global decay), reward as a tool (innate values only), split/delete, lineage/merging, whole-model scaling.
- Also ideas that would answer one of the open items in §四 (1–14), e.g. how the moving thought follows the same path for the same question, how the PFC stays stable via growth/decay parameters, the two correction mechanisms, adults learning new things via the hippocampus, the concept area's form.

## For each idea give (compact)
1. **Name** and the source (file + section).
2. **Which part** of ARCHITECTURE.md it refines (section number) or which §四 item it answers.
3. **What exactly** changes, in one or two sentences, in the designer's terms (neurons, connections, the one rule, parameters). If it needs a NEW module, a new rule, a timer, a symbolic structure, time cells, event boundaries, refractoriness/fatigue, an error-correction readout, a separate BG/value module, or anything in §三, say so and mark it REJECT.
4. **Evidence**: brain (cited), or our experiments (file + number). Say how strong.
5. **Verdict**: ADOPT (fits, low risk), TEST (fits, needs an experiment; name the smallest test), or REJECT (replaces or patches; say why).

Also list, separately and briefly, the ideas from your documents that *replaced* the designer's design (so we know what not to repeat).

## Rules
- Read only. Do not modify the repo or other folders. No runs needed (a tiny local check is fine: `free -m` first, ≤ 1 python process < 300 MB; other experiments share the box).
- Return your full answer inline in your final reply, in English, as a table or numbered list, ranked by expected value to the designer's architecture. Keep it under ~2,500 words. End with a 6–10 line `summary_zh` in plain Chinese.
