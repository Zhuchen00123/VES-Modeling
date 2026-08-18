# 搜索策略（Search Strategy）· 固定版
> **状态**：已固定（2026-08-18）。
> **来源**：R3 实验结论与多次实证。
> **位置**：docs/search-strategy.md
---
## 1. 核心结论
1. 长单次 run（long-single）是最稳形态：方差小、均值高。
2. 搜索应默认使用 long-single，特殊场景才考虑其他形态。
## 2. 默认搜索策略（写死）
1. 首选：long-single（长单次 run）作为主搜索形态。
2. 候选数默认 N；结果以 Judge 判定为准，不轻信 LLM 原始数值。
3. 仅当 long-single 不适用（如需分簇对比）时才切换其他形态。

## 3. 判据与可靠性
1. 最终结果以 Judge 判定为准，LLM 数值可发散（见过 rmse 2e22），须防伪。
2. 拒绝把单个数字当结论；至少 3 次 repeat 看分布，报 mean + 方差。
3. short run 方差大，单次失败不能当"更差"的证据。

## 4. 固定用法（checklist）
1. 先确认要搜索的方法簇。
2. 用 long-single 跑，每臂 >= 3 repeat。
3. 收集 runs/<x>/report.json 归档为 report-<arm>-r3.json。
4. 用 Judge 判 final，写进实验报告。
5. 不跑新重型实验；仅做简单验证实验（供应商改用新 key，见群聊）。
