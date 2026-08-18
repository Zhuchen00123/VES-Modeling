# 实验总结（Experiment Summary）
> 状态：已结合结论总结（2026-08-18）。
> 范围：R0–R7.3 及 R3 三臂对照实验结论。
> 位置：docs/experiment-summary.md

## 1. R0–R7.3 里程碑结论（HANDOFF 已记）
- 完成 apply API、稳定 JSON 协议、provenance、统一 run tree、结构化失败、目标/任务类型建模。
- POC T-032 结论："问题类型 ≠ 方法类型"，批准按方法簇搜索（由此引出 R3）。

## 2. R3 三臂对照结论（forecasting 分簇 vs 单空间）
- 真实 LLM 单次：单空间 44.53 反超分簇 49.18，但结论受单次噪声影响。
- 多 repeat 后：单空间（long-single）稳定最优，建议作为默认策略。
- LLM 数值会发散（rmse 2e22），依赖 Judge 收敛。
