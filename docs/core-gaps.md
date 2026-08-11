# VES Core 差异记录（Rule 1 / §39）

## Gap 1：Docker 运行契约输出目录固定为 /workspace

- Observed problem：`DockerProcessRunner` 把 candidate 输出目录挂在 `/workspace`（workdir=/workspace），而 Modeling 候选契约是 `/output/predictions.json`（idea.md §8）；且 `DockerProcessRunner` 位于 `aide_solver`（source-checkout only，不是 wheel 公共 API），Modeling 无法稳定 import。
- Current workaround：Modeling 在 `ves_modeling/regression/runner.py` 实现薄 `DockerRegressionRunner`（CodeRunner adapter）：沿用同一套安全参数（network none、read-only、cap-drop ALL、no-new-privileges、非 root、memory/cpu/pids/tmpfs 限制、超时、清理、public_files 逐文件挂载），仅把输出目录挂到 `/output`。
- Why application-layer solution is insufficient：`aide_solver` 不在 `ves` 公共 facade（`ves/__init__.py` 未导出 runner）；不修改 Core 的前提下必须自建薄 adapter。
- Affected modeling domains：Regression（当前）；Optimization/Mechanism 若也用 `/output/<artifact>` 会复用同一 adapter。
- Proposed Core primitive（v0.2 候选，**暂不实施**）：把 DockerProcessRunner 提为 `ves.runner` 公共 API，并允许配置输出挂载点（`output_mount: str = "/workspace"`）与 public_files 默认值。
- Backward compatibility impact：新增参数默认 `/workspace` 即完全兼容现有行为。
