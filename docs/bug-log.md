# VES Modeling 测试 Bug 台账（待修复）

> 记录 R6 真实闭环测试中发现的问题；修复建议面向 VES Core（.vendor/Verified-Executable-Search，勿直接改）与 ves_modeling 自身。
> 状态：`recorded` = 已记录待修；`fixed` = 已在应用层绕行（Core 仍待修）。

## VES Core 上游处理规则

所有 VES Core 问题先在本文件登记编号、最小复现、影响范围、应用层绕行与建议修复，
不得直接修改本项目的 `.vendor/Verified-Executable-Search` 副本。

- **小型 Core bug**：改动局部、行为边界清楚、兼容风险低且能补充回归测试时，可以向
  VES Core 提交 PR；PR 链接和上游提交号回填本台账。
- **大型 Core 问题**：涉及公共协议、架构、持久化格式、跨模块行为或兼容风险不明确时，
  只向 VES Core 提 Issue，由维护团队设计和修复；本项目只保留必要的应用层绕行。
- 无论 PR 或 Issue，都不得包含 API key、隐藏标签、私有数据、内部团队目录或运行密钥。

## B-001（VES Core）SearchEngine 对非 legacy LLM 异常的 "reasoning-only" 重试失效

- **组件**：`ves/search_engine.py:192-208`（`_call_generator` / `_infrastructure_error_type`）
- **现象**：LLM 客户端返回空内容（reasoning-only）时抛 `RuntimeError`，`SearchEngine.search()` 直接崩溃，没有按设计重试 1 次。
- **根因**：`_infrastructure_error_type()` 在 aide_solver 已安装时返回 `GenerationInfrastructureError`；`_call_generator` 里 `if error_type is not None and not isinstance(error, error_type): raise` 会把一切非 legacy 类型异常直接重抛，"reasoning-only" 字符串检查永远到不了。
- **影响**：v0.1 的 `LlmClient` 是 `Protocol`（理应通用），但重试逻辑强耦合 aide_solver legacy 类型；任何第三方客户端遇到空内容都会崩溃。
- **应用层绕行（fixed）**：`ves_modeling/llm.py` 在客户端内部对 "reasoning-only" 与瞬时错误重试（max_attempts=6，退避 15s*(n+1)）。
- **建议修复**：`_call_generator` 改为对所有异常统一检查 `"reasoning-only" in str(error)` 后重试一次（移除 legacy 类型强耦合），或把 `LlmClient` 空内容定义为明确的协议错误。
- **状态**：✅ **Core 已修复**（VES Core main `485ec38d` "fix(search): provider-neutral reasoning-only retry (B-001)"，2026-08-11；应用层绕行保留，不影响）。

## B-002（ves_modeling / VES runner 同款）workspace 复用导致 FileExistsError，非幂等

- **组件**：`ves_modeling/regression/runner.py`（Local/Docker `mkdir(parents=True, exist_ok=False)`）；VES `aide_solver/execution.py` 同款行为。
- **现象**：同一 `workspace` 第二次跑 demo（相同 run_id：draft0/draft1/improve*），`runs/<run_id>/code` 已存在 → `FileExistsError`。
- **影响**：重跑必须手动清 `runs/`；真实多轮运行易踩。
- **绕行**：每次运行前清空 `runs/`（或换新 workspace）。
- **建议修复**：run 前对已存在的 `run_root` 做安全清理，或 run_id 加时间戳保证唯一。
- **状态**：✅ 已修复（`_prepare_run_dir` 在运行前清理陈旧 run_root，Local/Docker runner 均可复用 workspace）。

## B-003（外部服务，非 VES 代码）OpenCode Go 网关对长 max 推理不稳定

- **现象**：非流式请求在 ~100-130s 被网关断连（`RemoteProtocolError`），偶发 500/503；短请求正常（~20s）。
- **结论**：非 VES bug；`deepseek-v4-flash` + `reasoning_effort=max` + 100k 预算的长生成必须走 **streaming**（实测流式 433s 完成，finish=stop）。客户端已改为 SSE 流式 + 重试。
- **运维提示**：网关故障期间单次调用可能反复重试，R6 一次完整运行（2 draft + 3 improve）约需 30-50 分钟。

## B-004（运行观察，非 bug）draft0 被拒：LLM 候选未产出 predictions.json

- **现象**：R6 第一次运行 draft0 代码已生成（HistGradientBoosting 方案）且容器执行完毕，但 `/output/predictions.json` 缺失 → 宿主判 rejected。
- **分析**：候选代码可能运行异常/未写 artifact；SearchEngine 行为正确（runner 失败 → rejected → 继续下一候选）。若复现需看候选 stderr（run 记录未保留 stdout/stderr，见 B-005）。

## B-005（ves_modeling 可观测性）RunResult 未落盘 stdout/stderr，候选失败难排查

- **组件**：`ves_modeling/regression/runner.py` RunResult 有 stdout/stderr 字段，但 SearchEngine 只消费 succeeded/run_dir；demo 不打印失败候选的 stderr。
- **影响**：draft0 这类"运行了但没产出"的候选，无法直接从 run 产物看出失败原因。
- **建议修复**：runner 把 stdout/stderr 写入 `run_dir/run.log`（或 demo 对 rejected 候选打印 stderr 前 2000 字符）。
- **状态**：✅ 已修复（Local/Docker runner 每次运行把完整 stdout/stderr 落盘到 run 目录
  `stdout.log` / `stderr.log`；成功、失败、timeout 均保存；`_prepare_run_dir` 清理属 B-002 修复，与本条无关）。

## B-006（VES Core）SearchResult 丢失被拒候选的结构化诊断与运行映射

- **分类**：大型 Core 公共协议/架构问题，按本台账规则提交 Issue，不在 Modeling 的
  `.vendor` 中直接修改。
- **组件**：VES Core main `485ec38d`，`ves/search_engine.py:41-50,72-85,141-176,198-263`。
- **最小复现**：让 runner 分别返回执行失败、成功但 artifact 缺失/非法，或让 verifier
  返回拒绝。`SearchEngine._run_and_verify()` 对这些分支统一返回 `None`；最终
  `SearchResult` 只增加 `rejected`，`records` 只包含 verified 候选。
- **影响**：调用方无法从 Core 公共结果判断某次尝试属于 generation/execution、timeout、
  artifact_missing、artifact_invalid 或 verification_failed，也无法可靠关联 attempt id、
  run directory、stdout/stderr、artifact hash 与拒绝原因。工作流只能复制 SearchEngine
  控制流或依赖 runner 私有状态，造成公共协议漂移和审计链断裂。
- **应用层绕行（fixed）**：R7.3 使用 `ClassifyingSearchEngine` 在 Modeling 层保留每次
  attempt，写入 `candidates/<attempt>/run.json`，并在 summary 中返回 candidate status。
- **建议 Core 方向**：为每次 attempt 定义稳定的 outcome/diagnostic 类型和状态枚举；
  `SearchResult` 同时返回 verified records 与全部 attempt outcomes，并保留 run identity、
  可公开错误摘要和 artifact/record 关联。需先设计兼容与序列化协议，不建议局部打补丁。
- **状态**：🟢 已提交 VES Core
  [Issue #2](https://github.com/Zhuchen00123/Verified-Executable-Search/issues/2)，
  等待维护团队设计公共 outcome/diagnostic 协议。

## 注意事项（非 bug）
- `Observation.value` 允许 NaN（只校验 uncertainty）；宿主 verifier 必须保证有限，`JudgeSpec` 可加 `Gate(finite)` 兜底——Modeling 已做。
- `ArtifactContract.numeric_fields` 只接受单个有限数字，数组元素/长度语义由领域 verifier 承担（设计边界，与 idea.md §9 一致）。
