───────────# VES-Modeling 下一阶段任务说明

## 0. 项目定位

你当前维护的项目：

`VES-Modeling`

定位固定为：

> **Verifier-first computational modeling engine**

它是数学建模系统中的“核心计算建模引擎”，不是完整数学建模工作流。

核心职责：

```text
Computational Modeling Task
        ↓
Generate Candidate
        ↓
Execute
        ↓
Independent Verify
        ↓
Judge
        ↓
Search / Improve
        ↓
Best Verified Solution
```

长期保持：

```text
VES-MathModeling-Skill
        ↓
VES-Modeling
        ↓
Verified-Executable-Search
```

VES-Modeling 不负责完整比赛流程。

---

# 1. 明确不做什么

不要把 VES-Modeling 扩展成以下系统：

- 数学建模题目 PDF 阅读器
- 自动读题 Agent
- 完整问题拆解器
- 文献检索系统
- Word / LaTeX 论文生成系统
- 科学绘图工作流
- 完整数学建模比赛 Agent
- 多 Agent orchestration 平台

这些属于上层 `VES-MathModeling-Skill`。

VES-Modeling 只关心：

> 一个已经比较明确的计算建模问题，怎样找到更好的、经过独立验证的可执行解。

---

# 2. 当前已有状态

目前 Regression vertical slice 已经跑通：

```text
train.csv / test_features.csv
        ↓
LLM / Mock CandidateGenerator
        ↓
solution.py
        ↓
Docker / Local Runner
        ↓
predictions.json
        ↓
Host RegressionVerifier
        ↓
Evidence(rmse, mae)
        ↓
Judge
        ↓
SearchEngine
        ↓
Improve
        ↓
Best Verified Candidate
```

已经存在的正确设计原则继续保持：

- Candidate 不能给自己评分；
- candidate 自报 metric 无效；
- hidden labels 不进入 prompt；
- hidden labels 不挂载进入真实 LLM candidate 容器；
- Judge 只消费 Host Evidence；
- 真实 LLM candidate 使用 Docker；
- 当前不要建立 Universal Modeling Framework；
- concrete first。

不要重写已经跑通的 Regression vertical slice。

---

# 3. 当前第一优先级：把 Regression slice 做成真正可复用的核心 API

目前 demo 已经能跑，但上层项目以后不应该调用：

```text
examples/regression_demo.py
```

或者直接操作内部：

```text
SearchEngine
Runner
Problem
Generator
```

需要给外部调用者提供一个简单、稳定的 Regression API。

建议设计类似：

```python
from ves_modeling.regression import run_regression_search
```

具体接口根据现有实现设计，不要求完全照此。

目标是让调用方能够提供：

```text
public data
hidden verification data
LLM config / generator
search config
workspace
```

得到：

```text
RegressionSearchResult
```

结果至少包含：

```text
status
best_solution
best_evidence
best_rmse
best_mae
run information
artifacts
rejected count
```

要求：

- 上层不需要理解 VES Core 内部细节；
- 不暴露没有必要的内部类型；
- 不为了 API 建 Manager / Factory / Registry；
- 先只设计 Regression concrete API。

---

# 4. 修复当前已经发现的小问题

## C1. 修复 B-005：运行日志真正落盘

当前 `RunResult` 中有：

```text
stdout
stderr
returncode
timed_out
```

但失败 candidate 的 stdout/stderr 没有可靠保存到运行产物中。

修改 Runner，使每次执行至少保留：

```text
run/
├── solution.py
├── stdout.log
├── stderr.log
└── predictions.json
```

Docker 与 Local Runner 行为尽量一致。

要求：

- 成功运行也保存日志；
- timeout 也保存；
- candidate Python exception 可以从 stderr.log 直接看到；
- 日志做合理大小限制；
- 不记录 API key 等 secret。

同步修正：

`docs/bug-log.md`

B-005 不应继续错误标记为由 `_prepare_run_dir` 修复。

---

## C2. 修复 hidden labels finite 校验

检查当前：

```python
np.isnan(...)
```

类逻辑。

hidden labels 的要求是：

```text
non-empty
numeric
finite
```

应该使用等价于：

```python
np.isfinite(labels).all()
```

的完整检查。

补测试：

```text
NaN
+Inf
-Inf
empty
```

---

## C3. 加强 RegressionVerificationContext invariant

保持简单，但至少保证明显非法状态无法构造。

考虑检查：

```text
labels non-empty
labels finite
expected_count > 0
expected_count 与 labels.size 的关系合理
```

不要为此重新设计 Context 框架。

---

## C4. CI

增加最小 GitHub Actions。

普通 CI：

```text
Python 3.11
install
ruff check .
pytest
```

Docker-heavy tests 如果不适合普通 CI：

```text
@pytest.mark.docker
```

独立运行。

不要为了 CI 引入复杂矩阵。

---

# 5. 运行记录与可观测性

真实 LLM 搜索以后会大量出现：

```text
candidate syntax error
dependency error
artifact missing
timeout
wrong artifact format
verification rejected
```

因此 VES-Modeling 应优先提高“失败可解释性”。

每个 candidate 最好能从 run 目录确定：

```text
生成了什么代码
是否成功执行
return code
stdout
stderr
artifact 是否存在
verification 是否通过
Evidence 是什么
为什么 rejected
```

但是不要构建大型 observability platform。

文件 + JSON 足够。

---

# 6. 运行目录策略

当前 `_prepare_run_dir` 会清理同名旧目录，以解决 runner 重复运行问题。

短期可以保留。

但是要检查：

> 完整一次 Search 的不同实验是否会意外覆盖历史运行证据。

如果确实存在问题，优先增加 experiment-level id：

```text
runs/
└── <search-id>/
    ├── draft0/
    ├── draft1/
    ├── improve0/
    └── ...
```

不要立刻做大型 RunManager。

只有现有调用确实需要时才修改。

---

# 7. 不要现在实现所有 Modeling Domain

当前不要一次增加：

```text
Classification
Time Series
Optimization
ODE
PDE
Simulation
Clustering
AHP
TOPSIS
Graph
...
```

下一 vertical slice 应由 `VES-MathModeling-Skill` 的真实使用需求驱动。

流程应当是：

```text
Regression
   ↓
真实 Workflow 使用
   ↓
发现高频 unsupported computational task
   ↓
选择第二个 vertical slice
```

而不是提前造算法大全。

---

# 8. 特别禁止提前建立以下东西

当前不要建立：

```text
UniversalModelingTask
BaseModelingSolver
ModelingTaskRegistry
UniversalMetric
UniversalVerifier
SolverFactory
SolverManager
PluginManager
DomainManager
TaskFactory
```

除非至少已经实现两个到三个真实 domain，并能明确证明存在重复逻辑。

原则：

> Duplication before abstraction.

先接受少量 domain-specific 重复。

---

# 9. 第二 vertical slice 的选择方法

不要现在决定。

等待上层 `VES-MathModeling-Skill` 跑真实题。

记录：

```text
unsupported_task_type
frequency
verification feasibility
candidate artifact type
objective structure
```

然后选择需求最明确的一类。

可能候选：

```text
Optimization
Time Series
Simulation
```

但必须由真实需求决定。

---

# 10. 对上层 Workflow 的接口原则

VES-Modeling 不理解：

```text
这是国赛第几问
这是美赛哪道题
论文如何组织
哪个图放正文
摘要写什么
```

只理解类似：

```text
task type
input data
artifact contract
verification context
objective
constraints
search config
```

第一版只需要 Regression。

---

# 11. 与 VES Core 的关系

继续遵守：

```text
VES-Modeling
      ↓
Verified-Executable-Search
```

VES Core 负责通用 primitive，例如：

```text
Artifact
VerificationContext
Evidence
Judge
SearchEngine
execution primitive
```

VES-Modeling 负责领域语义，例如：

```text
RegressionVerificationContext
RegressionVerifier
Regression candidate prompt
Regression artifact semantics
Regression search API
```

发现 Core gap 时：

1. 先记录；
2. 能薄 adapter 解决则优先 adapter；
3. 不直接复制一套 Core；
4. 真正通用且必要时再推动 Core 修改。

---

# 12. 测试重点

继续强化以下行为：

### Verifier

- prediction count mismatch
- invalid JSON
- missing predictions
- bool
- string
- NaN
- Infinity
- candidate fake metric
- hidden label isolation

### Runner

- success
- Python exception
- timeout
- no artifact
- repeated workspace
- output size limit
- Docker unavailable
- stdout/stderr persistence

### Search API

至少有：

```text
Mock generator
→ run_regression_search()
→ verified best result
```

不依赖真实外部 API。

---

# 13. 当前里程碑

## R7.1 — Cleanup

完成：

- B-005
- hidden labels finite
- Context basic invariant
- bug log 修正

---

## R7.2 — Stable Regression API

提供真正供外部项目调用的 Regression API。

增加：

```text
API unit test
integration test
README usage
```

---

## R7.3 — Workflow Integration Support

配合 `VES-MathModeling-Skill` 的 Regression adapter。

如果集成暴露出问题：

优先做最小修正。

不要为上层 Workflow 增加不属于核心的功能。

---

## R7.4 — Real Regression Subproblem

接受由上层 Workflow 准备好的真实数学建模 Regression 子任务。

只验证：

```text
Task
→ VES-Modeling
→ Search
→ Verified Result
```

无需独立读完整题。

---

## R7.5 — Decide Next Slice

根据真实 Workflow 缺口决定第二 vertical slice。

---

# 14. 成功标准

这一阶段成功不是：

> VES-Modeling 自动做完整数学建模比赛。

成功标准是：

1. Regression API 稳定；
2. 核心验证边界继续可靠；
3. LLM candidate 失败容易诊断；
4. 上层 Workflow 可以把 Regression 子任务干净地交给 VES；
5. VES 返回 verified result；
6. VES 完全不依赖完整数学建模 Skill 才能使用；
7. 没有为了未来制造大型万能抽象。

---

# 15. 一句话原则

以后所有开发决策先问：

> 这个功能属于“寻找并验证更好的可执行计算建模方案”，还是属于“完整数学建模工作流”？

如果是前者：

放 VES-Modeling。

如果是后者：

放 VES-MathModeling-Skill。

当前优先把 Regression 核心变得稳定、可调用、可诊断，然后等待真实 Workflow 推动下一 vertical slice。───
