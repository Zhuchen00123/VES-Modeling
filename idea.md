可以。下面这份可以**整段复制到新项目文件夹里的 Agent 新会话**。

我特意按你当前 Core 的真实接口来写：`SearchEngine` 只依赖 `VerifiedProblem + CandidateGenerator + CodeRunner + AnchorPolicy`；`VerifiedProblem` 负责 contract/context/verifier/judge；当前 `ArtifactContract` 可以验证 JSON object 的基本结构，而 predictions 数组长度、有限数值等领域规则应由 Regression Verifier 做。  

直接复制下面这一整块即可：

# VES Modeling 项目启动与实现总指令

你现在负责从零创建 **VES Modeling**。

这不是一个新的 Agent Framework，也不是重新实现 VES Core。你的任务是基于现有 **Verified Executable Search (VES)** Core，构建它的第一个真实应用层，并通过真实计算建模任务验证 VES 的设计是否成立。

请直接执行，不要只给我规划或架构建议。

如果环境允许你读取文件、执行 shell、运行测试、调用 Git、Docker 等工具，请主动使用这些能力完成工作。

除非遇到无法继续的外部阻塞，否则不要因为小问题停下来询问我；优先自行检查代码、文档、测试和环境，并作出保守、可逆的工程决策。

---

# 0. 项目背景

VES Core：

https://github.com/Zhuchen00123/Verified-Executable-Search

项目名称：

**Verified Executable Search (VES)**

核心原则：

> AI can propose. It cannot grade itself.

中文：

> AI 可以提出方案，但不能给自己打分。

VES 是一个 verifier-first runtime。

核心信任模型：

```text
Generator
   ↓
Candidate                  untrusted
   ↓
Runner
   ↓
Raw Artifact               untrusted
   ↓
──────── Host Trust Boundary ────────
   ↓
ArtifactContract
   ↓
EvidenceVerifier
   ↑
VerificationContext        host-owned truth
   ↓
Evidence                   verified facts
   ↓
Judge
   ├ Gates
   └ Objectives
   ↓
Judgment
   ↓
SearchEngine
```

最重要的原则是：

```text
Creation Authority != Judgment Authority
```

Candidate 可以生成方案。

Candidate 不能决定：

- 自己是否正确；
- 自己是否可行；
- 自己得了多少分；
- 自己是否应该成为 best candidate。

所有这些事实必须由 Host 独立验证。

---

# 1. 你的第一任务：先真正理解 VES Core

在写 VES Modeling 代码之前，必须检查现有 VES Core。

不要仅根据本提示词推测 API。

以实际代码为 Source of Truth。

如果当前项目目录中没有 VES Core：

优先检查：

```bash
pwd
find .. -maxdepth 2 -type d -name "Verified-Executable-Search" 2>/dev/null
```

如果附近没有，则克隆：

```bash
mkdir -p .vendor
git clone --branch v0.1.0 --depth 1 \
  https://github.com/Zhuchen00123/Verified-Executable-Search.git \
  .vendor/Verified-Executable-Search
```

如果需要阅读最新文档，可以另外读取 GitHub `main`。

**代码集成基准优先使用 v0.1.0。**

不要修改 `.vendor/Verified-Executable-Search`。

把它视为外部依赖。

---

# 2. 必须阅读的 VES 文件

至少检查：

```text
README.md

ves/__init__.py
ves/artifact.py
ves/context.py
ves/evidence.py
ves/verifier.py
ves/judge.py
ves/problem.py
ves/record.py
ves/search.py
ves/search_engine.py

examples/search_demo.py

docs/writing-a-verifier.md
docs/ves-search.md
docs/ves-cli.md
SECURITY.md
```

同时检查：

```text
aide_solver/execution.py
aide_solver/llm_client.py
```

了解现有 Docker Runner 和 OpenAI-compatible client 的实现。

但注意：

`aide_solver` 属于 VES 当前 source-checkout adapter / legacy implementation，不应成为 VES Modeling 长期公共 API。

可以参考或临时复用，但不要让 VES Modeling 的领域设计依赖 aide_solver 的内部抽象。

---

# 3. 阅读 Core 后输出理解文档

创建：

```text
docs/ves-core-understanding.md
```

至少写清楚：

## Verification

```text
RawArtifact
ArtifactContract
SafeArtifactLoader
VerificationContext
Evidence
Observation
EvidenceVerifier
VerificationResult
```

每一个对象：

- 谁创建它；
- 是否可信；
- 生命周期；
- Modeling 会如何使用。

## Judgment

解释：

```text
Evidence != Judgment
```

为什么：

```text
rmse=0.31
runtime=1.8
```

都可以是 Evidence，但 Judge 可以只优化：

```text
rmse
```

而不自动优化 runtime。

## Search

确认实际签名：

```text
CandidateGenerator
CodeRunner
RunResult
SearchEngine
AnchorPolicy
```

确认：

```text
draft()
improve()
runner.run()
SearchEngine.search()
```

实际要求。

## Security

确认：

```text
DockerProcessRunner
LocalProcessRunner
SafeArtifactLoader
hidden truth
network isolation
filesystem isolation
```

各自的安全边界。

---

# 4. VES Modeling 的定位

VES Modeling 是：

> A verifier-first single-file search system for computational modeling.

第一版目标：

> 给定一个明确的计算建模任务，让 AI 反复修改一个 `solution.py`，但每一次成绩都由 Host 独立验证。

VES Modeling 不应该重新实现：

```text
Evidence
Judge
VerificationPipeline
SearchEngine
Candidate lineage
VerificationRecord
Artifact safety
```

这些属于 VES Core。

VES Modeling 负责：

```text
Dataset conventions
Modeling problem definitions
Domain Verifiers
Prompt construction
Modeling-oriented CandidateGenerator
Runner integration
Session / reporting
Modeling examples
```

关系：

```text
VES Core
   ↑
VES Modeling
   ↑
Regression / Optimization / Mechanism Fitting
```

---

# 5. 强制工程原则

以下规则必须遵守。

## Rule 1

**不要修改 VES Core。**

如果发现 Core 缺少功能：

首先在 Modeling application layer 解决。

记录：

```text
docs/core-gaps.md
```

只有 correctness/security bug 才可以立即建议 Core 修改。

普通抽象问题至少等两个不同 Modeling 场景都出现同一问题以后，再考虑上移 Core。

---

## Rule 2

**不要一开始设计 Universal Modeling Framework。**

第一阶段只实现：

```text
Tabular Regression
```

先完成一个真实 vertical slice。

不要先创建大量：

```text
BaseTask
AbstractModelingTask
UniversalMetric
UniversalDataset
UniversalSolver
PluginManager
TaskRegistry
AgentOrchestrator
```

之类的抽象。

原则：

> Concrete first, abstraction second.

---

## Rule 3

Candidate 必须是单文件：

```text
solution.py
```

第一版坚持 AIDE-style single-file iterative search。

---

## Rule 4

Candidate 不能访问 hidden evaluation truth。

例如：

```text
candidate can see:

train.csv
test_features.csv

candidate cannot see:

hidden_test_labels.csv
```

Host Verifier 可以访问 hidden labels。

---

## Rule 5

Candidate 自报指标永远不能作为 Evidence。

例如 candidate 输出：

```text
RMSE = 0.000001
```

必须被忽略。

真正的：

```text
RMSE
MAE
constraint violation
objective
```

必须由 Host Verifier 独立复算。

---

## Rule 6

真正的 LLM-generated code 不允许使用 LocalProcessRunner 当安全边界。

Local runner 只能用于：

```text
trusted test fixture
hand-written candidate
unit test
```

真正 AI-generated executable candidate：

必须通过 Docker sandbox 或安全等级不低于现有 VES DockerProcessRunner 的执行边界。

---

# 6. 第一阶段任务：Regression Vertical Slice

第一版只做：

# Tabular Regression

目标完整链：

```text
train.csv
test_features.csv
hidden_test_labels.csv
        │
        ↓
Modeling Prompt
        ↓
Generator
        ↓
solution.py
        ↓
Runner
        ↓
predictions.json
        ↓
VES ArtifactContract
        ↓
RegressionVerifier
        ↓
Evidence
   rmse
   mae
        ↓
Judge
   minimize rmse
        ↓
SearchEngine
        ↓
improve solution.py
        ↓
BEST VERIFIED MODEL
```

---

# 7. 数据集

不要第一版依赖 Kaggle 或网络数据。

创建：

```text
scripts/generate_regression_data.py
```

使用：

```python
sklearn.datasets.make_regression
```

固定：

```python
random_state
```

保证测试可复现。

建议：

```text
1000~3000 samples
10~20 features
适量 noise
```

划分：

```text
train
hidden test
```

目录建议：

```text
data/regression/public/train.csv
data/regression/public/test_features.csv

data/regression/host/hidden_test_labels.csv
```

注意：

`host/` 绝不能挂载到 candidate container。

train.csv：

```text
feature_0
feature_1
...
feature_n
target
```

test_features.csv：

```text
feature_0
feature_1
...
feature_n
```

hidden_test_labels.csv：

```text
target
```

---

# 8. Candidate Contract

Candidate 接口必须非常简单。

Candidate：

```text
solution.py
```

允许读取：

```text
/data/train.csv
/data/test_features.csv
```

必须输出：

```text
/output/predictions.json
```

建议格式：

```json
{
  "predictions": [
    12.34,
    15.67,
    9.81
  ]
}
```

Candidate 可以额外输出：

```json
{
  "predictions": [...],
  "claimed_rmse": 0.000001
}
```

但：

```text
claimed_rmse
```

永远不能用于 Judge。

它只用于 adversarial demo。

---

# 9. ArtifactContract

创建适合：

```text
predictions.json
```

的 VES ArtifactContract。

Contract 负责：

```text
filename
media_type
size
JSON root
required predictions field
```

不要强行让 ArtifactContract 承担所有领域验证。

例如：

```text
predictions 是不是 list
长度是否正确
每个 prediction 是否 finite
是否和 hidden labels 对齐
```

应该由：

```text
RegressionVerifier
```

检查。

这就是：

```text
generic artifact validity
vs
domain semantic validity
```

的边界。

---

# 10. Regression VerificationContext

实现 Host-owned context。

例如：

```text
RegressionVerificationContext
```

它应该包含验证需要的可信信息，例如：

```text
hidden labels
expected prediction count
dataset identity / fingerprint
```

但是：

- 不暴露给 Candidate；
- 不写入 candidate workspace；
- 不挂载进 candidate Docker；
- 不进入 prompt。

确保 context fingerprint / record 能支持 replay。

按照 VES 当前 `VerificationContext` API 实现，不要自行另造一套 context 系统。

---

# 11. RegressionVerifier

实现：

```text
RegressionVerifier
```

职责：

1. 解析 predictions.json；
2. 确认 predictions 是数组；
3. 确认数量和 hidden labels 完全一致；
4. 拒绝 bool；
5. 拒绝 NaN；
6. 拒绝 Infinity；
7. 拒绝非数字；
8. Host 独立计算：
   - RMSE
   - MAE
9. 返回 VES Evidence。

目标 Evidence 类似：

```python
Evidence(
    observations=(
        Observation(
            name="rmse",
            value=...,
            provenance="host:hidden-test",
        ),
        Observation(
            name="mae",
            value=...,
            provenance="host:hidden-test",
        ),
    )
)
```

不要相信：

```text
claimed_rmse
claimed_mae
score
objective
```

Candidate 里的任何自报指标。

---

# 12. JudgeSpec

第一版保持最简单。

Objective：

```text
MINIMIZE rmse
```

MAE 只作为 Evidence。

不要一开始做：

```text
RMSE
MAE
runtime
complexity
memory
model size
robustness
```

多目标搜索。

第一版核心目标是证明：

```text
verified modeling search works
```

不是建立最终 Modeling benchmark。

---

# 13. Phase A：先不用 LLM

首先创建至少三个可信手写 candidate：

```text
fixtures/candidates/linear_regression.py
fixtures/candidates/random_forest.py
fixtures/candidates/gradient_boosting.py
```

要求它们全部遵守：

```text
/data
/output
predictions.json
```

协议。

先验证：

```text
candidate
↓
runner
↓
artifact
↓
VES VerificationPipeline
↓
RegressionVerifier
↓
Evidence
↓
Judge
```

完整成立。

此阶段不引入 LLM 不确定性。

---

# 14. Phase A 验收标准

必须存在自动测试。

至少：

## Test 1

正确 predictions 可以：

```text
VERIFIED
```

## Test 2

缺少 predictions：

```text
INVALID / VERIFICATION_FAILED
```

## Test 3

prediction 数量错误：

```text
REJECT
```

## Test 4

NaN / Infinity：

```text
REJECT
```

## Test 5

candidate 写：

```json
{
  "claimed_rmse": 0.000001
}
```

真实 predictions 很差。

必须证明：

```text
Host RMSE != claimed RMSE
```

并且 Judge 使用 Host RMSE。

## Test 6

hidden labels 不存在于 candidate workspace。

## Test 7

RegressionVerifier 结果可重复。

---

# 15. Phase B：Mock CandidateGenerator

实现一个：

```text
MockRegressionGenerator
```

符合 VES：

```text
CandidateGenerator
```

协议。

它必须实现：

```python
draft(problem, index)
improve(problem, anchor)
```

可以按顺序返回预先准备的 candidate code。

目标是让真实 VES：

```text
SearchEngine
```

跑起来。

不要自己实现新的 search loop。

---

# 16. Mock Search 目标输出

最终至少可以执行：

```bash
python examples/regression_demo.py --mock
```

得到类似：

```text
VES Modeling — Regression

Draft #1
  model: LinearRegression
  verification: VERIFIED
  rmse: 63.421
  mae: 50.118

Draft #2
  model: RandomForest
  verification: VERIFIED
  rmse: 42.721
  mae: 33.012

Improve #1
  model: GradientBoosting
  verification: VERIFIED
  rmse: 25.391
  mae: 18.842

BEST VERIFIED
  rmse: 25.391
```

具体数值不要求一致。

要求：

```text
SearchEngine
```

真实参与选择。

不能伪造：

```text
33 → 40 → 51
```

字符串。

---

# 17. Phase B 必须验证 Candidate Lineage

检查：

```text
draft
improve
parent/root lineage
VerificationRecord
candidate_id
candidate hash
```

正确存在。

不要在 Modeling 自己发明：

```text
ModelingCandidate
ModelingRecord
```

来复制 VES Core 已经有的东西。

---

# 18. Phase C：Adversarial Modeling Demo

实现专门的：

```text
cheating_candidate.py
```

它故意：

```python
print("RMSE = 0.000001")
```

或者 artifact：

```json
{
  "predictions": [...bad predictions...],
  "claimed_rmse": 0.000001
}
```

运行后必须展示：

```text
Candidate claimed RMSE:
0.000001

VES ignored candidate claim.

Host verified RMSE:
<真实较差值>
```

这是 VES Modeling 最重要的传播 Demo 之一。

确保这是**真实验证逻辑**，不是打印预设字符串。

---

# 19. Phase D：真实 LLM Generator

在 Mock Search 完全通过后，再实现：

```text
LLMRegressionGenerator
```

不要提前做。

它应该符合：

```text
CandidateGenerator
```

协议。

Provider 设计保持中立。

建议环境变量：

```text
VES_MODELING_LLM_BASE_URL
VES_MODELING_LLM_API_KEY
VES_MODELING_LLM_MODEL
```

如果现有 VES OpenAI-compatible client 可以安全、干净地复用，可以写 adapter。

如果只能依赖：

```text
aide_solver
```

内部实现，则：

1. 明确标注 temporary adapter；
2. 不让 aide_solver 类型进入 VES Modeling public API；
3. 后续可以替换。

不要把：

```text
OpenAI
DeepSeek
Claude
Kimi
```

硬编码到 Modeling Core。

---

# 20. Draft Prompt

真实 Draft Prompt 至少告诉模型：

```text
You are solving a tabular regression task.

Available files:

/data/train.csv
/data/test_features.csv

train.csv contains the target column:
target

You must create one complete Python program.

The program must:

1. Load /data/train.csv.
2. Load /data/test_features.csv.
3. Train a regression model.
4. Predict every row in test_features.csv.
5. Write exactly one artifact:
   /output/predictions.json

Artifact format:

{
  "predictions": [...]
}

You cannot access hidden evaluation labels.

Do not report or rely on a self-computed test score.
The host verifier will independently evaluate the predictions.

Your response must contain the complete solution.py only.
```

可以告诉模型允许库：

```text
numpy
pandas
scikit-learn
```

第一版不要让模型：

```text
pip install
curl
wget
联网
```

---

# 21. Improve Prompt

Improve 必须把：

```text
previous candidate code
+
Host-verified Evidence
```

提供给模型。

例如：

```text
Previous candidate:

<solution.py>

Host-verified evidence:

RMSE: 42.71
MAE: 33.01

This evidence was independently recomputed by the host.

Improve the executable solution.

You may change:

- preprocessing
- feature engineering
- estimator
- hyperparameters
- ensembling

You must still write:

/output/predictions.json

Return the complete solution.py only.
```

关键：

> 模型改进依据必须来自 VES Evidence。

不要把 candidate 自己的 stdout score 当 feedback。

---

# 22. Phase E：安全 Docker Runner

真实 LLM candidate 必须在 Docker 中运行。

首先研究 VES：

```text
aide_solver/execution.py
SECURITY.md
tests/test_docker_runner.py
tests/test_ves_security_docker.py
```

优先复用现有安全设计。

如果无法作为稳定 package API 复用，则在 VES Modeling 创建一个**薄 adapter**实现：

```text
CodeRunner
```

不要创建新的 execution framework。

Runner 至少满足：

```text
network disabled
read-only root filesystem where practical
candidate public data read-only
output directory writable
hidden labels NOT mounted
time limit
memory limit
CPU limit
container cleanup
```

Candidate container：

```text
/data/train.csv          read-only
/data/test_features.csv  read-only

/output                  writable
```

绝不能：

```text
/data/hidden_test_labels.csv
```

---

# 23. Docker 必须做真实 Hidden Truth Attack Test

写一个恶意 candidate 尝试读取：

```text
/data/hidden_test_labels.csv
/host/hidden_test_labels.csv
/workspace/hidden_test_labels.csv
```

以及尝试枚举可能目录。

测试必须证明：

```text
candidate cannot read hidden labels
```

不要只检查 Docker 参数。

要做真实 attack。

---

# 24. Phase F：真实 Search Smoke Test

环境满足：

```text
Docker available
LLM API configured
```

时运行：

```bash
python examples/regression_demo.py \
  --llm \
  --drafts 2 \
  --improves 3
```

目标完整闭环：

```text
LLM
↓
solution.py
↓
Docker
↓
predictions.json
↓
Host RegressionVerifier
↓
Evidence
↓
Judge
↓
SearchEngine
↓
Improve Prompt
↓
LLM
```

要求：

搜索可以完成即可。

**不要求每一次 improve 都变好。**

不要为了漂亮 Demo 伪造 improvement。

---

# 25. Modeling 输出与 UX

第一版终端体验保持简单。

建议：

```text
VES Modeling
task: regression
metric: RMSE
search: 2 drafts + 3 improves

[draft0]
verification: VERIFIED
rmse: 43.271
mae: 34.816

[draft1]
verification: VERIFIED
rmse: 37.284
mae: 29.012

[improve0 <- draft1]
verification: VERIFIED
rmse: 31.827
mae: 24.992

BEST VERIFIED
candidate: ...
rmse: 31.827
```

不要因为缺少 callback 去修改 SearchEngine。

第一版 search 完成后统一输出 summary 也可以。

---

# 26. Run Artifacts

建议每一次 Modeling Session 创建：

```text
runs/<run-id>/
```

保存：

```text
summary.json
best_solution.py
verified_records/
config.json
```

如果容易实现，可以保存 candidate code。

不要保存：

```text
API key
secret
hidden labels
```

summary 至少包括：

```text
task
dataset fingerprint
drafts
improves
best candidate id
best verified rmse
best verified mae
rejected count
```

优先复用 VES VerificationRecord。

不要创建第二套重复审计格式。

---

# 27. 项目目录建议

第一版保持克制。

建议：

```text
ves-modeling/
│
├── pyproject.toml
├── README.md
├── .gitignore
│
├── docs/
│   ├── ves-core-understanding.md
│   ├── architecture.md
│   └── core-gaps.md
│
├── src/
│   └── ves_modeling/
│       ├── __init__.py
│       ├── session.py
│       │
│       └── regression/
│           ├── __init__.py
│           ├── problem.py
│           ├── context.py
│           ├── verifier.py
│           ├── generator.py
│           └── runner.py
│
├── fixtures/
│   └── candidates/
│       ├── linear_regression.py
│       ├── random_forest.py
│       ├── gradient_boosting.py
│       └── cheating_candidate.py
│
├── examples/
│   └── regression_demo.py
│
├── scripts/
│   └── generate_regression_data.py
│
├── tests/
│   ├── test_regression_verifier.py
│   ├── test_regression_problem.py
│   ├── test_mock_search.py
│   ├── test_candidate_claim_ignored.py
│   └── test_docker_hidden_truth.py
│
├── data/
│   └── regression/
│       ├── public/
│       └── host/
│
└── .vendor/
    └── Verified-Executable-Search/
```

如果发现某个文件暂时没必要：

不要为了满足目录结构创建空抽象。

---

# 28. Python 与依赖

使用：

```text
Python >= 3.11
```

第一版主要依赖：

```text
numpy
pandas
scikit-learn
pytest
```

如果需要 HTTP：

```text
httpx
```

不要加入大型框架。

不要使用：

```text
LangChain
CrewAI
AutoGen
复杂 Agent Framework
```

除非未来有明确理由。

VES Modeling 的 Agent/Search runtime 就是：

```text
VES
```

---

# 29. pyproject.toml

建立正常 Python package。

项目名：

```text
ves-modeling
```

import：

```python
import ves_modeling
```

由于 VES Core 当前可能尚未通过 PyPI 稳定分发：

开发阶段可以 pin Git dependency：

```text
verified-executable-search @ git+https://github.com/Zhuchen00123/Verified-Executable-Search.git@v0.1.0
```

但是：

在写入之前先检查 VES 当前 packaging 是否支持这种安装方式。

如果安装方式存在问题：

使用：

```text
.vendor/Verified-Executable-Search
```

editable/source development setup。

不要为了 dependency resolution 修改 VES Core。

---

# 30. 第一阶段 README

README 第一版不要吹成：

```text
Universal AI Scientist
Fully Autonomous Modeling
General Mathematical Modeling Agent
```

真实定位：

# VES Modeling

> Verifier-first executable search for computational modeling.

中文：

> 让 AI 反复改进可执行建模方案，但每一次成绩都由独立验证器决定。

README 第一屏至少展示：

```text
AI generates solution.py
        ↓
candidate claims RMSE = 0.000001
        ↓
VES ignores the claim
        ↓
Host recomputes RMSE = 31.82
        ↓
Search improves candidate
```

---

# 31. 第一阶段测试要求

使用：

```bash
pytest
```

至少覆盖：

```text
Artifact contract
Regression context
Regression verifier
RMSE calculation
MAE calculation
invalid prediction shape
non-finite predictions
claimed score ignored
Mock generator
SearchEngine integration
candidate lineage
hidden truth isolation
```

真实 Docker test 可以：

```text
skip if Docker unavailable
```

但：

Docker 可用的开发环境必须实际跑一次。

---

# 32. 静态检查

优先跟随 VES Core 风格。

建议：

```text
ruff
pyright
pytest
```

不要因为追求零 warning 大规模改依赖代码。

---

# 33. 每完成一个阶段都运行验收

阶段顺序：

## R0 — Core Understanding

完成：

```text
docs/ves-core-understanding.md
```

并确认不修改 Core。

---

## R1 — Regression Verification

完成：

```text
data generation
context
artifact
verifier
judge
hand-written candidates
```

验收：

```bash
pytest tests/test_regression_verifier.py
```

---

## R2 — Mock Search

完成：

```text
MockRegressionGenerator
SearchEngine integration
```

验收：

```bash
python examples/regression_demo.py --mock
```

必须是真 SearchEngine。

---

## R3 — Adversarial Demo

完成：

```text
claimed score attack
invalid artifact attack
hidden truth attack
```

验收：

candidate 自报指标不能影响 best selection。

---

## R4 — LLM Generator

完成：

```text
provider-neutral LLM adapter
draft prompt
improve prompt
```

无 API 时 unit test 使用 fake client。

---

## R5 — Docker Execution

完成真正：

```text
LLM code
→ Docker
→ artifact
```

并通过 hidden truth attack。

---

## R6 — Real Closed Loop

完成：

```text
LLM
→ Docker
→ Verification
→ Evidence
→ Judge
→ Search
→ LLM improve
```

至少：

```text
2 drafts
3 improves
```

成功完成一次。

---

# 34. Milestone 1 Definition of Done

VES Modeling Regression MVP 只有满足下面条件才算完成：

```text
[ ] VES Core 未被修改

[ ] Candidate 只看到 public modeling data

[ ] Host 持有 hidden labels

[ ] Candidate 输出 executable artifact

[ ] Host 独立计算 RMSE

[ ] Candidate self-reported RMSE 被忽略

[ ] VES VerificationPipeline 被真实使用

[ ] VES Evidence 被真实使用

[ ] VES Judge 被真实使用

[ ] VES SearchEngine 被真实使用

[ ] Candidate lineage 被保留

[ ] Mock Search 完整通过

[ ] adversarial candidate 被正确处理

[ ] Docker hidden-truth attack 通过

[ ] LLM Generator 可以 provider-neutral 配置

[ ] 至少一次真实 LLM + Docker closed-loop smoke test

[ ] README 可以让陌生人 30 秒理解项目
```

---

# 35. Milestone 1 完成前禁止做的东西

禁止：

```text
MCTS
Monte Carlo Tree Search
multi-agent
task decomposition
paper writing agent
EDA agent
notebook agent
automatic report generation
plugin system
distributed search
Pareto frontier
Rust rewrite
Web UI
Hugging Face app
complex database
Redis
Celery
Ray
Kubernetes
```

这些都不属于 Regression MVP。

---

# 36. 第二阶段：Constrained Optimization

Regression MVP 完成之后，再创建第二个独立 vertical slice：

```text
Constrained Optimization
```

例如：

```text
candidate output:
solution.json
```

Host 重新计算：

```text
objective
constraint violation
feasibility
```

Evidence：

```text
objective
weight
violation
```

Judge：

```text
Gate:
violation <= 0

Objective:
MAXIMIZE objective
```

目标：

验证同一个 Core 可以支持：

```text
Regression
MINIMIZE score
```

以及：

```text
Optimization
constraint + MAXIMIZE objective
```

---

# 37. 第三阶段：Mechanism Parameter Fitting

之后再做：

```text
Mechanism / Physical Parameter Fitting
```

例如：

```text
y = a * exp(-b*x) + c
```

Candidate 输出：

```json
{
  "a": ...,
  "b": ...,
  "c": ...
}
```

Host：

```text
重新计算 predictions
重新计算 RMSE
检查 parameter range
检查 physical constraints
```

这会成为 VES Modeling 面向：

```text
AI4Science
CAE
scientific modeling
```

的重要案例。

---

# 38. 什么时候才能开始抽象 ModelingTask？

不要在 Regression 阶段抽象。

只有：

```text
Regression
+
Constrained Optimization
```

至少两个场景完成后：

比较重复代码。

如果确实出现共同结构，再考虑：

```python
ModelingTask
ModelingSession
ModelingGenerator
```

原则：

> 两个真实实例之后再抽象。

而不是：

> 先想象未来需求再抽象。

---

# 39. VES Core 修改准入规则

如果你认为 VES Core 必须修改：

首先创建：

```text
docs/core-gaps.md
```

记录：

```text
Observed problem
Current workaround
Why application-layer solution is insufficient
Affected modeling domains
Proposed Core primitive
Backward compatibility impact
```

除 bug/security 问题外：

如果只有 Regression 一个场景需要：

```text
不要改 Core。
```

如果 Regression + Optimization 都出现相同缺陷：

才可以提出：

```text
VES v0.2 Core change proposal
```

不要自行实施，除非我明确要求。

---

# 40. Git 工作方式

如果当前目录不是 Git repo：

```bash
git init
```

建立：

```text
.gitignore
```

至少忽略：

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.pyright/
.ruff_cache/
.env
runs/
.vendor/
data/regression/host/
```

尤其：

```text
API keys
hidden labels
```

绝不能 commit。

如果 git identity 已配置，可以在重要里程碑做小 commit：

```text
R0 understand VES core
R1 add regression verifier
R2 add verified mock search
R3 add adversarial regression demo
R4 add LLM generator
R5 add Docker runner
R6 close real regression loop
```

如果 identity 未配置：

不要因为不能 commit 停止实现。

---

# 41. 安全要求

始终把 AI-generated code 当成 hostile code。

不得：

```text
exec(candidate_code)
eval(candidate_code)
subprocess python solution.py
```

直接运行真实 LLM candidate 于 Host。

测试 fixture 除外。

Candidate Docker：

```text
network = none
hidden truth = not mounted
public data = read-only
output = constrained writable dir
timeout
memory limit
cpu limit
cleanup
```

不要把：

```text
.env
API key
GitHub token
SSH key
host home
Docker socket
```

挂载进 candidate container。

---

# 42. 设计审美

代码优先：

```text
small
typed
explicit
testable
boring
```

避免：

```text
magic
global state
huge classes
deep inheritance
premature abstraction
hidden fallback
silent error swallowing
```

VES 最重要的是：

```text
trust boundary
```

不是框架炫技。

---

# 43. 遇到不确定 API 时怎么做

不要猜。

直接阅读：

```text
VES source
VES tests
VES docs
```

测试优先级：

```text
actual code
>
tests
>
current docs
>
this prompt
```

如果本提示词与当前 VES v0.1.0 实际代码不一致：

以实际代码为准。

在：

```text
docs/ves-core-understanding.md
```

记录差异。

---

# 44. 不要只告诉我你准备怎么做

从收到本提示词开始：

首先执行：

```bash
pwd
ls -la
git status 2>/dev/null || true
python --version
docker --version 2>/dev/null || true
```

然后定位 / clone VES。

然后阅读 Core。

然后创建：

```text
docs/ves-core-understanding.md
```

接着开始 R1。

不要在输出一份十页计划后停止。

规划应该服务于实现。

---

# 45. 工作汇报格式

每完成一个重要阶段，简短汇报：

```text
Completed:
- ...

Verified:
- command
- result

Files:
- ...

Core changes:
- none

Next:
- ...
```

如果测试失败：

说明：

```text
actual failure
root cause
fix
rerun result
```

不要隐藏失败。

---

# 46. 最终目标

第一阶段结束时，我希望可以运行：

```bash
python examples/regression_demo.py --mock
```

看到真实 VES verified search。

然后配置：

```text
VES_MODELING_LLM_BASE_URL
VES_MODELING_LLM_API_KEY
VES_MODELING_LLM_MODEL
```

运行：

```bash
python examples/regression_demo.py \
  --llm \
  --drafts 2 \
  --improves 3
```

看到：

```text
AI generates solution.py
        ↓
Docker executes it
        ↓
predictions.json
        ↓
Host recomputes RMSE
        ↓
VES records Evidence
        ↓
Judge ranks candidate
        ↓
SearchEngine selects anchor
        ↓
LLM improves solution.py
        ↓
BEST VERIFIED MODEL
```

并且存在恶意 Candidate：

```text
claimed RMSE = 0.000001
```

最终展示：

```text
Candidate claimed RMSE: 0.000001
Host verified RMSE:      <actual value>

Candidate claim ignored.
```

如果这条链真实成立：

**VES Modeling Regression MVP 就完成了。**

---

# 47. 最后的项目原则

整个开发过程中反复检查这句话：

> **AI can propose. It cannot grade itself.**

如果某个设计允许 Candidate：

```text
定义自己的评分
决定自己的可行性
访问隐藏答案
影响 Verifier
篡改 Judge
```

那么设计就是错的。

VES Modeling 不是为了让 AI 更自由地评价自己。

它是为了：

> **让 AI 可以自由探索解空间，同时把“什么是真的、什么更好”的裁判权牢牢留在可信 Host 一侧。**

现在开始执行 R0：理解 VES Core。

这份提示词我建议你**第一条消息就完整贴进去**。它最关键的地方不是告诉 Agent “帮我做数学建模框架”，而是把开发顺序锁成：

```text
读懂 VES
→ Regression Verifier
→ 手写 candidate
→ Mock Search
→ 对抗测试
→ LLM
→ Docker
→ 真正 closed loop
→ 第二个领域后才抽象
```

这样能显著降低 Agent 一上来给你造一堆 `BaseModelingTaskFactory` 之类空架构的概率。
