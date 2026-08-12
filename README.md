# VES Modeling

> Verifier-first executable search for computational modeling.

让 AI 反复改进可执行建模方案，但每一次成绩都由独立验证器决定。

```
AI generates solution.py
        ↓
candidate claims RMSE = 0.000001
        ↓
VES ignores the claim
        ↓
Host recomputes RMSE = <真实值>
        ↓
Search improves candidate
```

核心原则（继承 VES Core）：**AI can propose. It cannot grade itself.**

## 支持的十六类建模场景（R7.3–R22）

| Slice | 输入（candidate 可见） | Host 真值 | Verified 指标 | 稳定 API |
|---|---|---|---|---|
| **Regression** 回归 | `train.csv` / `test_features.csv` | `hidden_test_labels.csv` | rmse, mae | `run_regression_search` |
| **Forecasting** 时间序列 | 多系列长表 + 可选外生变量 | 未来各步 hidden target | rmse, mae, smape | `run_forecasting_search` |
| **Classification** 分类 | 特征 + 类别标签（公开顺序） | hidden 标签/类别 | accuracy, macro_f1, log_loss, auroc, brier, ece, confusion | `run_classification_search` |
| **Optimization** 约束优化 | `problem.json`（完整公开实例） | 无（实例即完整真值） | 界/约束/整性违反 + objective | `run_optimization_search` |
| **ODE** 微分方程 | `train.csv`(t,y) + `test_features.csv` | `hidden_test_values.csv` | rmse, mae | `run_ode_search` |
| **Clustering** 聚类 | 特征矩阵（无标签） | `hidden_test_labels.csv`（参考划分） | ari, nmi, v_measure, silhouette | `run_clustering_search` |
| **Anomaly** 异常检测 | 正常样本特征 + 待判样本 | `hidden_test_labels.csv`（二元） | auroc, average_precision / f1, balanced_accuracy | `run_anomaly_search` |
| **Graph** 图论 | `problem.json`（图完整实例） | 无（实例即完整真值） | 路径/流量/树总权重 + 违反残差 | `run_graph_search` |
| **Monte Carlo** 随机模拟 | `problem.json`（随机问题定义） | 无（host 持解析真值） | 相对/绝对误差 + CI 覆盖 | `run_montecarlo_search` |
| **Multi-objective** 多目标优化 | `problem.json`（双目标完整实例） | 无（实例即完整真值） | hypervolume(2D) + 支配计数 | `run_multiobjective_search` |
| **Recommendation** 推荐系统 | user-item 评分历史 + 待预测对 | `hidden_test_ratings.csv` | rmse, mae, ndcg@5(审计) | `run_recommendation_search` |
| **Probabilistic** 概率推断 | `problem.json`(分布族) + 观测样本 | `hidden_parameters.json`（真参数） | 相对/绝对误差 + CI 覆盖 | `run_probabilistic_search` |
| **Queueing** 排队论 | `problem.json`（排队系统定义） | 无（host 持解析公式真值） | 相对/绝对误差 + CI 覆盖 | `run_queueing_search` |
| **Association** 关联规则 | `train.csv`（事务长表） | `hidden_test_transactions.csv` | mean_lift, mean_confidence | `run_association_search` |
| **Survival** 生存分析 | 观测 time/event + 特征 | `hidden_test_outcomes.csv` | c_index, mae(time 模式) | `run_survival_search` |
| **Assignment/TSP** 指派/旅行商 | `problem.json`（成本/距离矩阵） | 无（实例即完整真值） | total_cost | `run_assignment_search` |

每个 slice 都有配套 `apply_*_solution`（无 host 真值时唯一成功状态
`produced_unverified`，绝不伪造指标）与稳定 `capabilities()` JSON 能力声明
（`API_SCHEMA_VERSION=1.0`）。跨 slice 公共契约与冻结决策见
[`docs/multislice-contract.md`](docs/multislice-contract.md)。

## Regression vertical slice 完整闭环

真实运行，非演示字符串：

```text
train.csv / test_features.csv / hidden_test_labels.csv
  → LLM/Mock generator → solution.py
  → Docker/Local runner → predictions.json
  → SafeArtifactLoader → ArtifactContract → RegressionVerifier（宿主复算 RMSE/MAE）
  → Evidence → Judge（MINIMIZE rmse）→ SearchEngine → improve(anchor)
  → BEST VERIFIED MODEL
```

- Candidate 只能看到 `/data/train.csv`、`/data/test_features.csv`；`hidden_test_labels.csv` 永不挂载、不进入 prompt、不写进 record。
- Candidate 自报的 `claimed_rmse` / stdout 分数**一律忽略**，Judge 只消费宿主 `Evidence`。
- 真实 LLM 候选只在 Docker 沙箱运行（`--network none`、`--read-only`、`--cap-drop ALL`、no-new-privileges、非 root、资源限制、public_files 白名单逐文件挂载）。

## 快速开始

```bash
# 1. 依赖（Python >= 3.11；自动安装 PyPI VES Core + numpy/pandas/scikit-learn/scipy）
pip install -e ".[dev]"

# 2. 生成数据（固定 seed，可复现）
python scripts/generate_regression_data.py

# 3. 测试
pytest

# 4. Mock 搜索（真 SearchEngine + 可信手写候选）
python examples/regression_demo.py --mock

> Demo 会在 `--root` 下生成 `data/` 与 `runs/` 目录（均已 gitignore，不入库）。

# 5. 真实 LLM + Docker 闭环（需先构建沙箱镜像）
bash scripts/build_runner_image.sh
export VES_MODELING_LLM_BASE_URL=... VES_MODELING_LLM_API_KEY=... VES_MODELING_LLM_MODEL=...
python examples/regression_demo.py --llm --drafts 2 --improves 3
```

每次搜索在 `runs/<run-id>/` 保存 `summary.json`、`best_solution.py`、`config.json`（不保存 API key / 隐藏标签）。

## 稳定 Regression API（R7.2–R7.3）

上层项目通过 `run_regression_search` 调用，不依赖 `examples/regression_demo.py`，
也不需要理解 VES Core 内部类型。

```python
from pathlib import Path

from ves_modeling.regression import run_regression_search

result = run_regression_search(
    public_dir=Path("data/regression/public"),  # train.csv + test_features.csv（candidate 可见）
    host_dir=Path("data/regression/host"),      # hidden_test_labels.csv（host 专属，绝不外泄）
    drafts=2,
    improves=3,
    workspace=Path("runs"),
    generator="mock",  # "mock"=可信夹具+本地 runner；"llm"=LLM generator+Docker runner
    target_column="target",
    id_column=None,
    row_order="input",
)
print(result.status, result.best_rmse, result.best_mae, result.rejected)
print(result.to_summary())  # API_SCHEMA_VERSION=1.0 的纯 JSON 视图
```

- `RegressionSearchResult` 字段：`status`（verified / no_verified）、`best_code`、
  `best_evidence`、`best_rmse`、`best_mae`、`rejected`、`run_dir`、`records`、
  `candidates`、`data_contract`
- 每次搜索在统一的 `runs/<run-id>/` 保存 `best_solution.py`、`summary.json`、
  `config.json`、`provenance.json`，每个候选在 `candidates/<attempt>/` 保存代码、
  stdout/stderr、artifact 与结构化 `run.json`
- 成绩全部来自宿主 verifier，从不采信 candidate 自报指标

### 应用已选方案到未知测试集

```python
from ves_modeling.regression import apply_regression_solution

applied = apply_regression_solution(
    result.best_code,
    public_dir=Path("formal/public"),  # full train.csv + unknown test_features.csv
    workspace=Path("runs"),
    # 未指定 trusted_code=True 时默认在 Docker 中执行
)
print(applied.status)       # produced_unverified
print(applied.to_summary()) # 无官方 labels 时绝不生成 RMSE/MAE
```

`apply_regression_solution` 成功只表示预测文件已按契约产出并通过结构校验，状态固定为
`produced_unverified`。它保存代码/数据/预测哈希、日志、Docker 镜像身份与运行环境，
但在没有官方标签时不会伪造任何质量指标。

### 显式数据契约

- `target_column` 默认 `target`，可自定义。
- `id_column` 可选；配置后 train/test/host 的 ID 必须非空、唯一且集合一致，宿主标签按
  public test ID 对齐。
- `row_order="input"` 保持兼容格式 `{"predictions": [number, ...]}`；
  `row_order="id"` 要求 `{"predictions": [{"id": ..., "prediction": number}, ...]}`。
- train 去掉 target 后的输入列必须与 test 名称及顺序完全一致；ID 列记录在
  `input_columns`，但不会被宣称为模型 `feature_columns`。
- `capabilities()` 返回稳定的 JSON 能力声明；完整交付契约见
  [`docs/r7.3-delivery-contract.md`](docs/r7.3-delivery-contract.md)。

## 其余十五个 slice 的稳定 API

```python
from ves_modeling.forecasting import run_forecasting_search
from ves_modeling.classification import run_classification_search
from ves_modeling.optimization import run_optimization_search
from ves_modeling.ode import run_ode_search
from ves_modeling.clustering import run_clustering_search
from ves_modeling.anomaly import run_anomaly_search
from ves_modeling.graph import run_graph_search
from ves_modeling.montecarlo import run_montecarlo_search
from ves_modeling.multiobjective import run_multiobjective_search
from ves_modeling.recommendation import run_recommendation_search
from ves_modeling.probabilistic import run_probabilistic_search
from ves_modeling.queueing import run_queueing_search
from ves_modeling.association import run_association_search
from ves_modeling.survival import run_survival_search
from ves_modeling.assignment import run_assignment_search

# 时间序列：key 模式（series_id + 严格 ISO 时间戳）或 input 模式
fc = run_forecasting_search(public_dir, host_dir, drafts=2, improves=3,
                            workspace=Path("runs"), generator="mock",
                            frequency="D", row_order="key")
print(fc.status, fc.best_rmse, fc.best_mae, fc.best_smape)

# 分类：host 固定类别顺序；candidate 输出 label + probabilities
clf = run_classification_search(public_dir, host_dir, drafts=2, improves=3,
                                workspace=Path("runs"), generator="mock",
                                classes=["no", "yes"])
print(clf.status, clf.best_macro_f1, clf.best_log_loss, clf.best_auroc)

# 约束优化：problem.json 即完整实例，无 host 目录
opt = run_optimization_search(public_dir, drafts=2, improves=3,
                              workspace=Path("runs"), generator="mock")
print(opt.status, opt.best_feasible, opt.best_objective)

# ODE：train.csv(t,y) + test_features.csv，host 隐藏未来/插值点真值
ode = run_ode_search(public_dir, host_dir, drafts=2, improves=3,
                     workspace=Path("runs"), generator="mock")
print(ode.status, ode.best_rmse, ode.best_mae)

# 聚类：无监督；ARI/NMI 对簇命名排列不变
clustering = run_clustering_search(public_dir, host_dir, drafts=2, improves=3,
                                   workspace=Path("runs"), generator="mock")
print(clustering.status, clustering.best_ari, clustering.best_nmi,
      clustering.best_v_measure)

# 异常检测：score（越大越异常）或 label 输出
anomaly = run_anomaly_search(public_dir, host_dir, drafts=2, improves=3,
                             workspace=Path("runs"), generator="mock")
print(anomaly.status, anomaly.best_auroc, anomaly.best_average_precision)

# 图论：problem.json 完整实例（最短路/最大流/最小生成树）
graph = run_graph_search(public_dir, drafts=2, improves=3,
                         workspace=Path("runs"), generator="mock")
print(graph.status, graph.best_feasible, graph.best_total_weight)

# 蒙特卡洛：problem.json 定义随机问题，host 持解析真值
mc = run_montecarlo_search(public_dir, drafts=2, improves=3,
                           workspace=Path("runs"), generator="mock")
print(mc.status, mc.best_relative_error, mc.best_absolute_error)

# 多目标优化：双目标 Pareto 解集，hypervolume 2D 精确
moo = run_multiobjective_search(public_dir, drafts=2, improves=3,
                                workspace=Path("runs"), generator="mock")
print(moo.status, moo.best_hypervolume, moo.best_non_dominated_count)

# 推荐系统：user-item 评分预测，host 隐藏评分 + NDCG@5 审计
rec = run_recommendation_search(public_dir, host_dir, drafts=2, improves=3,
                                workspace=Path("runs"), generator="mock")
print(rec.status, rec.best_rmse, rec.best_mae, rec.best_ndcg)

# 概率推断：从观测样本估计分布参数，host 持真参数
prob = run_probabilistic_search(public_dir, host_dir, drafts=2, improves=3,
                                workspace=Path("runs"), generator="mock")
print(prob.status, prob.best_relative_error, prob.best_absolute_error)

# 排队论：M/M/1 或 M/M/c 仿真估计，host 持解析公式真值
que = run_queueing_search(public_dir, drafts=2, improves=3,
                          workspace=Path("runs"), generator="mock")
print(que.status, que.best_relative_error, que.best_absolute_error)

# 关联规则：train 学规则，host 在隐藏事务上重算 lift/confidence
assoc = run_association_search(public_dir, host_dir, drafts=2, improves=3,
                               workspace=Path("runs"), generator="mock")
print(assoc.status, assoc.best_mean_lift, assoc.best_mean_confidence)

# 生存分析：风险分数或预测时间，host 隐藏结局
surv = run_survival_search(public_dir, host_dir, drafts=2, improves=3,
                           workspace=Path("runs"), generator="mock")
print(surv.status, surv.best_c_index)

# 指派/TSP：problem.json 完整实例（成本/距离矩阵）
assign = run_assignment_search(public_dir, drafts=2, improves=3,
                               workspace=Path("runs"), generator="mock")
print(assign.status, assign.best_total_cost)
```

各 slice 的 apply API（`apply_forecasting_solution` /
`apply_classification_solution` / `apply_optimization_solution` / `apply_ode_solution` /
`apply_clustering_solution` / `apply_anomaly_solution` /
`apply_graph_solution` / `apply_montecarlo_solution` /
`apply_multiobjective_solution` / `apply_recommendation_solution` /
`apply_probabilistic_solution` / `apply_queueing_solution` / `apply_association_solution` /
`apply_survival_solution` / `apply_assignment_solution`）与 regression 同构：
默认 Docker 执行、无 host 真值时状态为 `produced_unverified`；optimization 的 apply
额外附 host 重算的可行性/目标事实（全局最优从不声称）。详见
`docs/multislice-contract.md` 与各 slice 的 `capabilities()`。

## 进度（idea.md R0-R10）

- R0 Core 理解 ✅ `docs/ves-core-understanding.md`
- R1 Regression Verifier ✅ `pytest tests/test_regression_verifier.py`
- R2 Mock Search ✅ `python examples/regression_demo.py --mock`（真 SearchEngine）
- R3 Adversarial ✅ claim-ignored + 真实 Docker hidden-truth attack
- R4 LLM Generator ✅ fake-client 单测（`tests/test_llm_generator.py`，无需 API）
- R5 Docker Execution ✅ `tests/test_docker_hidden_truth.py` 真实容器攻击
- R6 Real Closed Loop ✅ 真实 LLM（deepseek-v4-flash / OpenCode Go / reasoning_effort=high / SSE 流式）+ 真实 Docker 闭环跑通：
  `2 drafts + 3 improves` 全部 VERIFIED（rejected=0），BEST VERIFIED rmse=62.428 mae=47.177（run 3f9b78c1a9a6）
- R7.1 Cleanup ✅ B-005 运行日志落盘、hidden labels finite 校验、context invariant、CI
- R7.2 Stable Regression API ✅ `run_regression_search` + API 集成测试
- R7.3 Regression delivery closure ✅ apply API、稳定 JSON 协议、provenance、统一 run 树、结构化失败、显式数据契约
- R8 Forecasting ✅ key/input 双模式、真实 frequency 校验、严格 ISO 时间、rmse/mae/smape
- R9 Classification ✅ label/prob 契约、host class order、六指标 + confusion、argmax tie-first
- R10 Optimization ✅ 公开 problem.json、host 重算违反/目标、tol 1e-6、绝不声称全局最优
- R11 ODE ✅ 轨迹/单轨迹双模式、严格递增 t、host 按 (trajectory_id,t) 对齐重算 rmse/mae、solve_ivp Mock
- R12 Clustering ✅ ARI/NMI/V-measure 排列不变、silhouette 可选、KMeans+spectral Mock
- R13 Anomaly ✅ score/label 双模式、AUROC/AP/F1/balanced-accuracy、IsolationForest+z-score Mock
- R14 Graph ✅ 最短路/最大流/最小生成树、纯 Python Mock、host 重算权重/流量+可行性
- R15 Monte Carlo ✅ expectation/integral/probability 三 kind 解析真值、relative/absolute error、CI 审计
- R16 Multi-objective ✅ 双目标 Pareto、hypervolume 2D 精确、随机权重标量化 Mock
- R17 Recommendation ✅ user-item 评分预测、rmse/mae/ndcg@5 审计、偏差基线+SVD Mock
- R18 Probabilistic ✅ normal/gamma/beta 参数估计、host 真参数、MLE+bootstrap CI Mock
- R19 Queueing ✅ M/M/1 与 M/M/c、解析公式真值、离散事件仿真 Mock
- R20 Association ✅ 隐藏事务验证规则、mean_lift/mean_confidence、Apriori+随机 Mock
- R21 Survival ✅ Harrell C-index、risk/time 双模式、Cox 简化+KM Mock
- R22 Assignment/TSP ✅ 匈牙利/最近邻+2-opt 纯 Python Mock、host 复算总成本
- T-010 Suite ✅ 四 slice 集成测试、跨 slice 契约文档、本 README 概览

当前全量测试：**non-Docker 390 passed / 37 deselected；Docker marker 37 passed**
（Docker Desktop 真实容器 hidden-truth attack）。

## 目录

```text
src/ves_modeling/regression/     API / apply / data contract / verifier / generator / runner
src/ves_modeling/forecasting/    R8 时间序列 slice（10 模块）
src/ves_modeling/classification/ R9 分类 slice（10 模块）
src/ves_modeling/optimization/   R10 约束优化 slice（10 模块）
src/ves_modeling/ode/           R11 ODE slice（10 模块）
src/ves_modeling/clustering/    R12 聚类 slice（10 模块）
src/ves_modeling/anomaly/      R13 异常检测 slice（10 模块）
src/ves_modeling/graph/        R14 图论 slice（10 模块）
src/ves_modeling/montecarlo/   R15 随机模拟 slice（10 模块）
src/ves_modeling/multiobjective/ R16 多目标优化 slice（10 模块）
src/ves_modeling/recommendation/ R17 推荐系统 slice（10 模块）
src/ves_modeling/probabilistic/  R18 概率推断 slice（10 模块）
src/ves_modeling/queueing/       R19 排队论 slice（10 模块）
src/ves_modeling/association/   R20 关联规则 slice（10 模块）
src/ves_modeling/survival/      R21 生存分析 slice（10 模块）
src/ves_modeling/assignment/    R22 指派/TSP slice（10 模块）
fixtures/candidates/             可信手写候选 + 对抗候选（cheating_candidate.py）
fixtures/{forecasting,classification,optimization,ode,clustering,anomaly,graph,montecarlo,multiobjective,recommendation,probabilistic,queueing,association,survival,assignment}/  各 slice 可信候选
examples/regression_demo.py      --mock / --llm 入口
scripts/generate_regression_data.py
tests/                           verifier / problem / mock search / claim-ignored / docker hidden truth / suite integration
docs/                            ves-core-understanding / architecture / core-gaps / r7.3 contract / multislice-contract
PyPI: verified-executable-search    VES Core v0.1.0（正式依赖）
```

## 安全

- `LocalRegressionRunner` 只用于可信 fixture / 单元测试；真实 LLM candidate 必须走 `DockerRegressionRunner`。
- Docker 隐藏真值攻击测试用真实容器验证 candidate 读不到 hidden labels（classification 同款；optimization 验证只读挂载 problem.json）。
- 详见 `SECURITY.md`（VES Core）与 `docs/core-gaps.md`。
