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

## 第一版：Tabular Regression vertical slice

完整闭环（真实运行，非演示字符串）：

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
# 1. 依赖（Python >= 3.11；自动安装 PyPI VES Core）
pip install -e ".[dev]"

# 2. 生成数据（固定 seed，可复现）
python scripts/generate_regression_data.py

# 3. 测试
pytest

# 4. Mock 搜索（真 SearchEngine + 可信手写候选）
python examples/regression_demo.py --mock

# 5. 真实 LLM + Docker 闭环（需先构建沙箱镜像）
bash scripts/build_runner_image.sh
export VES_MODELING_LLM_BASE_URL=... VES_MODELING_LLM_API_KEY=... VES_MODELING_LLM_MODEL=...
python examples/regression_demo.py --llm --drafts 2 --improves 3
```

每次搜索在 `runs/<run-id>/` 保存 `summary.json`、`best_solution.py`、`config.json`（不保存 API key / 隐藏标签）。

## 稳定 Regression API（R7.2–R7.3）

上层项目（VES-MathModeling-Skill 等）通过 `run_regression_search` 调用，
不依赖 `examples/regression_demo.py`，也不需要理解 VES Core 内部类型。

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

## 进度（idea.md R0-R7）

- R0 Core 理解 ✅ `docs/ves-core-understanding.md`
- R1 Regression Verifier ✅ `pytest tests/test_regression_verifier.py`
- R2 Mock Search ✅ `python examples/regression_demo.py --mock`（真 SearchEngine）
- R3 Adversarial ✅ claim-ignored + 真实 Docker hidden-truth attack
- R4 LLM Generator ✅ fake-client 单测（`tests/test_llm_generator.py`，无需 API）
- R5 Docker Execution ✅ `tests/test_docker_hidden_truth.py` 真实容器攻击
- R6 Real Closed Loop ✅ 真实 LLM（deepseek-v4-flash / OpenCode Go / reasoning_effort=high / SSE 流式）+ 真实 Docker 闭环跑通：
  `2 drafts + 3 improves` 全部 VERIFIED（rejected=0），BEST VERIFIED rmse=62.428 mae=47.177（run 3f9b78c1a9a6）；
  配置 `VES_MODELING_LLM_BASE_URL/API_KEY/MODEL` 后跑 `python examples/regression_demo.py --llm --drafts 2 --improves 3`
- R7.1 Cleanup ✅ B-005 运行日志落盘（stdout.log/stderr.log）、hidden labels finite 校验、
  RegressionVerificationContext invariant、CI（GitHub Actions，Docker 测试独立 marker）
- R7.2 Stable Regression API ✅ `ves_modeling.regression.run_regression_search`（见上）+ API 集成测试
- R7.3 Regression delivery closure ✅ apply API、稳定 JSON 协议、provenance、统一 run 树、
  结构化失败分类与显式 target/ID/row-order 数据契约；`122 passed`（含 Docker 安全测试）

## 目录

```text
src/ves_modeling/regression/   API / apply / data contract / verifier / generator / runner
fixtures/candidates/           可信手写候选 + 对抗候选（cheating_candidate.py）
examples/regression_demo.py    --mock / --llm 入口
scripts/generate_regression_data.py
tests/                         verifier / problem / mock search / claim-ignored / docker hidden truth
docs/                          ves-core-understanding / architecture / core-gaps
PyPI: verified-executable-search    VES Core v0.1.0（正式依赖）
```

## 安全

- `LocalRegressionRunner` 只用于可信 fixture / 单元测试；真实 LLM candidate 必须走 `DockerRegressionRunner`。
- Docker 隐藏真值攻击测试（`tests/test_docker_hidden_truth.py`）用真实容器验证 candidate 读不到 hidden labels。
- 详见 `SECURITY.md`（VES Core）与 `docs/core-gaps.md`。
