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
# 1. 依赖（Python >= 3.11）
pip install -e .vendor/Verified-Executable-Search
pip install -e ".[dev]"

# 2. 生成数据（固定 seed，可复现）
python scripts/generate_regression_data.py   # 默认加州房价（20640 样本）

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

## 进度（idea.md R0-R6）

- R0 Core 理解 ✅ `docs/ves-core-understanding.md`
- R1 Regression Verifier ✅ `pytest tests/test_regression_verifier.py`
- R2 Mock Search ✅ `python examples/regression_demo.py --mock`（真 SearchEngine）
- R3 Adversarial ✅ claim-ignored + 真实 Docker hidden-truth attack
- R4 LLM Generator ✅ fake-client 单测（`tests/test_llm_generator.py`，无需 API）
- R5 Docker Execution ✅ `tests/test_docker_hidden_truth.py` 真实容器攻击
- R6 Real Closed Loop ✅ 真实 LLM（deepseek-v4-flash / OpenCode Go / reasoning_effort=high / SSE 流式）+ 真实 Docker 闭环跑通：
  `2 drafts + 3 improves` 全部 VERIFIED（rejected=0），BEST VERIFIED rmse=62.428 mae=47.177（run 3f9b78c1a9a6）；
  配置 `VES_MODELING_LLM_BASE_URL/API_KEY/MODEL` 后跑 `python examples/regression_demo.py --llm --drafts 2 --improves 3`

## 目录

```text
src/ves_modeling/regression/   problem / context / verifier / generator / runner
fixtures/candidates/           可信手写候选 + 对抗候选（cheating_candidate.py）
examples/regression_demo.py    --mock / --llm 入口
scripts/generate_regression_data.py
tests/                         verifier / problem / mock search / claim-ignored / docker hidden truth
docs/                          ves-core-understanding / architecture / core-gaps
.vendor/Verified-Executable-Search   VES Core v0.1.0（不改动）
```

## 安全

- `LocalRegressionRunner` 只用于可信 fixture / 单元测试；真实 LLM candidate 必须走 `DockerRegressionRunner`。
- Docker 隐藏真值攻击测试（`tests/test_docker_hidden_truth.py`）用真实容器验证 candidate 读不到 hidden labels。
- 详见 `SECURITY.md`（VES Core）与 `docs/core-gaps.md`。
