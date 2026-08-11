# VES Modeling 架构（第一版：Tabular Regression）

```text
train.csv / test_features.csv / hidden_test_labels.csv
        |
        v
RegressionProblem (VerifiedProblem: contract + context_factory + verifier + judge_spec)
        |
        v
CandidateGenerator (MockRegressionGenerator | LLMRegressionGenerator)
        |  solution.py
        v
CodeRunner (LocalRegressionRunner [fixtures] | DockerRegressionRunner [LLM])
        |  predictions.json
        v
SafeArtifactLoader -> ArtifactContract -> RegressionVerifier(context)
        |  Evidence(rmse, mae)  [claimed_rmse 永远忽略]
        v
Judge (MINIMIZE rmse) -> SearchEngine -> improve(anchor) -> BEST VERIFIED
```

模块职责（Concrete first）：
- `regression/context.py`：`RegressionVerificationContext`（隐藏 labels、id、fingerprint）。
- `regression/verifier.py`：`RegressionVerifier`（结构校验 + 宿主 RMSE/MAE）。
- `regression/problem.py`：`build_regression_problem(...)` 组装 VerifiedProblem。
- `regression/generator.py`：Mock 与 LLM 两类 generator；prompt 在 LLM 版内。
- `regression/runner.py`：Local/Docker 两类 CodeRunner adapter。
- `examples/regression_demo.py`：--mock / --llm 入口 + runs/<run-id>/summary.json。
- `scripts/generate_regression_data.py`：固定 random_state 造数。

本版本不引入 ModelingTask/UniversalMetric/TaskRegistry 等抽象（Rule 2）。
