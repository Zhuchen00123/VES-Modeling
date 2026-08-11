# VES Core v0.1.0 理解文档（R0）

> Source of truth：`.vendor/Verified-Executable-Search`（commit `fe4fcc4`，tag v0.1.0）。
> 本文件记录 Modeling 层对 Core 的理解、每个对象的生命周期/信任边界，以及与 idea.md 的差异。

## 1. Verification 链路

```text
RawArtifact         不可信：候选产物（name + content）
ArtifactContract    可信：宿主规格（filename/media_type/size/required_fields/numeric_fields）
SafeArtifactLoader  宿主读取防护：O_NOFOLLOW + fstat(S_ISREG) + basename-only + 5MB + UTF-8
VerificationContext 宿主控制：隐藏 truth 只经 id + fingerprint()（单向摘要）参与审计
EvidenceVerifier    verify(raw_artifact, context) -> Evidence；只建立事实，不判优劣
VerificationPipeline 把 artifact 跑一遍 contract.validate -> context -> verifier -> VerificationRecord
VerificationResult  status(VERIFIED/INVALID_ARTIFACT/VERIFICATION_FAILED) + evidence + record + issues
```

各对象：
- **RawArtifact**：Candidate runner 输出物，由 `SafeArtifactLoader` 从磁盘读取（也可直接构造）。完全不可信；`ArtifactContract.validate` 之前宿主不信任 content。
- **ArtifactContract**：宿主声明，纯函数校验（filename 精确匹配、JSON 根必须 object、required_fields 存在、numeric_fields 有限数字且拒绝 bool/NaN/Infinity）。**只做通用合法性**，领域语义（predictions 长度、是否与 hidden labels 对齐）留给领域 verifier。
- **SafeArtifactLoader**：宿主读取防护（race-safe、symlink-safe、regular-file、大小上限、UTF-8）。生命周期：SearchEngine 每次验证时实例化，root=run_dir。
- **VerificationContext**：宿主工厂每次 verify 创建（`VerifiedProblem.make_context()`）；`id` 稳定标识，`fingerprint()` 必须单向摘要（reversible 禁止）；**不序列化进 record**（record 只带 context_id + context_fingerprint）。
- **EvidenceVerifier**：协议 `version: str` + `verify(raw_artifact, context) -> Evidence`。只返回 `Observation` 集合；`version` 写进 record。
- **Evidence / Observation**：不可变；Observation(value, uncertainty|None, provenance, name)；name 非空且 Evidence 内唯一；uncertainty 有限非负。
- **VerificationStatus**：VERIFIED / INVALID_ARTIFACT / VERIFICATION_FAILED —— 只判验证，不判可行。
- **VerificationRecord**：审计记录（schema v1）：verifier_version、context_id、context_fingerprint、evidence、artifact_sha256、contract_fingerprint、judge_spec_fingerprint、candidate_id、candidate_sha256、problem_ref 等；replay 引用（verifier_module/attr、context_factory_ref）均为非敏感。

## 2. Judgment：Evidence != Judgment

- `JudgeSpec`：gates（约束）+ objectives（参与排名的 observation、方向、容差）+ rule（lexicographic/pareto）。
- `Gate`：`holds(value)`/`violated_by(evidence)`；`observation=None` 作用于所有 observation；目标 observation 缺失视为违反。
- `Judge.verdict(evidence, spec)` -> Verdict(passed, failed_gates)；`Judge.compare(left, right, spec)` -> Comparison。
- **Feasibility first**：不可行恒输给可行；两个不可行按 gate 超量排序。
- 目标只从 `objectives` 声明的事实取（`rmse=0.31` 可以是 Evidence，但 Judge 只优化声明的 rmse，不自动优化 runtime）。
- Modeling 用法：Regression JudgeSpec = `objectives=(ObjectiveSpec("rmse", MINIMIZE),)`，MAE 只作 Evidence 不参与排名；可选 Gate(finite) 防非有限。

## 3. Search

实际签名（以源码为准）：
- `CandidateGenerator(Protocol)`：`draft(problem: VerifiedProblem, index: int) -> str`；`improve(problem, anchor: VerifiedCandidate) -> str`。
- `CodeRunner(Protocol)`：`run(code: str, run_id: str) -> RunResult`；RunResult 仅要求 `succeeded: bool` + `run_dir: Path`（预留 duration/stdout/stderr）。
- `AnchorPolicy(Protocol)`：`next_anchor(verified, judge, spec, rng=None, *, allow_infeasible=True) -> VerifiedCandidate | None`。
- `SearchEngine(problem=, generator=, runner=, anchor_policy=, drafts=, improves=, rng=)`；`search() -> SearchResult`。
- `SearchResult`：best_code / best_evidence / best_record / records / rejected / policy / drafts / improves / best_feasible。
- `_strip_code_fence`：SearchEngine 内部已剥离 markdown 围栏（真实 LLM 会包 ```python）。
- 引擎零题型引用：artifact filename 来自 `problem.contract.filename`；prompt 完全由注入的 generator 负责。
- 注意：`SearchEngine` 构造时若 `judge_spec.rule is PARETO` 直接 ValueError（v0.1 只支持全序）。
- 每轮 `_run_and_verify`：runner.run -> SafeArtifactLoader(contract.filename) -> pipeline.verify；只有 VERIFIED 才进 pool；rejected 计数非 VERIFIED/runner 失败。
- `Candidate` lineage：draft（root_id==id, parent=None, generation=0）/ improve（parent_id=anchor.id, root_id=anchor.root_id, generation+1）；`VerifiedCandidate(candidate, artifact, record)`。

## 4. Security（R5 依据）

- `DockerProcessRunner`（aide_solver/execution.py，**非 wheel 公共 API**，源码 checkout 可用）：`--network none --read-only --cap-drop ALL --security-opt no-new-privileges --user <uid>:<gid> --pids-limit --memory --cpus --tmpfs /tmp:noexec --ulimit nofile`；代码 bind 到 `/readonly`，输出写到 `/workspace`；`public_files` 白名单逐文件挂 `/data/<name>`（**不挂整个目录** → hidden labels 可安全放在同目录但绝不进容器）；超时/输出字节上限/容器清理。
- `LocalProcessRunner` **不是安全边界**：仅 trusted fixture / 单元测试。
- 安全边界结论：真实 LLM candidate 必须走 Docker（Rule 6）。
- idea.md 的 `/output/predictions.json` 契约与 VES DockerProcessRunner 的 `/workspace` 不一致 → Modeling 需要**薄 adapter**（R5）：同安全参数，输出目录挂到 `/output`。

## 5. 与 idea.md 的差异 / 注意点

1. `SearchEngine` 的 runner 契约是 `run(code, run_id)`，与 idea.md 一致；RunResult 最少字段 `succeeded` + `run_dir`（没有 artifact 字段）——artifact 由 SafeArtifactLoader 从 run_dir 读 contract.filename。
2. `CandidateGenerator.draft(problem, index)` / `improve(problem, anchor)` —— idea.md 描述一致。
3. VES 没有独立的 ArtifactContract JSON root/数组领域校验——predictions 长度/有限性由 RegressionVerifier 承担（与 idea.md §9 一致）。
4. `SearchEngine` 已内置 `_strip_code_fence`，prompt 仍建议显式要求不要 markdown 围栏。
5. DockerProcessRunner 输出目录是 `/workspace` 而非 `/output`：Modeling 的 DockerRegressionRunner 以薄 adapter 方式把输出挂到 `/output`（见 docs/architecture.md、docs/core-gaps.md）。
6. VES 的 `ves replay` 需要 record 的 verifier_module/context_factory_ref 指向可导入的 `module:attr`——Modeling 的 problem/context 放 `ves_modeling.regression.*` 以满足 replay 能力。
