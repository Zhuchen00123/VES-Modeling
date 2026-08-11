Coding Spec

基本原则

本项目主要通过 AI 辅助开发。

- 优先简单、直接、可维护的实现。
- 优先最小修改，不做无关重构。
- 不为了“更优雅”“更现代”或“未来可能需要”增加复杂设计。
- 尽量保持现有行为和公开接口不变。
- 优先复用现有代码，不重复实现已有功能。
- 没有实际问题时，不主动优化性能或更换语言。
- 修改前先阅读相关代码和调用关系。

---

模块设计

每个模块尽量只负责一类事情。

核心逻辑尽量保持：

输入
 ↓
处理
 ↓
输出

推荐：

result = process(data, options)

避免核心函数：

- 到处读取或修改全局变量
- 隐式依赖其他模块状态
- 同时承担网络、文件、数据库和算法逻辑
- 内部自行寻找大量外部数据

模块之间通过明确的函数、类或数据结构通信。

---

接口

公开接口尽量保持稳定。

例如：

def solve(data: InputData, options: Options) -> Result:
    ...

除非当前任务明确要求，否则不要随意：

- 修改函数参数
- 修改返回值格式
- 修改公开函数名称
- 修改公开类名称
- 改变已有行为

如果必须修改接口，应同步检查并修改所有调用方和测试。

---

类型

新代码尽量添加 Python 类型标注：

def calculate(
    values: list[float],
    threshold: float,
) -> CalculationResult:
    ...

优先使用明确的数据结构，而不是大量无结构的 "dict"。

例如：

from dataclasses import dataclass


@dataclass
class Result:
    value: float
    confidence: float

避免为了类型系统引入不必要的复杂泛型和抽象。

---

错误处理

不要静默吞掉异常。

禁止：

try:
    ...
except:
    pass

推荐：

try:
    ...
except ValueError as exc:
    ...

原则：

- 捕获尽可能明确的异常。
- 底层提供清晰的错误信息。
- 上层决定重试、记录、返回还是终止。
- 不要使用异常控制正常业务流程。

---

日志

正式代码使用 "logging"，不要大量依赖 "print()"：

import logging

logger = logging.getLogger(__name__)

日志中禁止输出：

- 密码
- Token
- Cookie
- API Key
- 其他敏感信息

---

测试

核心功能应有测试。

重点覆盖：

- 正常输入
- 边界输入
- 错误输入
- 已知 Bug
- 关键业务行为

修复 Bug 时：

«优先添加能够复现 Bug 的测试，再修复代码。»

重构时：

«原有测试必须继续通过。»

默认测试：

pytest

---

代码检查

推荐使用：

ruff
pyright
pytest

修改完成后运行：

ruff check .
pyright
pytest

如启用了 Ruff 格式化：

ruff format .

不要引入新的明显 warning 或 error。

---

AI 修改规则

修改代码前：

1. 阅读相关现有实现。
2. 查找相关调用方。
3. 检查项目中是否已有可复用功能。
4. 确定完成任务所需的最小修改范围。

修改代码时：

- 不修改与当前任务无关的文件。
- 不随意删除已有功能。
- 不擅自改变已有行为。
- 不创建重复实现。
- 不进行无关格式化。
- 不顺手升级无关依赖。
- 不为了简单需求重新设计整个架构。
- 优先修改现有实现，而不是建立一套平行的新实现。
- 保持代码风格与现有项目一致。

修改完成后：

1. 检查相关调用方。
2. 更新必要测试。
3. 运行相关测试。
4. 运行静态检查。
5. 确认没有引入无关行为变化。

---

防止过度设计

如果普通函数能够解决问题：

def solve(...):
    ...

不要无理由改成：

BaseSolver
SolverInterface
SolverFactory
SolverManager
SolverContext
DefaultSolver

除非当前项目确实存在对应的复杂性，否则避免增加：

- Factory
- Manager
- Wrapper
- Adapter
- Abstract Base Class
- 多层继承
- 复杂依赖注入
- 无必要的设计模式
- 无必要的抽象层

不要为了假设中的未来需求增加复杂度。

---

重构规则

适合重构的情况：

- 存在明显重复代码
- 模块职责严重混乱
- 当前结构已经妨碍功能修改
- 当前任务明确要求重构
- 有足够测试保证行为不变

重构原则：

1. 尽量保持行为不变。
2. 尽量保持公开接口不变。
3. 分阶段修改。
4. 每一步保持测试通过。
5. 不顺带修改无关代码。

尽量不要同时进行：

功能修改
+
大规模重构
+
依赖升级

除非确实必要。

---

性能

默认原则：

«Correctness first. Profile before optimizing.»

没有明确性能问题时：

- 不进行无意义的微优化。
- 不提前增加复杂缓存。
- 不提前引入复杂并发。
- 不因为未来可能高并发而重写代码。
- 不主动把 Python 改写成 Rust 或 C++。

出现真实性能问题后：

1. Benchmark
2. Profile
3. 找到真实瓶颈
4. 优先改善算法
5. 优先使用成熟库
6. 最后考虑 Rust / C++ 等底层实现

---

Rust / C++ 重写

如果未来需要把 Python 核心重写为 Rust 或 C++，保留 Python 原实现作为正确性参考，直到新实现验证完成。

使用相同输入进行对拍：

                 ┌─→ Python 原实现 ─→ Result A
相同测试输入 ────┤
                 └─→ Rust/C++实现 ─→ Result B

Result A ≈ Result B

应进行：

- 固定测试数据对比
- 边界输入测试
- 随机输入对拍
- 异常输入测试
- 性能 Benchmark

确认行为一致后再替换旧实现。

---

完成标准

一次代码修改完成前确认：

- [ ] 实现了当前要求
- [ ] 没有修改无关行为
- [ ] 没有不必要的大规模重构
- [ ] 没有增加不必要的依赖
- [ ] 相关测试通过
- [ ] Ruff 没有明显新增问题
- [ ] Pyright 没有明显新增问题
- [ ] 接口变化已经同步修改调用方
- [ ] 没有泄露敏感信息
- [ ] 代码与现有项目风格一致

---

AI 默认工作原则

Read the relevant existing code before modifying it.

Prefer the smallest correct change.

Preserve existing behavior and public interfaces unless changing them is necessary.

Do not introduce speculative abstractions or premature optimization.

Do not refactor unrelated code.

Reuse existing implementations when possible.

Keep module boundaries clear and dependencies explicit.

Add or update tests when behavior changes.

Run relevant tests and static checks after modifications.

If you identify a larger architectural improvement, mention it separately instead of implementing it automatically.
