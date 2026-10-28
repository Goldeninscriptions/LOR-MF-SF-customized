# LOR-MF-SF 最小论文原型开发指南

## 1. 文档用途

本文档是 `LOR-MF-SF-customized` 的持续更新开发指南，用于记录：

- 论文原型的研究边界；
- 已达成的设计决定；
- 尚待讨论的问题；
- 从 `Parallel-Computing` 迁移代码的原则与顺序；
- 数值实验与论文内容之间的对应关系。

本项目采用“讨论—设计—小步迁移—验证—更新指南”的工作方式。未经讨论确认，不一次性复制旧代码库，也不提前引入暂时用不到的功能。

## 2. 研究目标

研究高阶 B-spline 等几何分析离散求解 Laplace 方程时，以下技术的效果及其组合：

1. 使用低阶精化（low-order refined, LOR）离散构造预条件子；
2. 使用 matrix-free 方法避免显式组装和存储高阶全局刚度矩阵；
3. 使用 sum factorization 降低高阶算子作用的计算成本；
4. 在统一、可复现的 CPU 实现中比较上述方法的数值性能。

项目服务于 `Matrix-free-IGA` 论文中的理论说明和数值实验，而不是发展成通用 IGA 软件库。

## 3. 已确定的项目边界

### 3.1 原型定位

- 只追求能够支撑论文结论的最小原型。
- 新项目保持独立、清晰、可复现，不保留旧代码中的历史实验分支。
- 优先保证数学正确性、实验可解释性和方法间公平比较。
- 优化工作以论文中需要回答的问题为导向。

### 3.2 计算平台

- 目标平台：单机 CPU。
- 可以使用 OpenMP 进行共享内存并行。
- 当前不迁移 CUDA/GPU 实现。
- 当前不以分布式 MPI 为目标。
- 若依赖 PETSc，其用途需要进一步收缩并明确；不因旧代码使用 PETSc 就默认迁移全部 PETSc/MPI 基础设施。

### 3.3 代码来源

- `Parallel-Computing` 是可参考和选择性迁移的旧代码库。
- 迁移以算法职责为单位，而不是按原目录整体复制。
- 对迁移代码进行接口简化、命名统一和最小化依赖。
- CUDA、MPI 分区、GPU ghost communication、重复的 CPU/GPU 版本及无关测试不进入首版原型。

## 4. 建议的首版数学范围（待确认）

以下是为了形成最小可验证原型而提出的初始建议，并非最终决定。

- 空间维度：二维。
- 几何：单 patch、规则张量积参数域，优先从单位正方形开始。
- 基函数：两个方向使用相同次数的开区间 B-spline，优先研究最大连续性。
- 模型问题：具有齐次 Dirichlet 边界条件的 Poisson/Laplace 问题
  \[
  -\Delta u=f \quad \text{in }\Omega,
  \qquad u=0 \quad \text{on }\partial\Omega.
  \]
- 系数：首版采用常系数；变系数和复杂几何作为后续扩展项。
- 高阶算子：同时保留显式组装参考实现和 matrix-free 实现。
- LOR：在与高阶 B-spline 自由度数一致的低阶网格上组装稀疏线性有限元刚度矩阵。

采用上述范围可以先隔离并验证 LOR、matrix-free 和 sum factorization 本身的效果，避免几何、变系数和多 patch 耦合掩盖核心结论。

## 5. 方法与实验基线（待确认）

建议至少区分以下实现或求解配置：

| 标识 | 高阶算子实现 | 预条件 | 主要用途 |
|---|---|---|---|
| `assembled` | 显式组装稀疏高阶矩阵 | 无或简单预条件 | 正确性参考与传统方法基线 |
| `mf` | 逐单元 matrix-free | 无 | 分离“避免组装”的效果 |
| `mf-sf` | matrix-free + sum factorization | 无 | 测量 sum factorization 的收益 |
| `assembled-lor` | 显式高阶矩阵 | LOR | 单独研究 LOR 对迭代的作用 |
| `mf-sf-lor` | matrix-free + sum factorization | LOR | 论文目标组合 |

是否保留 `mf` 而不使用 sum factorization 的完整求解版本，需要根据论文是否要分别量化 MF 与 SF 的贡献决定。即使不作为最终求解配置，也建议保留一个局部算子参考实现用于验证 `mf-sf`。

## 6. LOR 设计中的待讨论问题

### 6.1 低阶节点

需要确定论文比较哪些点集：

- Greville abscissae；
- 真正的 Demko abscissae；
- 其他对照点集，例如均匀点或 Chebyshev–Lobatto 点。

注意：`Parallel-Computing` 当前的 `DemkoAbscissae()` 并未实现真正的 Demko 算法，而是使用 Chebyshev–Lobatto 点近似。新项目中不能将该实现作为 Demko 点直接使用。

### 6.2 LOR 系统的求解方式

需要区分两个研究层面：

1. LOR 算子与高阶算子的谱等价性；
2. 实际求解中如何近似应用 LOR 逆算子。

候选方式包括：

- 精确稀疏直接求解，用于隔离 LOR 预条件子本身的质量；
- 不完全分解；
- AMG；
- 内层迭代法。

首版建议至少提供一种足够准确的 LOR 求解作为理论效果基线，避免内层 Jacobi/CG 的误差被误认为是 LOR 构造本身的问题。最终性能实验是否加入 AMG，需要结合依赖选择和论文重点讨论。

### 6.3 自由度对应

高阶 B-spline 空间与 LOR 空间采用相同数量、相同全局编号的系数自由度，使 LOR 预条件作用不需要额外的空间转移矩阵。实现和论文都需要清楚说明这是一种代数系数对应，并不表示两个空间的基函数相同。

## 7. 建议的软件架构（讨论稿）

首版建议采用较少层次，避免过度抽象：

```text
LOR-MF-SF-customized/
├── CMakeLists.txt
├── README.md
├── DEVELOPMENT_GUIDE.md
├── include/
│   ├── bspline.hpp
│   ├── quadrature.hpp
│   ├── tensor_mesh.hpp
│   ├── assembled_operator.hpp
│   ├── matrix_free_operator.hpp
│   ├── sum_factorized_operator.hpp
│   ├── lor_operator.hpp
│   └── solver.hpp
├── src/
├── apps/
│   ├── verify_operators.cpp
│   ├── study_spectrum.cpp
│   └── benchmark_solver.cpp
├── tests/
├── configs/
└── results/
```

目录及文件名只是职责草案，迁移前仍可调整。首版不建议复制旧仓库中多个近似相同的 `matrix_free_*` 目录；不同算法应通过清晰的算子实现和运行参数进行选择。

### 7.1 建议的模块职责

- `bspline`：一维 B-spline 基函数、导数、knot span 和必要的 extraction 数据。
- `quadrature`：一维 Gauss 积分点与权重。
- `tensor_mesh`：二维规则张量积网格、自由度编号和齐次 Dirichlet 边界。
- `assembled_operator`：组装高阶刚度矩阵，作为正确性和性能基线。
- `matrix_free_operator`：朴素逐单元算子作用，作为参考实现。
- `sum_factorized_operator`：通过逐维收缩实现高阶算子作用。
- `lor_operator`：构造低阶网格与稀疏 LOR 刚度矩阵。
- `solver`：外层 Krylov 方法以及 LOR 预条件应用。
- `apps`：将正确性验证、谱实验和端到端性能实验分开。

## 8. 外部依赖与构建配置

项目采用“核心算法代码与线性代数后端分离”的方向，并以 PETSc 作为首版求解后端：

- PETSc：首版 Krylov、稀疏矩阵和预条件器后端；
- OpenMP：可选的共享内存并行依赖；
- SLEPc：用于条件数、极端特征值和谱等价性实验，仅由谱分析程序启用，不作为普通求解与 benchmark 的强制依赖；
- Eigen：可选，不作为核心离散代码的默认依赖；
- MPI：不作为本项目的分布式计算功能使用。本机 PETSc 基于 MPICH 构建，因此使用相同的 `mpicc`/`mpicxx` wrapper 以保证 ABI 兼容，但论文原型限定为单进程运行；
- VTK、HDF5、YAML、METIS 和 CUDA：首版不作为项目直接依赖。PETSc 自身内部依赖不等同于项目直接依赖。

机器路径与通用构建逻辑分离：

- `conf/system_lib_loading.cmake`：当前 macOS/Apple Silicon 工作站的路径；
- `conf/system_lib_loading.linux.example.cmake`：根据旧 Linux 环境整理的示例；
- `conf/project_build_setup.cmake`：统一加载机器配置、选择兼容编译器并按需发现依赖；
- `conf/FindPETSc.cmake` 与 `conf/FindSLEPc.cmake`：提供 `LOR::PETSc` 和 `LOR::SLEPc` imported targets；
- 可通过 `-DSYSTEM_LIB_LOADING_FILE=/absolute/path/to/file` 选择仓库外的机器配置。

核心离散与算子类不应直接持有 PETSc 对象。PETSc 的 `Vec`、`MatShell`、`KSP` 和 `PC` 应限制在后端适配与可执行程序层，以便独立测试局部算子并保持算法代码清晰。

## 9. OpenMP 使用原则

- 首先完成正确且可重复的串行实现。
- OpenMP 优先用于单元级并行。
- matrix-free 输出向量的并发累加需要明确处理竞争问题，可比较：
  - element coloring；
  - thread-local 向量后归并；
  - 原子累加。
- 计时实验应分别报告串行和固定线程数结果，并记录线程绑定等运行设置。
- 不让 OpenMP 改变串行代码的数学结果或模糊不同算法间的比较。

## 10. 正确性验证要求

每次迁移或实现一个优化版本，都先与简单参考实现比较：

1. 一维 B-spline 基函数满足非负性、局部支撑和单位分解；
2. 基函数导数与有限差分或可信参考值一致；
3. assembled、朴素 matrix-free 和 sum-factorized 算子对同一向量的结果一致；
4. 高阶离散解具有预期的网格收敛行为；
5. LOR 矩阵对称正定；
6. 预条件与未预条件系统的残差、迭代数和谱结果可解释；
7. 性能测试前先通过小规模数值一致性测试。

数值容差、问题规模、编译选项和硬件信息应随实验结果一起保存。

## 11. 论文实验建议指标（待确认）

### 11.1 数值与谱指标

- 离散误差与收敛阶；
- 未预条件和 LOR 预条件后的条件数或极端特征值；
- 外层 Krylov 迭代次数；
- 对网格尺寸、B-spline 次数和连续性的鲁棒性；
- 不同 LOR 节点选择的影响。

### 11.2 性能指标

- 预处理/组装时间；
- 单次算子作用时间；
- 总求解时间；
- 峰值或估算内存占用；
- MF 与 MF+SF 随次数变化的复杂度趋势；
- LOR 构造及每次预条件应用的成本；
- OpenMP 加速比和并行效率（若纳入论文）。

应同时报告迭代次数和时间，避免只展示算法速度却忽略预条件质量，或只展示谱性质却忽略应用预条件子的实际成本。

## 12. 迁移阶段建议

每一阶段都应产生可运行、可验证的状态。

### 阶段 0：确定研究与依赖范围

- 确认数学模型、维度、几何和连续性范围；
- 确认 LOR 节点类型；
- 确认线性代数与求解器依赖；
- 确认论文必须包含的比较组。

### 阶段 1：最小离散核心

- 迁移并精简 B-spline、积分和规则网格功能；
- 实现小规模高阶刚度矩阵组装；
- 建立单元测试和制造解验证。

### 阶段 2：算子实现

- 实现朴素 matrix-free 算子；
- 实现 sum-factorized 算子；
- 对三种算子进行逐向量一致性验证；
- 建立算子应用时间和内存基线。

### 阶段 3：LOR

- 生成低阶节点和网格；
- 组装 LOR 刚度矩阵；
- 验证自由度对应、SPD 性质和谱行为；
- 接入外层 Krylov 求解。

### 阶段 4：性能和 OpenMP

- 在串行结果稳定后加入单元级 OpenMP；
- 固化 benchmark 配置与结果格式；
- 生成论文图表所需数据。

## 13. 首轮需要讨论并决定的事项

建议按以下顺序讨论：

1. 首版是否严格限定为二维单位正方形、单 patch、常系数 Laplace 问题？
2. B-spline 连续性是否只研究最大连续性，还是必须把不同 knot multiplicity 纳入论文？
3. LOR 节点比较是否包括 Greville 和真正的 Demko 点？
4. 是否需要把“朴素 MF”作为论文正式基线，还是只作为 SF 正确性参考？
5. 线性代数与求解器选用 PETSc、Eigen，还是采用核心独立的混合接口？
6. 为了测量 LOR 本身的质量，首版用什么方式求解低阶系统？
7. OpenMP 结果是否属于论文必须内容，还是完成串行主线后的可选增强？
8. 首篇论文实验需要覆盖哪些次数、网格尺寸和连续性组合？

## 14. 当前状态

- 已建立项目范围和初始设计讨论稿。
- 尚未迁移任何源代码。
- 已建立首版第三方库配置：PETSc 后端、可选 OpenMP/SLEPc/Eigen，以及 macOS 与 Linux 机器配置入口。
- 最终目录结构仍待确认。
- 下一步是在讨论第 13 节问题后更新本指南，再开始第一批最小迁移。
