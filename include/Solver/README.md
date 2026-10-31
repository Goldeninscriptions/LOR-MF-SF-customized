# Solver public interfaces

Solver 负责线性系统求解、谱分析和离散解误差，不负责组装网格或局部矩阵。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LinearSolver.hpp` | 定义 `cg_lu`/`gmres_shell`、两层容差、外层/内层迭代统计和求解入口 | 输入 GlobalAssembly 产生的 K、K_LOR、F；返回 PETSc solution Vec |
| `ConditionNumber.hpp` | 定义极端奇异值与 2-范数条件数结果，声明带 SVD 容差、最大迭代数和子空间维度的联合分析入口 | 实现使用 SLEPc，不依赖外层 KSP 的迭代投影 |
| `ErrorAnalysis.hpp` | 声明高阶离散解物理域 L2 误差积分 | MPI 下读取共享/非本地系数，各 rank 只积分自己的元素，最后全局归约 |
| `ManufacturedSolution.hpp` | header-only 的轴对齐矩形/长方体正弦乘积精确解、对应 `-Delta u` 和适用性检查 | AssembledLaplaceExperiment 用其生成 Assembly source 和 ErrorAnalysis exact function |
| `AssembledLaplaceExperiment.hpp` | 声明完整 assembled 实验编排入口 | 接受最小几何/options；root refinement 后依次调用 Mesh、Assembly 和其他 Solver 模块 |

当前 `ManufacturedSolution` 构造时会验证 Patch 边界是否适用，仍不支持曲边 Patch。
`cg_lu` 与 `gmres_shell` 的外层算法
不同，因此二者的迭代数不能脱离残差容差直接比较。默认模式为 `gmres_shell`，两层
KSP 和 SLEPc 均继承输入矩阵的 MPI communicator。
