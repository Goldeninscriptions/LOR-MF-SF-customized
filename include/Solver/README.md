# Solver public interfaces

Solver 负责线性系统求解、谱分析和离散解误差，不负责组装网格或局部矩阵。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LinearSolver.hpp` | 定义 `cg_lu`/`gmres_shell`、两层容差、外层/内层迭代统计和求解入口 | 输入 GlobalAssembly 产生的 K、K_LOR、F；返回 PETSc solution Vec |
| `ConditionNumber.hpp` | 定义极端奇异值与 2-范数条件数结果，声明带 SVD 容差、最大迭代数和子空间维度的联合分析入口 | 实现使用 SLEPc，不依赖外层 KSP 的迭代投影 |
| `ErrorAnalysis.hpp` | 声明高阶离散解物理域 L2 误差积分 | 复用 Patch、BSplineIEN、ID、Gauss rule 和逐方向 Basis tables |
| `ManufacturedSolution.hpp` | header-only 的轴对齐矩形/长方体正弦乘积精确解及对应 `-Delta u` | Driver 用其生成 Assembly 的 source 和 ErrorAnalysis 的 exact function |

当前 `ManufacturedSolution` 不适用于曲边 Patch。`cg_lu` 与 `gmres_shell` 的外层算法
不同，因此二者的迭代数不能脱离残差容差直接比较。
