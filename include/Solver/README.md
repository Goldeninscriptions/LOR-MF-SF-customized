# Solver public interfaces

Solver 负责可选线性系统求解、离散解误差和实验编排，不负责组装网格或局部矩阵。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LinearSolver.hpp` | 定义 `cg_lu`/`gmres_shell`、独立 `left|right` 方向、两层容差、迭代统计和求解入口 | 输入 GlobalAssembly 产生的 K、K_LOR、F；返回 PETSc solution Vec |
| `ErrorAnalysis.hpp` | 声明高阶离散解物理域 L2 误差积分 | MPI 下读取自由系数 halo，并从 Dirichlet lift 恢复受约束系数；各 rank 只积分自己的元素，最后全局归约 |
| `ManufacturedSolution.hpp` | header-only 的物理解 `u=1+sum(x_d)+bubble(x)`、解析源项和精确仿射边界系数 | 轴对齐几何使用张量正弦 bubble，曲边基准使用四个边界多项式因子；实验编排用其生成 source、非零 lift 和 L2 exact function |
| `AssembledLaplaceExperiment.hpp` | 声明完整 assembled 实验编排入口 | 接受最小几何/options；root refinement 后依次调用 Mesh、Assembly 和其他 Solver 模块 |

条件数接口已归入 `Assembly/GlobalAssembly.hpp`，且只有
`compute_condition_number=true` 时由实验编排显式调用。制造解的仿射边界迹通常
非零，但通过几何控制点得到的系数可被 B-spline 精确再现；零迹 bubble 只改变域内
解和源项，不增加边界投影误差。
`cg_lu` 与 `gmres_shell` 的外层算法不同，因此二者的迭代数不能脱离残差容差直接
比较。默认模式为 `gmres_shell + left`，与论文和 Parallel-Computing 的
`K_LOR^{-1}K` 定义一致；`right` 作为独立研究选项。外层和内层默认均使用
`rtol=1e-12, atol=1e-14`。两层 KSP 继承输入矩阵的 MPI
communicator，结果文件同时记录求解后重算的绝对和相对真实残差。
