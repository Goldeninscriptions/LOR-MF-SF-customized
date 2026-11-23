# Solver public interfaces

Solver 负责可选线性系统求解、离散解误差和实验编排，不负责组装网格或局部矩阵。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LinearSolver.hpp` | 定义十二个方向/外层/预条件组合方案、内层 `jacobi|gamg`、AMG/Jacobi-only、固定 Jacobi 参数、两层容差及分阶段统计 | 输入 GlobalAssembly 产生的 K、可选 K_LOR、F；返回 PETSc solution Vec |
| `ErrorAnalysis.hpp` | 声明高阶离散解物理域 L2 误差积分 | MPI 下读取自由系数 halo，并从 Dirichlet lift 恢复受约束系数；各 rank 只积分自己的元素，最后全局归约 |
| `GeneralizedConditionNumber.hpp` | 声明 MPI 并行的 `EPS_GHEP` 广义谱条件数计算器及极值误差、迭代、原因和计时结果 | 直接消费分布式 SPD 矩阵 A/M；不构造逆、平方根、预条件乘积或稠密矩阵 |
| `GrevilleMassPC.hpp` | 将 `T^T M_g T` 封装为 CG 可用的 `PCShell`；逐方向执行带状 `T^{-T}`、三对角 `M_g^{-1}`、带状 `T^{-1}` | 保存逐次外层迭代、逐次 PCApply 三阶段时间/线求解数及因子、工作区、算子内存统计；当前单 rank |
| `DistributedGrevilleMassPC.hpp` | 提供二维 Greville mass 预条件器的两种 MPI 实现：两次全局 pencil transpose，或沿分区方向传递带宽边界的分布式带状求解 | 提供 spline mass 与 `T^T M_g T` 的矩阵自由张量 `MatShell`；可给 EPS/ST 的 KSP 安装精确 tensor-transpose 逆，并统计应用成本 |
| `ManufacturedSolution.hpp` | header-only 的物理解 `u=1+sum(x_d)+bubble(x)`、解析源项和精确仿射边界系数 | 轴对齐几何使用张量正弦 bubble，曲边基准使用四个边界多项式因子；实验编排用其生成 source、非零 lift 和 L2 exact function |
| `AssembledLaplaceExperiment.hpp` | 声明完整 assembled 实验编排入口 | 接受最小几何/options；root refinement 后依次调用 Mesh、Assembly 和其他 Solver 模块 |

SVD 条件数接口位于 `Assembly/GlobalAssembly.hpp`；广义特征值条件数接口位于
`Solver/GeneralizedConditionNumber.hpp`。二者都只在对应开关启用时由实验编排显式
调用。制造解的仿射边界迹通常
非零，但通过几何控制点得到的系数可被 B-spline 精确再现；零迹 bubble 只改变域内
解和源项，不增加边界投影误差。
不同方案的外层算法和实际近似逆不同，因此迭代数不能脱离方案与残差容差直接比较。
默认 `left_gmres_fixed_jacobi` 对固定线性映射建立标准 Arnoldi 关系；
`right_fgmres_cg` 处理变化的内层 CG；`right_fgmres_amg` 在预条件矩阵上复用
一次 GAMG PCApply，不启动内层 CG。`right_fgmres_jacobi` 同样直接作用于所选
预条件矩阵，但应用 `diag(P)^{-1}`，不是 `diag(K)^{-1}`。三个 `*_cg` 方案的外层和内层默认均使用
Parallel-Computing 实际生效的 `rtol=1e-10, atol=1e-10`。两层 KSP 继承输入矩阵的 MPI
communicator，结果文件同时记录求解后重算的绝对和相对真实残差。
其中外层时间是完整 `KSPSolve`，已经包含所有内层 CG 调用；内层累计时间用于解释
预条件器成本，不能再次加到外层时间上。并行计时取最大 rank 的 wall time。
GAMG 层次构造归入 setup 时间，并在同一次外层求解的所有内层调用中复用。
`left_gmres_amg` 在同一 selected P 上复用直接 GAMG，但使用左预条件普通 GMRES；
其停止判断对应预条件残差，因此必须结合输出中的原系统真实残差与 L2 误差分析。
固定 Jacobi 默认执行 `m=4, omega=2/3`，其总步数和应用时间使用独立字段输出。
`left_gmres_k_jacobi` 使用 PETSc `PCJACOBI` 直接预条件 K，不启动 LOR 逆应用。
`left_gmres_k_gamg` 与 `right_gmres_k_gamg` 使用 PETSc `PCGAMG` 直接从 K 构造层次，
并分别采用左、右 GMRES；这两个参考模式不组装 K_LOR。
