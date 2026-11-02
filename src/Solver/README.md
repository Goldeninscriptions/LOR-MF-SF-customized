# Solver implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LinearSolver.cpp` | 统一 baseline 与六个 LOR 方案；精确 LU、标准 GMRES+内层 CG、右 FGMRES+内层 CG及左 GMRES+固定 weighted Jacobi 共用统计入口 | 输入为 GlobalAssembly 的 K、K_LOR、F |
| `ErrorAnalysis.cpp` | 把本 slab 所需的连续 ID 区间 scatter 到局部 halo 向量，从独立 lift 恢复非零边界系数，各 rank 只积分自己的 BSplineIEN 元素，再对误差平方做 MPI sum | 正确处理跨 slab 单元与非齐次 Dirichlet 值，不复制完整全局解 |
| `AssembledLaplaceExperiment.cpp` | root-only 构造/细化并广播 Patch，随后编排边界、LOR、MPI 分区和组装；按三个独立开关调用两次求解、L2 与 GlobalAssembly 条件数入口 | 关闭功能时不构造其荷载、解向量、误差表或谱对象 |

`include/Solver/ManufacturedSolution.hpp` 是 header-only：它定义
`u(x)=1+sum_d x_d+bubble(x)`。轴对齐几何使用物理坐标张量正弦 bubble；当前双三次
曲边基准使用四个精确边界因子的乘积，并按乘积法则解析计算 Laplacian。bubble 在
边界为零，非零仿射迹仍通过几何控制点精确生成 B-spline 系数。

三个 `*_cg` 方案在每次 PCSHELL 应用中累计内层 CG 的迭代与 wall time；固定 Jacobi
累计确定的 polynomial steps。完整外层时间已包含这些调用，不能重复相加。
GAMG 层次由显式 `KSPSetUp(inner_ksp)` 一次构造，其成本计入预条件求解的 setup。
