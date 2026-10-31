# Solver implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LinearSolver.cpp` | 统一 baseline 与 LOR 求解；`cg_lu` 配置外层 CG 和 LOR LU；`gmres_shell` 配置外层 restarted GMRES、PCSHELL 与内层 CG/Jacobi；记录外层/内层迭代和真实残差 | 输入为 GlobalAssembly 的 K、K_LOR、F |
| `ConditionNumber.cpp` | 在矩阵 MPI communicator 上用 SLEPc SVD 分析 K 和 MatShell `K_LOR^{-1}K`；并行精确 LOR 逆使用 MUMPS | 与外层 KSP 迭代计数独立；并行 PETSc 必须带 MUMPS 才能计算预条件后真 2-范数条件数 |
| `ErrorAnalysis.cpp` | 把本 slab 所需的连续 ID 区间 scatter 到局部 halo 向量，各 rank 只积分自己的 BSplineIEN 元素，再对误差平方做 MPI sum | 正确处理跨 slab 单元需要的公共/非本地系数，不复制完整全局解 |
| `AssembledLaplaceExperiment.cpp` | root-only 构造/细化 Patch并同步错误，广播 refined Patch，随后编排边界、LOR、MPI 分区、组装、两次求解、误差、条件数和输出 | 把端到端用例逻辑从主程序移出；调用模块公开接口而不复制其算法 |

`include/Solver/ManufacturedSolution.hpp` 是 header-only：它从轴对齐矩形/长方体 Patch
边界框构造正弦乘积精确解及对应源项，并自行验证每个参数边界对应常物理坐标；
因此该制造解当前不能用于曲边 Patch。
