# Solver implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LinearSolver.cpp` | 统一 baseline 与 LOR 求解；`cg_lu` 配置外层 CG 和 LOR LU；`gmres_shell` 配置外层 restarted GMRES、PCSHELL 与内层 CG/Jacobi；记录外层/内层迭代和真实残差 | 输入为 GlobalAssembly 的 K、K_LOR、F |
| `ConditionNumber.cpp` | 用 SLEPc SVD 计算 K 的极端奇异值；用 MatShell 表示左预条件算子及其 transpose，并计算 `K_LOR^{-1}K` 的极端奇异值 | 与 KSP 迭代计数独立，LU 只用于 MatShell 中应用 LOR 逆 |
| `ErrorAnalysis.cpp` | 按 BSplineIEN/ID 重构局部高阶系数，复用 BSplineElement 积分 `sqrt(int (u_h-u)^2 dx)` | 使用独立误差 Gauss rule；Dirichlet 系数按零恢复 |

`include/Solver/ManufacturedSolution.hpp` 是 header-only：它从轴对齐矩形/长方体 Patch
边界框构造正弦乘积精确解及对应源项。Driver 另行验证每个参数边界确实对应常物理
坐标，因此该制造解当前不能用于曲边 Patch。
