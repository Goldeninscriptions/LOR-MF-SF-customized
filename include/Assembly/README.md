# Assembly public interfaces

Assembly 把与方程无关的 Element 数据转换为 Laplace 离散系统，并负责约束与全局
编号的组合。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LexicographicOrdering.hpp` | 定义唯一的 `multi_index <-> global_index` 正逆映射；direction 0 最快 | Patch 边界、IEN、DirichletBC 和 refinement 中的张量排列都应调用它 |
| `DirichletBC.hpp` | 将四条边/六个面上的 boundary `multi_index` 转成分边界及去重后的受约束 `global_index` | 只描述约束拓扑并输出给 `ID`；非零系数由独立 Dirichlet lift 保存 |
| `LocalAssembly.hpp` | 定义独立的局部矩阵/向量结果，提供三个单项入口和三种含 load 的融合入口 | 只读取 Element 的 `N`、梯度、坐标、权重和 Jacobian determinant |
| `PreconditionerSide.hpp` | 定义论文默认的左预条件与可选右预条件代数方向 | 被 Solver、IO 与条件数分析共同使用，避免方向标签不一致 |
| `GlobalAssembly.hpp` | 定义分布式 PETSc 矩阵/向量结果、高阶/LOR 单项及融合入口、矩阵存储统计，以及显式调用的 SLEPc 条件数分析 | 遍历 IEN 的 assembly halo，但只插入 owned rows；条件数分析不由组装隐式触发 |

`quad_weight` 和 `det_Jac` 在 Element 接口中分开，LocalAssembly 显式计算二者乘积。
三个物理量互不隐式依赖：调用 `assemble_global_stiffness` 不会创建 mass 或 load，
其余两个入口同理。多个物理量会按用户的显式调用分别遍历单元；当前没有
matrix-free 算子接口。若调用者同时需要多个项，可显式选择
`assemble_global_stiffness_load`、`assemble_global_mass_load` 或
`assemble_global_stiffness_mass_load`，从而只遍历一次单元和 Gauss 点。
`assemble_global_stiffness_load_with_dirichlet` 额外接收完整 prescribed coefficient
向量，并在局部 scatter 前把 `-K_fc g_c` 加入自由右端。
`compute_stiffness_condition_numbers` 是独立且昂贵的后处理入口，只有调用者显式请求
时才创建四个 SVD 求解和预条件算子；它可分析精确 LOR 逆或直接的
`diag(K)^{-1}` 左预条件参考算子。
高阶和 LOR 矩阵都从张量支撑直接计算逐 owned-row 的精确 `d_nnz/o_nnz`，不创建旧
代码使用的临时 pattern 矩阵。结果同时保存 predicted/used/allocated nnz、PETSc
malloc 次数、PETSc memory 报告和不含 allocator 开销的可移植最低字节估计。
高阶接口还提供 row-filtered stiffness/load 内核：MPI halo 单元只计算当前 rank 拥有的
test rows，同时保留所有 trial columns，以正确完成跨分区耦合及非零 Dirichlet 消元。
