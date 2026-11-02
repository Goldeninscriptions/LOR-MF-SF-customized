# Assembly implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LexicographicOrdering.cpp` | 构造 strides 和总尺寸，检查 shape/overflow，完成正逆索引映射 | 被 DirichletBC、IEN/refinement 辅助逻辑及 Solver 实验编排使用 |
| `DirichletBC.cpp` | 验证每个边界 multi-index，按边界转换 global indices，用布尔 mask 去重并排序完整约束集合 | 输出交给 Mesh 的 `ID`；不读取整个 Patch |
| `LocalAssembly.cpp` | 实现三个单项内核和共享 Gauss 数据的 stiffness+load、mass+load、stiffness+mass+load 融合内核 | 每个入口只分配被请求的局部对象，并统计 det-Jac 极值 |
| `GlobalAssembly.cpp` | 从张量支撑精确计算逐行 `d_nnz/o_nnz` 并创建 MPIAIJ；遍历最小 assembly halo，在界面单元只积分并 scatter owned test rows；融合入口复用 Element/ID row；记录 PETSc 存储统计；支持非零 Dirichlet 的局部 `-K_fc g_c` 消元；实现显式调用的 SLEPc 条件数 | 高阶路径复用一个 `BSplineElement` 工作区，LOR 路径使用共享 Q1 表；owned-row 积分与 scatter 消除 off-process row stash；并行谱分析的精确 LOR 逆使用 MUMPS |

GlobalAssembly 只组装自由 ID 对应的约化系统。普通 source-load 入口保持通用；制造解
路径额外接收完整 Dirichlet 系数，并在局部 scatter 前对自由行执行
`F_f -= K_fc g_c`。stiffness、mass 与 load 的入口完全独立，不会伴随构造未请求的
对象；需要多个项时，融合入口仍让各 PETSc 结果保持独立所有权。
