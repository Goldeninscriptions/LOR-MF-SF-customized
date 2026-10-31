# Assembly implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LexicographicOrdering.cpp` | 构造 strides 和总尺寸，检查 shape/overflow，完成正逆索引映射 | 被 DirichletBC、IEN/refinement 辅助逻辑及 Driver 使用 |
| `DirichletBC.cpp` | 验证每个边界 multi-index，按边界转换 global indices，用布尔 mask 去重并排序完整约束集合 | 输出交给 Mesh 的 `ID`；不读取整个 Patch |
| `LocalAssembly.cpp` | 实现三个单项内核和共享 Gauss 数据的 stiffness+load、mass+load、stiffness+mass+load 融合内核 | 每个入口只分配被请求的局部对象，并统计 det-Jac 极值 |
| `GlobalAssembly.cpp` | 创建 MPIAIJ/VecMPI；遍历 rank-local IEN 区间；融合入口复用 Element 和 ID row；最终 collective assembly 汇总公共自由度并归约几何统计 | 高阶路径使用 `BSplineElement`，LOR 路径使用共享 `FEMQ1Basis` 与 Quad4/Hex8 |

GlobalAssembly 只组装自由 ID 对应的约化系统。由于当前边界值为零，受约束项无需
向荷载添加非零消元贡献。stiffness、mass 与 load 的入口完全独立，不会伴随构造
未请求的对象；需要多个项时，融合入口仍让各 PETSc 结果保持独立所有权。
