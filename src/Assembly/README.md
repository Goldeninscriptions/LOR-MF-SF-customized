# Assembly implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `LexicographicOrdering.cpp` | 构造 strides 和总尺寸，检查 shape/overflow，完成正逆索引映射 | 被 DirichletBC、IEN/refinement 辅助逻辑及 Driver 使用 |
| `DirichletBC.cpp` | 验证每个边界 multi-index，按边界转换 global indices，用布尔 mask 去重并排序完整约束集合 | 输出交给 Mesh 的 `ID`；不读取整个 Patch |
| `LocalAssembly.cpp` | 一个通用预计算单元积分内核；为 IGA、Quad4、Hex8 重载，在一次 Gauss 循环中填 stiffness/mass/load 和 det-Jac 极值 | 输入来自 Element；结果被 GlobalAssembly scatter |
| `GlobalAssembly.cpp` | 创建/预分配 PETSc Mat/Vec；gather 高阶控制点或 LOR 顶点；逐单元构造 Element、调用 LocalAssembly、经 IEN+ID scatter 并 finalize | 高阶路径使用 `BSplineElement`，LOR 路径使用共享 `FEMQ1Basis` 与 Quad4/Hex8 |

GlobalAssembly 只组装自由 ID 对应的约化系统。由于当前边界值为零，受约束项无需
向荷载添加非零消元贡献。
