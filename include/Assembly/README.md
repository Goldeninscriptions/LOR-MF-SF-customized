# Assembly public interfaces

Assembly 把与方程无关的 Element 数据转换为 Laplace 离散系统，并负责约束与全局
编号的组合。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LexicographicOrdering.hpp` | 定义唯一的 `multi_index <-> global_index` 正逆映射；direction 0 最快 | Patch 边界、IEN、DirichletBC 和 refinement 中的张量排列都应调用它 |
| `DirichletBC.hpp` | 将四条边/六个面上的 boundary `multi_index` 转成分边界及去重后的受约束 `global_index` | 输入只依赖 Patch 边界数据和外部 Ordering；输出交给 `ID` |
| `LocalAssembly.hpp` | 定义独立的局部矩阵/向量结果，提供三个单项入口和三种含 load 的融合入口 | 只读取 Element 的 `N`、梯度、坐标、权重和 Jacobian determinant |
| `GlobalAssembly.hpp` | 定义分布式 PETSc 矩阵/向量结果，并为高阶 Patch 与 LORMesh 重载单项及融合入口 | 遍历 IEN 的 rank-local 区间；公共 DOF 用 `ADD_VALUES` 交给 PETSc 汇总 |

`quad_weight` 和 `det_Jac` 在 Element 接口中分开，LocalAssembly 显式计算二者乘积。
三个物理量互不隐式依赖：调用 `assemble_global_stiffness` 不会创建 mass 或 load，
其余两个入口同理。多个物理量会按用户的显式调用分别遍历单元；当前没有
matrix-free 算子接口。若调用者同时需要多个项，可显式选择
`assemble_global_stiffness_load`、`assemble_global_mass_load` 或
`assemble_global_stiffness_mass_load`，从而只遍历一次单元和 Gauss 点。
