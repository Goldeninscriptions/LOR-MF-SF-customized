# Assembly public interfaces

Assembly 把与方程无关的 Element 数据转换为 Laplace 离散系统，并负责约束与全局
编号的组合。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `LexicographicOrdering.hpp` | 定义唯一的 `multi_index <-> global_index` 正逆映射；direction 0 最快 | Patch 边界、IEN、DirichletBC 和 refinement 中的张量排列都应调用它 |
| `DirichletBC.hpp` | 将四条边/六个面上的 boundary `multi_index` 转成分边界及去重后的受约束 `global_index` | 输入只依赖 Patch 边界数据和外部 Ordering；输出交给 `ID` |
| `LocalAssembly.hpp` | 定义 `ScalarSource` 与 `LocalAssemblyResult`，为 IGA、Quad4、Hex8 统一计算局部刚度、质量、荷载 | 只读取 Element 的 `N`、梯度、坐标、权重和 Jacobian determinant |
| `GlobalAssembly.hpp` | 定义 PETSc `GlobalAssemblyResult`，提供高阶 Patch 与 LORMesh 两套全局组装重载 | 通过 IEN gather/scatter，通过 ID 消去 Dirichlet 项，内部调用 LocalAssembly |

`quad_weight` 和 `det_Jac` 在 Element 接口中分开，LocalAssembly 显式计算二者乘积。
GlobalAssembly 同一次单元遍历组装 stiffness、mass 和 load；当前没有 matrix-free
算子接口。
