# Public interfaces

`include/` 保存可被主程序和其他模块调用的公开接口。当前依赖方向为
`IO/Mesh -> Element -> Assembly -> Solver`；Mesh 和 Element 不依赖 PETSc。

| 子目录 | 公开对象 | 与其他模块的关系 |
|---|---|---|
| `IO/` | `GeometryInput`、`ExperimentOptions`、Patch 转换与结果输出 | 把文本/PETSc options 转成 Mesh 和 Solver 可消费的数据 |
| `Mesh/` | Patch、三种 refinement、point set、LOR mesh、IEN、ID | 保存全局拓扑、几何和代数编号，不计算单元积分 |
| `Element/` | B-spline/Q1 基函数和高阶/低阶单元积分点数据 | 从 Mesh gather 几何，由 Assembly 只读消费 |
| `Assembly/` | ordering、Dirichlet、局部与 PETSc 全局组装 | 组合 Mesh 的映射和 Element 的积分数据 |
| `Solver/` | KSP、条件数、误差和制造解接口 | 消费 Assembly 产生的系统和高阶离散信息 |

具体到每个头文件的职责见相应子目录 README。接口设计的关键约束是：Element
不持有 IEN/ID 或矩阵，index 转换只由 `LexicographicOrdering` 完成，PETSc 对象只从
GlobalAssembly/Solver 层出现。
