# Implementations

`src/` 按公开接口的职责分层，不按具体算例复制实现。

| 子目录 | 当前实现范围 |
|---|---|
| `IO/` | 最小几何解析、Patch 转换、PETSc options 和结果文件 |
| `Mesh/` | Patch、细化、point set、LOR 网格、IEN、ID |
| `Element/` | Gauss rule、一维 B-spline、IGA 张量积单元、Q1 单元 |
| `Assembly/` | ordering、全 Dirichlet、局部积分和 PETSc 全局散射 |
| `Solver/` | KSP/LOR 应用、SLEPc 条件数和 L2 误差 |

当前没有 `Operator/`：matrix-free 与 sum factorization 尚未实现。各实现文件的
具体功能及其调用对象见子目录 README。
