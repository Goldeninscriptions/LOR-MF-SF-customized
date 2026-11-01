# Implementations

`src/` 按公开接口的职责分层，不按具体算例复制实现。

| 子目录 | 当前实现范围 |
|---|---|
| `IO/` | 最小几何解析、macOS/Linux 逻辑 CPU 检测、Patch 转换、PETSc options 和结果文件 |
| `Mesh/` | Patch、细化、point set、LOR 网格、按需 IEN、ID 与 MPI slab 分区 |
| `Element/` | Gauss rule、一维 B-spline、IGA 张量积单元、Q1 单元 |
| `Assembly/` | ordering、全 Dirichlet 拓扑、非零 lift 消元、局部/全局组装及显式 SLEPc 条件数入口 |
| `Solver/` | 可选 MPI KSP/LOR 应用、带 halo 的 L2 误差与实验编排 |

当前没有 `Operator/`：matrix-free 与 sum factorization 尚未实现。各实现文件的
具体功能及其调用对象见子目录 README。
