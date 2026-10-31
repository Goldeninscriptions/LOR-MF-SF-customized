# Mesh public interfaces

Mesh 保存单 patch 全局几何、拓扑、细化策略、LOR 节点和代数映射。它不计算 PDE
积分，也不持有 PETSc 对象。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `BSplinePatch.hpp` | 保存逐方向 degree/knot、控制点、active spans、四条边/六个面；提供几何单点/张量网格求值与 knot insertion | Basis 用于几何求值；Element、refinement、point set、DirichletBC 都读取它 |
| `PRefinement.hpp` | 把所有方向提高到同一目标次数，提高既有 knot multiplicity 并做张量 basis transformation | 输入/输出都是独立 Patch；保持元素划分、连续性和几何 |
| `HRefinement.hpp` | 对每个方向执行兼容的 Boehm knot insertion 到目标 \(n_i\) | 保持次数与已有 knot multiplicity；拒绝不包含旧断点的目标网格 |
| `KRefinement.hpp` | 组合 PRefinement 后再 HRefinement | 新插入简单 knots 在升阶空间中具有最大连续性 |
| `Refinement.hpp` | 定义不依赖 IO 字符串的 `RefinementMode/RefinementRequest` 和统一策略入口 | Solver 只在 MPI root 调用；内部选择 P/H/K 对象 |
| `DistributedPatch.hpp` | 从 root 广播 refined degree、逐向 knot vectors 和完整控制网格，在各 rank 重建相同 Patch | 当前消除重复 refinement；下一阶段再替换为局部 slab+halo 控制点 |
| `LORPointSet.hpp` | 不可变的逐方向参数点集合；具体 `GrevillePointSet` 与 Remez `DemkoPointSet` | 每方向读取自己的 knot vector；输出交给 LORMesh |
| `LORMesh.hpp` | 在 point set 上采样高阶几何得到 LOR 顶点，保存逐方向参数节点和物理节点 | LORIEN 读取 cell 拓扑；Quad4/Hex8 读取局部顶点 |
| `TensorSlabPartition.hpp` | 在 direction `Dim-1` 上均衡切分自由 ID、高阶元素和 LOR cells；保存每个 MPI rank 的半开区间 | 不改变 lexicographic/global_index/ID 编号；向 IEN 和 PETSc 局部尺寸提供一致所有权 |
| `IEN.hpp` | `IEN` 基类以及 `BSplineIEN`、`LORIEN`；按需计算 local basis/node 到 `global_index` | 不再在每个 rank 复制完整 element-major 表；Assembly 组合 IEN 与 ID |
| `ID.hpp` | 把完整 `global_index` 映射为自由代数 `ID`；Dirichlet 项为 `-1`，并提供逆映射 | 受约束列表来自 DirichletBC；高阶与 coefficient-aligned LOR 当前共享同一个 ID |

LOR 顶点位于高阶几何上，但每个 LOR cell 内部使用独立的双线性/三线性 Q1 映射；
因此 LOR 网格只用于构造预条件矩阵，不代表原高阶 Patch 几何。
