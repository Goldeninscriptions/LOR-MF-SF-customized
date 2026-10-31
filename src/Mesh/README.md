# Mesh implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `BSplinePatch.cpp` | 验证 Patch；建立 basis shape、active spans、element spans 和 boundary multi-indices；通过一维 B-spline 值做几何单点/张量网格求值；执行 Boehm knot insertion | 所有几何和高阶拓扑的中心对象 |
| `PRefinement.cpp` | 对每条张量 fiber 执行《The NURBS Book》Algorithm A5.9：Bezier 分段升阶并移除临时分解 knots，再逐方向重组控制网格 | 不依赖配点线性求解；保持旧连续性、元素划分和几何到舍入误差 |
| `HRefinement.cpp` | 构造目标均匀 breakpoints，验证包含全部旧断点，对缺失断点调用 Patch knot insertion | 每方向目标 \(n_i\) 独立，次数不变 |
| `KRefinement.cpp` | 依次构造并调用 PRefinement 和 HRefinement | 实现经典“先升阶、后插节点”语义 |
| `Refinement.cpp` | 验证共同次数以及纯 p/h 的不变量，统一分派 P/H/K refinement | 不读取 `ExperimentOptions`，保持 Mesh 与 IO 解耦 |
| `DistributedPatch.cpp` | 由 root 序列化并广播 refined Patch，在每个 rank 通过构造器重建和验证拓扑 | 一次 refinement、全量 Patch 复制的第一阶段 MPI 实现 |
| `LORPointSet.cpp` | 验证任意显式 point set；生成 Greville 点；实现逐方向 Greville 初始化的 Demko/Remez 插值、极值搜索与等振荡检查 | 输出参数点给 LORMesh；不计算物理几何 |
| `LORMesh.cpp` | 保存参数节点；调用 Patch tensor-grid geometry evaluation 得到全部 LOR 物理顶点 | LORIEN 与低阶 Element 读取它 |
| `TensorSlabPartition.cpp` | 用 `floor(rank*N/P)` 对最慢方向做无重叠 slab 分割；高阶元素、LOR cells 和自由 ID 各自按实际层数均衡划分 | 允许不能整除的规模；保持完整覆盖及连续 PETSc 行所有权 |
| `IEN.cpp` | 按输入的全局 element/cell index 即时计算高阶 active basis 或 Q1 corner 的 global index | MPI rank 只遍历 Partition 分配的区间，避免复制巨大 IEN 表 |
| `ID.cpp` | 验证约束 global indices；建立 `global_index -> ID` 与自由 `ID -> global_index` | Assembly 用于 Dirichlet 消元和 PETSc 下标，ErrorAnalysis 用于重构零边界系数 |

`BSplinePatch::uniform_refine` 是 Patch 上的基础 knot-insertion 便利接口；实验驱动器的
正式分析空间构造使用独立的 P/H/K refinement 对象，以明确区分三种数学语义。
