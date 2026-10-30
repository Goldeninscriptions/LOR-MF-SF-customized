# Mesh implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `BSplinePatch.cpp` | 验证 Patch；建立 basis shape、active spans、element spans 和 boundary multi-indices；通过一维 B-spline 值做几何单点/张量网格求值；执行 Boehm knot insertion | 所有几何和高阶拓扑的中心对象 |
| `PRefinement.cpp` | 构造提高 multiplicity 的目标 knot vectors；在目标 Greville 张量网格采样旧几何；用逐方向带主元 LU 解 target-basis transformation 得到升阶控制点 | 保持旧连续性、元素划分和几何到舍入误差 |
| `HRefinement.cpp` | 构造目标均匀 breakpoints，验证包含全部旧断点，对缺失断点调用 Patch knot insertion | 每方向目标 \(n_i\) 独立，次数不变 |
| `KRefinement.cpp` | 依次构造并调用 PRefinement 和 HRefinement | 实现经典“先升阶、后插节点”语义 |
| `LORPointSet.cpp` | 验证任意显式 point set；生成 Greville 点；实现逐方向 Greville 初始化的 Demko/Remez 插值、极值搜索与等振荡检查 | 输出参数点给 LORMesh；不计算物理几何 |
| `LORMesh.cpp` | 保存参数节点；调用 Patch tensor-grid geometry evaluation 得到全部 LOR 物理顶点 | LORIEN 与低阶 Element 读取它 |
| `IEN.cpp` | 建立 element-major 的高阶 active-basis 表和 Q1 cell-corner 表；所有 global index 使用 direction-0-fastest convention | GlobalAssembly、ErrorAnalysis 和未来 MF/SF 复用 |
| `ID.cpp` | 验证约束 global indices；建立 `global_index -> ID` 与自由 `ID -> global_index` | Assembly 用于 Dirichlet 消元和 PETSc 下标，ErrorAnalysis 用于重构零边界系数 |

`BSplinePatch::uniform_refine` 是 Patch 上的基础 knot-insertion 便利接口；实验驱动器的
正式分析空间构造使用独立的 P/H/K refinement 对象，以明确区分三种数学语义。
