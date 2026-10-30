# Element public interfaces

Element 只描述单元局部基函数和几何积分数据，不拥有 Mesh、IEN、ID 或离散矩阵。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `GaussQuadrature.hpp` | `QuadratureRule1D` 和任意阶 Gauss-Legendre 规则入口 | 高阶与 Q1 表共享同一种一维规则表示 |
| `BSplineBasis1D.hpp` | 一维 B-spline 即时 `evaluate_N`；按 span/Gauss/local-basis 排列的 `N_Gauss` 和 `N_der1_Gauss` | Patch 几何求值使用即时值；BSplineElement 使用预计算表；高阶导数递推保留为 private |
| `IGAElement.hpp` | 2D/3D IGA 单元只读父类；统一暴露 `N`、参数/物理梯度、三种坐标、Jacobian、`det_Jac`、`quad_weight` | LocalAssembly 只依赖该抽象接口；未来 NURBS 可实现同一契约 |
| `BSplineElement.hpp` | 一个 active knot-span 的具体张量积 B-spline 单元 | 从 Patch、局部控制点、Gauss rule 和逐方向 Basis 表构造父类积分点数据 |
| `FEMQ1Basis.hpp` | Quad4/Hex8 共享的 `[-1,1]^Dim` Q1 参考基函数、参考导数和张量权重表 | 每次 LOR 全局组装只构造一次，由所有物理低阶单元引用 |
| `FEMQuad4.hpp` | 四节点双线性物理单元，局部节点逆时针 | 输入 LORMesh 顶点和共享 Q1 表；向 LocalAssembly 暴露与 IGA 相近的接口 |
| `FEMHex8.hpp` | 八节点三线性物理单元，0--3 下表面、4--7 对应上表面 | 3D 版本的 LOR 物理单元，与 Quad4 使用平行的数据结构和命名 |

高阶 `quad_weight` 包含参考区间到 knot span 的尺度但不含 `det_Jac`；低阶
`quad_weight` 是参考 Q1 单元上的张量 Gauss 权重，同样不含 `det_Jac`。
