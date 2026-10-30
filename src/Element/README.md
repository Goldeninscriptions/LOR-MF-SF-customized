# Element implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `GaussQuadrature.cpp` | 用 Newton 迭代生成任意正阶 Gauss-Legendre 点和权重 | Driver 为高阶组装、Q1 组装和误差积分分别构造规则 |
| `BSplineBasis1D.cpp` | span 搜索、DersBasisFuns 型任意阶导数递推、任意点基函数值和逐 active-span Gauss 表 | Patch 几何求值和 BSplineElement 共用同一数学基函数实现 |
| `BSplineElement.cpp` | 张量组合逐方向表，计算参数/物理梯度、坐标、Jacobian、inverse、det 和参数积分权重 | GlobalAssembly 每个高阶 element 构造一次；LocalAssembly 通过 IGAElement 接口读取 |
| `FEMQ1Basis.cpp` | 预计算 2D/3D 参考 Q1 的张量 Gauss 点、基函数、参考梯度和权重 | 一次 LOR 组装只创建一个共享表 |
| `FEMQuad4.cpp` | 从四个物理顶点计算每个 Gauss 点的双线性坐标映射、Jacobian、det 和物理梯度 | 2D LOR GlobalAssembly 构造并交给 LocalAssembly |
| `FEMHex8.cpp` | 与 Quad4 平行的八节点三线性实现 | 3D LOR GlobalAssembly 构造并交给 LocalAssembly |

这些文件都不计算 stiffness/mass/load，也不访问 IEN 或 ID。低阶 Jacobian 虽在仿射
单元上可能处处相同，接口仍按 quadrature point 提供，以兼容一般双线性/三线性
LOR cell。
