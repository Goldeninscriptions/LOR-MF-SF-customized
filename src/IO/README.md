# IO implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `GeometryInput.cpp` | 去除 `#` 注释；解析 dimension、逐方向 knot vector 和显式索引控制点；由端点 multiplicity 推断 open-B-spline degree；验证缺失、重复、越界和非有限值 | 返回动态 `GeometryInput`，Driver 据 dimension 分派 2D/3D 模板 |
| `PatchInput.cpp` | 把动态 vectors 转为固定维度 arrays 并构造最小 `BSplinePatch<2/3>` | 不改变 degree/knot/control points；输出随后交给 refinement |
| `ExperimentIO.cpp` | 解析 `-refinement`、`-p`、`-n`、`-lor_points`、`-lor_pc`、KSP/SVD 容差和路径；写实际细化后 p/n 及六项数值结果 | 依赖 PETSc options 和 Solver 类型，由 assembled_driver 调用 |

几何 txt 中关于初始 p/n 的注释仅供用户阅读，真正数值由 knot vectors 推断。
