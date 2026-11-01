# IO implementations

| 文件 | 当前实现 | 调用关系 |
|---|---|---|
| `GeometryInput.cpp` | 去除 `#` 注释；解析 dimension、逐方向 knot vector 和显式索引控制点；由端点 multiplicity 推断 open-B-spline degree；验证缺失、重复、越界和非有限值 | 返回动态 `GeometryInput`，Driver 据 dimension 分派 2D/3D 模板 |
| `HardwareInfo.cpp` | macOS 读取 `hw.logicalcpu`；Linux 统计当前 CPU affinity；POSIX/C++ 查询作为回退 | 输出可作为 MPI ranks 上限的逻辑处理器数和检测来源 |
| `DistributedGeometryInput.cpp` | 仅由 root 调用文本解析器，广播错误状态及最小几何的 degrees/knots/control points | 所有 ranks 在解析失败时一致退出；不执行 refinement |
| `PatchInput.cpp` | 把动态 vectors 转为固定维度 arrays 并构造最小 `BSplinePatch<2/3>` | 不改变 degree/knot/control points；输出随后交给 refinement |
| `ExperimentIO.cpp` | 解析 refinement、p/n、三个独立功能开关、求解器和容差；仅写实际执行功能的结果字段 | Solver 实验只允许 root 写结果文件；禁用量不伪装成零值 |

几何 txt 中关于初始 p/n 的注释仅供用户阅读，真正数值由 knot vectors 推断。
