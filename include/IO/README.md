# IO public interfaces

IO 只负责数据边界：读取最小几何、转换为 Patch、解析实验参数并写结果。它不实现
refinement、Element 或 Assembly 算法。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `GeometryInput.hpp` | 定义不依赖 PETSc 的动态维度几何记录；声明 txt 解析入口 | 输出 dimension、逐方向 degree/knot vector 和 lexicographic control points |
| `DistributedGeometryInput.hpp` | 声明 root-only 文件读取和最小 GeometryInput 广播 | root 解析异常先同步给全部 ranks，避免 collective 死锁 |
| `PatchInput.hpp` | 将 `GeometryInput` 转成 `BSplinePatch<2/3>` | 只构造输入的最小几何；p/h/k 改变必须交给 Mesh refinement 类 |
| `ExperimentIO.hpp` | 定义 PETSc 选项、结果和 `ExperimentOptions -> RefinementRequest<Dim>` 适配 | IO 负责字符串到 Mesh enum/固定 tensor shape 的边界转换 |

几何次数从 open knot vector 两端重复度推断。控制点使用显式 tensor index，因此在
txt 中的书写先后不影响内部 direction-0-fastest 排列。
