# IO public interfaces

IO 只负责数据边界：读取最小几何、转换为 Patch、解析实验参数并写结果。它不实现
refinement、Element 或 Assembly 算法。

| 文件 | 当前职责 | 主要联系 |
|---|---|---|
| `GeometryInput.hpp` | 定义不依赖 PETSc 的动态维度几何记录；声明 txt 解析入口 | 输出 dimension、逐方向 degree/knot vector 和 lexicographic control points |
| `HardwareInfo.hpp` | 声明 macOS/Linux 逻辑 CPU 检测和每 10000 自由度一个 rank 的默认策略 | 供串行 case planner 在启动 MPI 前选择 ranks |
| `DistributedGeometryInput.hpp` | 声明 root-only 文件读取和最小 GeometryInput 广播 | root 解析异常先同步给全部 ranks，避免 collective 死锁 |
| `PatchInput.hpp` | 将 `GeometryInput` 转成 `BSplinePatch<2/3>` | 只构造输入的最小几何；p/h/k 改变必须交给 Mesh refinement 类 |
| `ExperimentIO.hpp` | 定义七个完整预条件求解方案、内层 `jacobi|gamg`、固定 Jacobi 参数、三个独立功能开关、稀疏矩阵与求解成本记录，以及 `ExperimentOptions -> RefinementRequest<Dim>` 适配 | 方向由方案名推导；IO 负责字符串到类型安全 enum/固定 tensor shape 的边界转换 |

几何次数从 open knot vector 两端重复度推断。控制点使用显式 tensor index，因此在
txt 中的书写先后不影响内部 direction-0-fastest 排列。
