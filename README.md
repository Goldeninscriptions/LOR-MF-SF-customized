# LOR-MF-SF customized

面向 `Matrix-free-IGA` 论文的 CPU/MPI 最小原型。当前版本已经贯通普通 B-spline
高阶 assembled Laplace 系统、Greville/Demko LOR 网格、Q1 LOR 预条件矩阵、两种
KSP 应用方式、SLEPc 条件数和 L2 误差分析。Matrix-free 与 sum factorization
尚未实现，当前 assembled 路径是后续优化的正确性基线。

## 当前能力

- 2D/3D、单 patch、普通非有理 B-spline 几何；
- 各方向相同次数、独立 knot vector 和独立 \(n_i\)；
- 数学正确的 p-/h-/k-refinement；
- 全边界 Dirichlet 约束；制造解使用精确非零仿射 lift，并在域内叠加正弦/曲边多项式 bubble；
- 高阶与 LOR Q1 的刚度、质量、荷载独立按需组装；
- `greville|demko` LOR point set；
- MPI 方向切片并行组装、分布式 PETSc 矩阵/向量和并行 KSP；
- 几何文件与 p/h/k refinement 仅在 rank 0 执行，refined Patch 广播重建；
- 默认高精度右预条件 `gmres_shell`，并保留 `cg_lu` 精确研究模式；
- 求解/迭代数、L2 误差和 SLEPc 条件数可独立启用，禁用项不计算也不输出；
- 串行 case planner 自动计算细化后自由度，并在 macOS/Linux 上按逻辑 CPU 数推荐 MPI ranks；
- 单算例与自动选择 MPI ranks 的顺序批处理脚本。

当前不支持 NURBS、周期边界、多 patch、MF/SF 算子或从输入文件指定任意边界函数。

## 构建

```bash
mkdir -p build
cd build
cmake ../examples -DCMAKE_BUILD_TYPE=Release
make -j
```

可执行文件为 `build/assembled_driver` 和 `build/case_planner`。新机器首先维护
`conf/system_lib_loading.cmake`，或用 `SYSTEM_LIB_LOADING_FILE` 指定外部配置。

## 运行

```bash
./work/run_case.sh \
  default \
  work/input/square_2d.txt \
  k_refine 5 16,16 \
  greville gmres_shell \
  work/results/example.log
```

单算例的第一个参数可写成 `default`，也可由用户强制指定，例如 `8`：

```bash
./work/run_case.sh 8 work/input/square_2d.txt \
  k_refine 5 16,16 greville gmres_shell work/results/example_mpi8.log
```

批处理预览和正式运行：

```bash
BATCH_DRY_RUN=1 ./work/run_batch_square_2d.sh
./work/run_batch_square_2d.sh
```

另外两个几何分别使用 `run_batch_cube_3d.sh` 和
`run_batch_curved_quadrilateral_2d.sh`；三个入口共享参数验证与日志执行器，但结果、
控制台日志和 summary 按几何分目录保存。

在不启动 MPI 的情况下查看细化后的精确自由度和默认 ranks：

```bash
./build/case_planner \
  -geometry work/input/square_2d.txt \
  -refinement k_refine -p 5 -n 128,128
```

默认策略为每 10000 个全 Dirichlet 自由度使用一个 rank，向上取整且不超过
检测到的逻辑 CPU 数。`--max-ranks M` 可进一步限制单次规划。批处理配置区接受
`mpi_rank=default` 或 `mpi_rank=8`；也可用 `MPI_RANK=8` 临时覆盖。仅在 default
模式下，环境变量 `MAX_MPI_RANKS=M` 才作为自动规划的额外上限。

主程序在 MPI 初始化后立即向控制台日志打印 `actual_mpi_ranks`；成功结束时再次打印
完成摘要，结果文件中同时保存 `mpi_ranks = ...`。

单算例通过环境变量选择独立功能：

```bash
COMPUTE_SOLUTION=true \
COMPUTE_L2_ERROR=true \
COMPUTE_CONDITION_NUMBER=false \
./work/run_case.sh default work/input/square_2d.txt \
  k_refine 5 16,16 greville gmres_shell work/results/example.log
```

批处理在配置区直接设置同名小写变量。L2 要求先启用 solution；条件数与二者独立，
默认关闭以避免普通批次隐式执行昂贵的四次 SVD。

`gmres_shell` 默认让外层和内层分别使用 `rtol=1e-12, atol=1e-14`，外层以原方程
未预条件残差作为停止判据。结果文件会记录求解后重算的绝对残差和相对残差；
`run_case.sh` 末尾四个可选位置参数仍可覆盖两层容差。

## 顶层文件和目录

| 路径 | 当前职责 |
|---|---|
| `.gitignore` | 忽略 `.DS_Store` 和整个 `build/` 目录 |
| `CMakeLists.txt` | 定义 `lor_core`、`lor_petsc`、`lor_solver` 和 examples 构建入口 |
| `DEVELOPMENT_GUIDE.md` | 当前数学范围、真实开发状态、架构约束与下一阶段 |
| `conf/` | 本机/示例机器配置以及 PETSc、SLEPc 查找逻辑 |
| `include/` | IO、Mesh、Element、Assembly、Solver 的公开接口 |
| `src/` | 上述接口的实现 |
| `examples/` | assembled 实验驱动器和串行 case planner |
| `work/` | 最小几何输入、单算例脚本、批处理脚本和运行产物 |

每个代码文件的职责和对象关系见各目录 README。完整开发现状见
`DEVELOPMENT_GUIDE.md`。
