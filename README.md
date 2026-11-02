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
- 张量支撑驱动的精确 MPIAIJ 逐行预分配、owned-row halo 组装和矩阵存储日志；
- 几何文件与 p/h/k refinement 仅在 rank 0 执行，refined Patch 广播重建；
- 六个完整求解方案的名称同时编码预条件方向、外层方法和 LOR 逆应用；默认
  `left_gmres_fixed_jacobi` 是固定线性映射；
- 三个 `*_cg` 方案的内层 CG 可选择 legacy `jacobi` 或 PETSc `gamg`；
- 求解输出同时记录 setup、完整外层时间，以及嵌套预条件操作的调用数、
  总/平均迭代数和总/平均时间；
- 求解、L2 误差和 SLEPc 条件数可独立启用，禁用项不计算也不输出；
- 串行 case planner 自动计算细化后自由度，并在 macOS/Linux 上按逻辑 CPU 数推荐 MPI ranks；
- 单算例与自动选择 MPI ranks 的顺序批处理脚本。
- 固定 case 的内层 CG 容差扫描及迭代、时间、真实残差、L2 误差 TSV 汇总。

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
  greville left_gmres_fixed_jacobi \
  work/results/example.log
```

单算例的第一个参数可写成 `default`，也可由用户强制指定，例如 `8`：

```bash
./work/run_case.sh 8 work/input/square_2d.txt \
  k_refine 5 16,16 greville left_gmres_fixed_jacobi work/results/example_mpi8.log
```

批处理预览和正式运行：

```bash
BATCH_DRY_RUN=1 ./work/run_batch_square_2d.sh
./work/run_batch_square_2d.sh
```

每个几何批处理脚本都列出了六个完整求解方案。需要方案 A1 时，在脚本配置区注释
当前 `left_gmres_fixed_jacobi` 块，并取消注释 `right_fgmres_cg` 块及其内层参数。

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
  k_refine 5 16,16 greville left_gmres_fixed_jacobi work/results/example.log
```

批处理在配置区直接设置同名小写变量。L2 要求先启用 solution；条件数与二者独立，
默认关闭以避免普通批次隐式执行昂贵的四次 SVD。

位置参数 `LOR_PC` 接受 `left_cg_lu`、`right_gmres_lu`、`left_gmres_cg`、
`right_gmres_cg`、`right_fgmres_cg` 和默认的
`left_gmres_fixed_jacobi`。方向由名称唯一确定；结果仍会记录
`preconditioner_side` 便于筛选。所有模式结束后都重算原方程真实残差；
`run_case.sh` 末尾四个可选位置参数用于三个 `*_cg` 方案。
每个方案只接收并记录自身有效的参数：LU 结果中没有 inner/fixed 字段，`*_cg` 没有
fixed 字段，固定 Jacobi 也没有 inner CG 字段。

方案 B 默认固定执行 `m=4, omega=2/3` 的 weighted Jacobi。单算例可用
`FIXED_JACOBI_STEPS` 和 `FIXED_JACOBI_OMEGA` 覆盖；批处理直接编辑已暴露的方案块。

三个 `*_cg` 方案的样例默认写为 `inner_pc="jacobi"`；要使用 PETSc GAMG，将所选
方案块中的值改成 `inner_pc="gamg"`。

GAMG 层次在 setup 阶段构造一次并在所有内层调用中复用。性能比较应使用
`setup_time_seconds + outer_solve_time_seconds`，不能只看内层迭代数。

结果中的 `outer_solve_time_seconds` 是完整外层 `KSPSolve` 时间，已经包含其间所有
内层 CG 调用；`inner_solve_time_seconds` 是这段时间内内层求解的累计子成本，二者
不能相加。MPI 计时采用所有 ranks 中的最大 wall time，代表该阶段的关键路径。

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
