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
- 十个完整求解方案的名称同时编码预条件方向、外层方法和预条件器；默认
  `left_gmres_fixed_jacobi` 是固定线性映射；
- 三个 `*_cg` 方案的内层 CG 可选择 legacy `jacobi` 或 PETSc `gamg`；
- 求解输出同时记录 setup、完整外层时间，以及嵌套预条件操作的调用数、
  总/平均迭代数和总/平均时间；
- 求解、L2 误差和 SLEPc 条件数可独立启用，禁用项不计算也不输出；
- 串行 case planner 自动计算细化后自由度，并在 macOS/Linux 上按逻辑 CPU 数推荐 MPI ranks；
- 单算例与自动选择 MPI ranks 的顺序批处理脚本。
- 固定 case 的内层 CG 容差扫描及迭代、时间、真实残差、L2 误差 TSV 汇总。

当前不支持 NURBS、周期边界、多 patch、MF/SF 算子或从输入文件指定任意边界函数。

## 可选 truncated-Cholesky tensor 预条件矩阵

默认矩阵来源仍是 `lor_q1`，所有既有命令保持原行为。轴对齐仿射
`square_2d`/`cube_3d` 可显式选择新来源：

```bash
PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor \
TRUNC_CHOL_K=4 COMPUTE_TRUNCATION_CERTIFICATE=true \
COMPUTE_SPECTRAL_CONDITION_NUMBER=true COMPUTE_CONDITION_NUMBER=true \
./work/run_case.sh 1 work/input/square_2d.txt k_refine 8 16,16 \
  greville left_cg_lu work/results/p8_n16_trunc_k4.log
```

`TRUNC_CHOL_K=auto`（默认）使用 `ceil(p/2)`。新路径要求 `k` 同时严格
小于两类质量矩阵的实际半带宽，所以 `p=3` 必须显式使用例如
`TRUNC_CHOL_K=1`。实现先做完整带状 Cholesky，再按边界保护规则截断
因子并形成 `L_tilde L_tilde^T`；不截断原质量矩阵，也不建立稠密采样映射。

直接调用 driver 时使用
`-preconditioner_matrix_source`、`-trunc_chol_k`、
`-compute_truncation_certificate`、`-verify_tensor_identity`、
`-trunc_chol_max_memory_bytes`、`-compute_spectral_condition_number` 以及
`-eps_tol/-eps_max_it/-eps_ncv`。原 `-compute_condition_number` 仍是 SVD
奇异值条件数 `kappa_sigma`；EPS 独立报告广义 SPD 问题的
`kappa_lambda`。新来源会拒绝曲边几何和直接 K-Jacobi/K-GAMG 模式，旧
Greville/Demko 曲边路径不受影响。

`-verify_tensor_identity` 默认关闭；开启时会独立计时并将完整（未截断）
Kronecker 刚度与装配的高阶 `K` 对照。内存上限默认 64 GiB；程序会在构造任何
逐行临时 map 或 PETSc `P` 之前，用包含 AIJ 与临时结构的保守估计执行预检。

`-trunc_chol_boundary_protection true|false` 默认 `true`，保持原边界保护规则；
设为 `false` 时，值空间和导数空间的 Cholesky 因子均严格截断到半带宽 `k`，
对角元始终保留。结果文件逐方向记录原质量矩阵半带宽、完整/截断因子的实际最大
非零半带宽、因子 nnz 与删除数。

`right_fgmres_amg` 是 AMG-only 逆作用：右 FGMRES 的每次预条件调用在已组装的
预条件矩阵 `P` 上执行一次复用的 `KSPPREONLY+PCGAMG`，不会启动内层 CG。它与
`right_fgmres_cg -inner_pc gamg`（CG+GAMG）是不同算法。
`-compute_unpreconditioned_solution false` 可只求预条件系统，同时继续计算该解的
真实残差和物理域 L² 误差；默认仍为 `true`，兼容历史运行。

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

每个旧几何批处理脚本都列出了原有求解方案。需要方案 A1 时，在脚本配置区注释
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
`right_gmres_cg`、`right_fgmres_cg`、`right_fgmres_amg`、作为参考的 `left_gmres_k_jacobi`、
`left_gmres_k_gamg`、`right_gmres_k_gamg` 和默认的
`left_gmres_fixed_jacobi`。两个 K-GAMG 模式直接在高阶 K 上构造一次 AMG 层次，
不组装 K_LOR；方向由名称唯一确定，结果仍会记录
`preconditioner_side` 便于筛选。所有模式结束后都重算原方程真实残差；
`run_case.sh` 末尾四个可选位置参数用于三个 `*_cg` 方案。
每个方案只接收并记录自身有效的参数：LU 结果中没有 inner/fixed 字段，`*_cg` 没有
fixed 字段，固定 Jacobi 也没有 inner CG 字段。

方案 B 默认固定执行 `m=4, omega=2/3` 的 weighted Jacobi。单算例可用
`FIXED_JACOBI_STEPS` 和 `FIXED_JACOBI_OMEGA` 覆盖；批处理直接编辑已暴露的方案块。

三个 `*_cg` 方案的样例默认写为 `inner_pc="jacobi"`；要使用 PETSc GAMG，将所选
方案块中的值改成 `inner_pc="gamg"`。

旧批处理方案块都显式列出 `outer_rtol`、`outer_atol`、`outer_dtol` 和
`outer_max_it`。这些值由公共执行器传给单算例，不再只能依赖 `run_case.sh` 的隐藏默认值。

GAMG 层次在 setup 阶段构造一次并在所有内层调用中复用。性能比较应使用
`setup_time_seconds + outer_solve_time_seconds`，不能只看内层迭代数。
K-GAMG 模式目前不支持 `compute_condition_number=true`，但仍统一输出真实相对残差和
L2 误差。

结果中的 `outer_solve_time_seconds` 是完整外层 `KSPSolve` 时间，已经包含其间所有
内层 CG 调用；`inner_solve_time_seconds` 是这段时间内内层求解的累计子成本，二者
不能相加。MPI 计时采用所有 ranks 中的最大 wall time，代表该阶段的关键路径。

## Truncated-Cholesky 二维/三维 sweep

两个新入口默认只生成计划，不启动数值求解：

```bash
RUN_ID=tc2d-trial BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_2d.sh
RUN_ID=tc3d-trial BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_3d.sh
```

检查对应 run 目录的 `manifest.tsv` 后，由用户手动使用相同 `RUN_ID` 正式运行：

```bash
RUN_ID=tc2d-trial BATCH_DRY_RUN=0 ./work/run_trunc_chol_sweep_2d.sh
RUN_ID=tc3d-trial BATCH_DRY_RUN=0 ./work/run_trunc_chol_sweep_3d.sh
```

二维默认覆盖 `p=4..8` 与 `n=4..1024` 的九个网格，三维覆盖 `p=4..8` 与
`n=4..64` 的五个网格。默认 `SOLVER_MODES="left_gmres_cg_jacobi"`、
`BOUNDARY_PROTECTION_MODES="true false"`；可用求解方式为
`lu left_gmres_cg_jacobi right_gmres_cg_jacobi left_gmres_cg_gamg
right_gmres_cg_gamg cg_jacobi cg_gamg amg`。显式的普通 GMRES 名称分别映射到
左/右 GMRES + 内层 CG/Jacobi 或 CG/GAMG；短名称 `cg_jacobi`、`cg_gamg` 则保留
为右 FGMRES 版本。`P_VALUES`、`N_VALUES` 可供用户先做小规模过滤，
未设置时不会删减规定的完整列表。

为与其他方法使用相同停止精度，专用 sweep 默认显式传递外层
`rtol=1e-6, atol=1e-9`，以及内层 CG `rtol=atol=1e-6`；这些值会进入配置哈希、
manifest 命令和结果文件。通用 `run_case.sh` 的历史默认值保持不变。

左普通 GMRES + 内层 CG/Jacobi 的专用扫描可显式选择：

```bash
RUN_ID=left-gmres-cg-jacobi-2d \
  BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_2d.sh
RUN_ID=left-gmres-cg-jacobi-3d \
  BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_3d.sh
```

左预条件普通 GMRES 的内部停止准则可能使用预条件残差；分析结果时应以输出中对
原方程重新计算的 `relative_true_residual` 和物理域 L² 误差为准。

左普通 GMRES + 内层 CG/GAMG 使用独立 `RUN_ID`：

```bash
RUN_ID=left-gmres-cg-gamg-2d SOLVER_MODES="left_gmres_cg_gamg" \
  BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_2d.sh
RUN_ID=left-gmres-cg-gamg-3d SOLVER_MODES="left_gmres_cg_gamg" \
  BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_3d.sh
```

每个 run 目录包含 `manifest.tsv`、稳定的一行一案例 `summary.tsv`、`cases/`、
`logs/` 和配置哈希成功标记。`RESUME=1` 只跳过结果存在且哈希一致的成功案例；
失败、超时及残缺结果可重试。内存预检拒绝记录为 `skipped_memory`。正式 sweep
默认关闭无预条件基线、EPS、SVD 和张量恒等式验证，并按案例顺序执行以避免聚合
内存峰值。外层求解时间已经包含 PC 应用时间，不可与应用累计时间再次相加。
`trunc_chol_conservative_memory_estimate_bytes` 是张量预条件矩阵及其装配临时结构的
保守预检值，不代表整个进程（尤其是高阶 `K`、GAMG 层次或 LU 因子）的全局峰值。

针对 truncated-Cholesky + 左 GMRES + 内层 CG/GAMG 的参数研究，使用独立脚本
`work/run_trunc_chol_gamg_tuning_2d.sh`。它按 `rtol -> threshold -> smooth ->
transfer` 四阶段运行，默认只生成计划，并为每阶段单独记录 setup、内外迭代、完整
外层求解时间、真实残差和 L²。具体命令及可覆盖参数见 `work/README.md`；选择参数时
应优先比较 `T_solve_s`，不能仅比较内层迭代数。
`p=8,n=1024x1024` 的独立 rtol 扫描入口为
`work/run_trunc_chol_gamg_rtol_p8_n1024_2d.sh`，默认 dry-run 且不会覆盖较小算例结果。
更严格的右 FGMRES 诊断组使用
`work/run_trunc_chol_fgmres_gamg_diagnostic_p8_n1024_2d.sh`，固定 `k=4,bp=true`，
扫描 `inner_rtol=1e-8..1e-3`，并写入独立 RUN_ID。

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
