# Work inputs and scripts

`work/` 保存可迁移到测试机器的几何输入和 shell 工作流。`results/`、`logs/` 与
各几何的 `batch_summary_*.log` 是运行时产物，不属于算法源码。

## 文件职责

| 文件 | 当前职责 | 关键内容 |
|---|---|---|
| `input/square_2d.txt` | 最小二维轴对齐矩形几何 | 初始 `p=1, n=(1,1)`，物理域 `[0,2] x [0,1]` |
| `input/cube_3d.txt` | 最小三维轴对齐长方体几何 | 初始 `p=1, n=(1,1,1)`，物理域 `[0,2] x [0,1.5] x [0,1]` |
| `input/curved_quadrilateral_2d.txt` | 精确多项式双三次曲边 Patch | 初始 `p=3, n=(1,1)`；使用非零仿射迹与曲边多项式 bubble 完成求解、L2 和条件数分析 |
| `run_case.sh` | 验证参数、拒绝过期 executable，并以 `mpirun -n MPI_RANKS` 运行一个 case | 传入 rank 数、refinement、p/n、point set、LOR 模式、结果路径和可选容差；环境变量控制三个独立功能 |
| `run_batch_square_2d.sh` | 二维矩形批处理入口 | `p={3,...,7}`，`n={(4,4),(16,16),(128,128)}`；默认求解并计算 L2 |
| `run_batch_cube_3d.sh` | 三维长方体批处理入口 | `p={3,...,7}`，`n={(4,4,4),(8,8,8),(16,16,16)}`；限制三维默认规模的立方增长 |
| `run_batch_curved_quadrilateral_2d.sh` | 双三次曲边 Patch 批处理入口 | 与二维矩形采用相同 p/n sweep 和功能默认值；默认完成求解与 L2 |
| `run_batch_common.sh` | 三个入口共享的执行器 | 统一参数验证、逐 case rank 规划、运行、日志和 TSV summary；不直接运行 |
| `run_inner_tolerance_sweep_square_2d.sh` | 固定二维 case 的内层 CG 精度扫描 | 默认扫描 `inner_rtol={1e-10,1e-6,1e-4,1e-3,1e-2}`，汇总内外迭代、时间、真实残差和 L2 误差 |
| `run_trunc_chol_gamg_tuning_2d.sh` | truncated-Cholesky + 左 GMRES + 内层 CG/GAMG 的四阶段调优 | 依次扫描内层 rtol、GAMG threshold、V-cycle level smoother 步数，再做 p/n 可迁移性扫描；默认只 dry-run |
| `run_trunc_chol_gamg_rtol_p8_n1024_2d.sh` | `p=8,n=1024x1024` 的专用内层 rtol 扫描入口 | 复用四阶段调优器、使用独立 RUN_ID、默认只 dry-run，支持断点续跑 |
| `run_trunc_chol_fgmres_gamg_diagnostic_p8_n1024_2d.sh` | `p=8,n=1024x1024` 的右 FGMRES + 内层 CG/GAMG 诊断扫描 | 固定 `k=4,bp=true`、更严格的内外层绝对容差，并扫描 `inner_rtol=1e-8..1e-3`；默认只 dry-run |
| `run_pc_comparison_p4_p7_n512_n1024_2d.sh` | `p=4..7,n={512,1024}²` 的九种 selected-P 公平对照入口 | 两类 P 各比较右侧三种应用和左 GMRES+GAMG，另含 Greville 一步 CG/Jacobi；共 72 cases，默认只 dry-run |
| `run_pc_comparison_p7_n512_2d.sh` | 上述扩展实验的兼容实现路径 | 保留早期脚本路径；默认参数现已与九组扩展实验一致 |
| `run_pc_comparison_cube_3d_p4_p7_n32_n64.sh` | `p=4..7,n={32,64}³` 的九种方法对照 | 七种 selected-P 方法加 K-Jacobi/K-GAMG，排除 A4/B5，共 72 cases；默认由 case planner 选择 ranks，且只 dry-run |
| `run_pc_baseline_high_order_k_p4_p7_n512_n1024_2d.sh` | 与二维扫描对应的高阶 K 基准 | K1 为左 GMRES+Jacobi(K)，K2 为左 GMRES+GAMG(K)；共 16 cases，写入独立结果目录 |
| `REMOTE_BATCH_GUIDE.md` | 远端 macOS/Linux 构建与批处理清单 | 说明 dry-run、`MPI_RANK`、`MAX_MPI_RANKS`、launcher 覆盖、输出和失败恢复 |

几何文件中的 p/n 注释供用户阅读；解析器只从 knot vectors 推断实际初始值。控制点
带显式 tensor index，文本书写顺序不影响几何。

`run_case.sh` 会比较 C++/CMake 输入和 `build/assembled_driver` 的修改时间。如果源码
更新后尚未重新构建，它会在启动 MPI 前停止并给出构建命令，避免新脚本向旧程序传入
其尚不认识的 PETSc 选项。

## 内层 CG 容差扫描

二维 `p=3, n=(512,512)` 的默认扫描可直接运行：

```bash
MPI_RANK=27 ./work/run_inner_tolerance_sweep_square_2d.sh
```

建议先用小网格 dry-run 检查远端配置：

```bash
BATCH_DRY_RUN=1 SWEEP_N=32,32 MPI_RANK=2 \
  ./work/run_inner_tolerance_sweep_square_2d.sh
```

扫描脚本默认使用 `INNER_PC=gamg`；设置 `INNER_PC=jacobi` 会生成独立的基线目录和
TSV。`INNER_RTOLS` 用空格分隔自定义扫描点；`INNER_ATOL` 默认 `1e-50`，避免绝对容差
掩盖相对容差实验。`SWEEP_P`、`SWEEP_N`、`OUTER_RTOL`、`OUTER_ATOL` 和
`COMPUTE_L2_ERROR` 也可由环境变量覆盖。结果汇总写入独立 TSV，不覆盖普通批处理。

新 tensor 来源通过环境变量传递，不改变 `run_case.sh` 的 8/12 个位置参数：

```bash
PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor \
TRUNC_CHOL_K=2 COMPUTE_TRUNCATION_CERTIFICATE=true \
COMPUTE_SPECTRAL_CONDITION_NUMBER=true EPS_TOL=1e-8 EPS_NCV=100 \
./work/run_case.sh 2 input/square_2d.txt k_refine 4 16,16 \
  greville left_cg_lu results/truncated_p4.log
```

这些变量及 `EPS_MAX_IT` 会由 `run_batch_common.sh` 传给二维/三维批处理。
`VERIFY_TENSOR_IDENTITY=true` 可开启独立张量恒等式验证；
`TRUNC_CHOL_MAX_MEMORY_BYTES` 设置装配前的保守内存上限（默认 64 GiB）。
二维和三维批处理中的 `COMPUTE_CONDITION_NUMBER`、EPS/SVD 开关均可由环境变量
覆盖，并已编码进结果文件名以避免不同配置互相覆盖。

专用 truncated-Cholesky 扫描使用：

```bash
RUN_ID=manual-2d BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_2d.sh
RUN_ID=manual-3d BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_3d.sh
# 审阅 manifest.tsv 后，由用户自行设置 BATCH_DRY_RUN=0。
```

默认求解方式为 `SOLVER_MODES="left_gmres_cg_jacobi"`，还可选择 `lu`、
`left_gmres_cg_jacobi`、`right_gmres_cg_jacobi`、`left_gmres_cg_gamg`、
`right_gmres_cg_gamg`、`cg_jacobi` 或 `cg_gamg`。显式左右名称映射到普通
GMRES + 内层 CG/Jacobi 或 CG/GAMG，短名称映射到右 FGMRES。边界开关
默认同时扫描 `true false`。正式 sweep 默认设置
`COMPUTE_UNPRECONDITIONED_SOLUTION=false` 并关闭 EPS/SVD/张量恒等式验证。
`summary.tsv` 中 `not_applicable` 表示算法没有相应概念（如 AMG-only/LU 的内层
CG 次数），`NA` 表示失败、未开始或结果中没有该量，`not_computed` 是 driver 对
明确禁用功能的状态。时间单位为秒，内存字段为字节。
曲边入口必须继续使用默认 `lor_q1`。

专用 sweep 默认使用外层 `rtol=1e-6, atol=1e-9`，内层 CG
`rtol=atol=1e-6`。这些容差被显式写入 manifest、配置哈希和 driver 参数；如需
覆盖，仍可设置 `OUTER_RTOL`、`OUTER_ATOL`、`INNER_RTOL` 和 `INNER_ATOL`。

准备左普通 GMRES + 内层 CG/Jacobi 批处理：

```bash
RUN_ID=left-gmres-cg-jacobi-2d \
  BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_2d.sh
RUN_ID=left-gmres-cg-jacobi-3d \
  BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_3d.sh
```

左预条件普通 GMRES 可能按预条件残差判停，结果筛选应同时检查输出中的
`relative_true_residual`，不能只看 PETSc 的 KSP reason。

准备左普通 GMRES + 内层 CG/GAMG 对照组：

```bash
RUN_ID=left-gmres-cg-gamg-2d SOLVER_MODES="left_gmres_cg_gamg" \
  BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_2d.sh
RUN_ID=left-gmres-cg-gamg-3d SOLVER_MODES="left_gmres_cg_gamg" \
  BATCH_DRY_RUN=1 ./run_trunc_chol_sweep_3d.sh
```

### GAMG 四阶段参数调优

独立脚本 `run_trunc_chol_gamg_tuning_2d.sh` 不修改或覆盖已有 sweep 结果。固定外层
左 GMRES 与内层 CG/GAMG，按以下阶段手动运行：

```bash
# 1. p=8,n=256：固定 threshold/smoother，扫描内层 rtol
RUN_ID=gamg-study TUNING_STAGE=rtol BATCH_DRY_RUN=1 \
  ./run_trunc_chol_gamg_tuning_2d.sh
RUN_ID=gamg-study TUNING_STAGE=rtol BATCH_DRY_RUN=0 \
  ./run_trunc_chol_gamg_tuning_2d.sh

# 2. 根据 summary_rtol.tsv 选择仍保持真实残差和 L2 的最大 rtol
RUN_ID=gamg-study TUNING_STAGE=threshold SELECTED_INNER_RTOL=1e-6 \
  BATCH_DRY_RUN=0 ./run_trunc_chol_gamg_tuning_2d.sh

# 3. 固定选择的 rtol/threshold，扫描每层平滑步数
RUN_ID=gamg-study TUNING_STAGE=smooth SELECTED_INNER_RTOL=1e-6 \
  SELECTED_GAMG_THRESHOLD=0.05 BATCH_DRY_RUN=0 \
  ./run_trunc_chol_gamg_tuning_2d.sh

# 4. 把选中的三元组用于 p=4..8、n=32,64,128,256
RUN_ID=gamg-study TUNING_STAGE=transfer SELECTED_INNER_RTOL=1e-6 \
  SELECTED_GAMG_THRESHOLD=0.05 SELECTED_GAMG_SMOOTH_STEPS=2 \
  BATCH_DRY_RUN=0 ./run_trunc_chol_gamg_tuning_2d.sh
```

每阶段分别生成 `manifest_<stage>.tsv` 和 `summary_<stage>.tsv`。汇总包含
`T_setup_pc_s`、`T_setup_total_s`、`N_inner`、`N_outer`、`T_solve_s`、真实相对残差
和 `L2_error`；调优首先比较 `T_solve_s`，不能仅以 `N_inner` 最少作为最优标准。
`GAMG_SMOOTH_STEPS` 控制 V-cycle 各 level smoother 的迭代步数，不是插值算子的
`pc_gamg_agg_nsmooths`。脚本支持 `RESUME=1`、`TIMEOUT_SEC`、`FAIL_FAST`、
`MPI_RANK` 和 `MAX_MPI_RANKS`。

昂贵的 `p=8,n=1024x1024` rtol 对照使用独立入口，不会覆盖上述 `p=8,n=256`
结果：

```bash
BATCH_DRY_RUN=1 ./run_trunc_chol_gamg_rtol_p8_n1024_2d.sh
BATCH_DRY_RUN=0 ./run_trunc_chol_gamg_rtol_p8_n1024_2d.sh
```

默认扫描 `1e-10 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2`，可用
`INNER_RTOLS` 缩小集合。固定 RUN_ID 为 `p8-n1024-gamg-rtol`，因此中断后以同一命令
重启会复用成功标记；可通过 `RUN_ID` 显式改名。

右 FGMRES 诊断组使用独立入口和结果目录，保留上述全部结果：

```bash
BATCH_DRY_RUN=1 ./run_trunc_chol_fgmres_gamg_diagnostic_p8_n1024_2d.sh
BATCH_DRY_RUN=0 ./run_trunc_chol_fgmres_gamg_diagnostic_p8_n1024_2d.sh
```

该组默认复用上一组实际采用的 107 ranks，并固定
`outer_rtol=1e-8, outer_atol=1e-10, inner_atol=1e-12, k=4, bp=true`；扫描点为
`inner_rtol={1e-8,1e-7,1e-6,1e-5,1e-4,1e-3}`。这里 `1e-3` 只是待比较的最后一个
扫描点，不是调优器或通用求解器的默认容差。结果应同时比较 `N_inner`、
`relative_true_residual` 与 `L2_error`，从中选择不再明显受代数误差影响的最大容差。

### p=4..7、512²/1024² 九种 selected-P 方法对照

专用入口 `run_pc_comparison_p4_p7_n512_n1024_2d.sh` 使用 `square_2d`，默认扫描
`p={4,5,6,7}` 与 `n={(512,512),(1024,1024)}`。512² 默认使用 27 ranks，1024²
默认使用 107 ranks，并采用共同的外层容差
`rtol=1e-6,atol=1e-9`。需要嵌套 CG 的
方法默认使用 `inner_rtol=1e-3,inner_atol=1e-12`；较小的绝对容差用于避免它掩盖
本次指定的相对容差。每个 p/n 组合运行九组方法：

1. A1：truncated-Cholesky `P` + 右 FGMRES + 内层 CG/GAMG；
2. A2：truncated-Cholesky `P` + 右 FGMRES + 直接 GAMG；
3. A3：truncated-Cholesky `P` + 右 FGMRES + 直接 `diag(P)^{-1}` Jacobi；
4. A4：truncated-Cholesky `P` + 左普通 GMRES + 直接 GAMG；
5. B1：Greville Q1-LOR `P` + 右 FGMRES + 内层 CG/GAMG；
6. B2：Greville Q1-LOR `P` + 右 FGMRES + 直接 GAMG；
7. B3：Greville Q1-LOR `P` + 右 FGMRES + 直接 `diag(P)^{-1}` Jacobi；
8. B4：Greville Q1-LOR `P` + 右 FGMRES + 一步内层 CG/Jacobi；
9. B5：Greville Q1-LOR `P` + 左普通 GMRES + 直接 GAMG。

其中 B4 的“一次内迭代”严格实现为一次 CG/Jacobi 迭代；它达到迭代上限时，
`inner_failed_applications` 可能非零，这是固定一步近似逆的诊断信息，不等同于外层
批处理失败。A2/A3/B2/B3 不启动内层 CG，因此其内层迭代字段为
`not_applicable`。

先预览计划，再用相同 `RUN_ID` 正式运行：

```bash
BATCH_DRY_RUN=1 ./work/run_pc_comparison_p4_p7_n512_n1024_2d.sh
BATCH_DRY_RUN=0 ./work/run_pc_comparison_p4_p7_n512_n1024_2d.sh
```

可显式指定资源，例如：

```bash
RUN_ID=p4-p7-n512-n1024-nine-way \
  BATCH_DRY_RUN=0 ./work/run_pc_comparison_p4_p7_n512_n1024_2d.sh
```

`MPI_RANK_512` 和 `MPI_RANK_1024` 可分别覆盖默认 ranks；`MPI_RANK` 会统一覆盖
两个网格。`DEGREES` 与 `ELEMENT_CASES` 可缩小扫描集合。`GAMG_THRESHOLD` 与
`GAMG_SMOOTH_STEPS` 同时作用于 A1/A2/A4/B1/B2/B5；默认分别为 `0.0` 和 `1`，
并通过统一的 `inner_` PETSc 前缀传递。

默认结果位于
`work/results/pc_comparison_2d/<RUN_ID>/`，其中 `manifest.tsv` 保存实际参数与命令，
`summary.tsv` 每组一行，包含 K/P 构造时间、AMG setup、外层 setup/solve、PC apply、
内层 CG、L2 与端到端时间，以及迭代数、应用次数、reason、真实残差、nnz 和内存。
`not_applicable` 表示算法没有该概念；`not_recorded` 表示理论上存在但当前无法可靠
取得。公平比较应使用端到端时间或明确的非嵌套阶段；`T_pc_apply_s` 和
`T_inner_CG_s` 都嵌套在 `T_outer_solve_s` 中，不能重复相加。

可用 `OUTER_RTOL`、`OUTER_ATOL`、`INNER_RTOL`、`INNER_ATOL`、`INNER_MAX_IT`、
`TRUNC_CHOL_K`、`TRUNC_CHOL_BOUNDARY_PROTECTION`、`GAMG_THRESHOLD` 和
`GAMG_SMOOTH_STEPS` 做显式覆盖。默认
`TRUNC_CHOL_K=auto`，逐 p 计算 `ceil(p/2)`。脚本关闭 EPS/SVD 和无预条件重复求解，
只计算九组预条件解、真实残差与物理域 L2 误差。尤其应比较 A4/B5 与对应右预条件
组的 `relative_true_residual`、`L2_error`，不能仅依据左 GMRES 的 KSP reason。

### 3D cube 七种 selected-P 方法与两个高阶 K 基准

三维入口固定使用 `cube_3d`，扫描 `p={4,5,6,7}` 和
`n={(32,32,32),(64,64,64)}`。它生成 A1/A2/A3/B1/B2/B3/B4，并增加：

1. K1：高阶 `K` + 左普通 GMRES + `PCJACOBI(K)`；
2. K2：高阶 `K` + 左普通 GMRES + `PCGAMG(K)`。

脚本不包含 A4/B5，共九种方法、72 个 case：

```bash
RUN_ID=cube3d-p4-p7-n32-n64-nine-way \
  BATCH_DRY_RUN=1 ./work/run_pc_comparison_cube_3d_p4_p7_n32_n64.sh
RUN_ID=cube3d-p4-p7-n32-n64-nine-way \
  BATCH_DRY_RUN=0 ./work/run_pc_comparison_cube_3d_p4_p7_n32_n64.sh
```

默认 `MPI_RANK=default`，因此每个 p/n 由 `case_planner` 根据实际自由度选择 ranks；
可以用 `MAX_MPI_RANKS` 限制自动规划，或用 `MPI_RANK` 为全部 72 个 case 固定相同
ranks。例如：

```bash
MAX_MPI_RANKS=64 BATCH_DRY_RUN=0 \
  ./work/run_pc_comparison_cube_3d_p4_p7_n32_n64.sh
```

输出写入
`work/results/pc_comparison_cube_3d/<RUN_ID>/`，目录结构和 `summary.tsv` 字段与二维
批次相同。三维规模对内存更敏感；默认 64 GiB 预检上限可用
`TRUNC_CHOL_MAX_MEMORY_BYTES` 覆盖，超过上限的 case 会记录为 `skipped_memory`，不会
继续装配 P。

### 二维高阶 K 基准

独立基准脚本只运行以下两种方法，不组装 truncated-Cholesky 或 Greville LOR 矩阵：

1. K1：高阶 `K` + 左普通 GMRES + `PCJACOBI(K)`；
2. K2：高阶 `K` + 左普通 GMRES + `PCGAMG(K)`。

它与二维 selected-P 批次使用相同的 `p={4,5,6,7}`、
`n={(512,512),(1024,1024)}`、外层容差和默认 ranks，共 16 个 case：

```bash
RUN_ID=high-order-k-p4-p7-n512-n1024 \
  BATCH_DRY_RUN=1 ./work/run_pc_baseline_high_order_k_p4_p7_n512_n1024_2d.sh
RUN_ID=high-order-k-p4-p7-n512-n1024 \
  BATCH_DRY_RUN=0 ./work/run_pc_baseline_high_order_k_p4_p7_n512_n1024_2d.sh
```

结果位于 `work/results/pc_baseline_high_order_k_2d/<RUN_ID>/`。直接 PETSc PC 没有经过
计时 shell，所以单独的 PC setup/apply 次数与时间标为 `not_separately_timed` 或
`not_recorded`；这些成本已经包含在 `T_outer_setup_s` 和 `T_outer_solve_s` 中。
K-GAMG 使用无前缀参数 `-pc_gamg_threshold` 和 `-mg_levels_ksp_max_it`，脚本仍通过
统一的 `GAMG_THRESHOLD`、`GAMG_SMOOTH_STEPS` 环境变量配置。

前述 `p=8,n=1024` 容差扫描显式固定为 `left_gmres_cg`，用于保留左预条件标准
GMRES 对照。PETSc
对该组合监测预条件残差，宽松且
随右端项变化的内层 CG 可能导致它先于原方程真实残差收敛；因此不能只按耗时选择
容差，必须同时检查 TSV 中的 `relative_residual_LOR` 与 `L2_error_LOR`。右预条件
FGMRES 已作为独立的 `right_fgmres_cg` 配置提供，不能与本扫描的迭代数直接混为
同一方法。

单 case 和普通批处理同样接受 `OUTER_RTOL`、`OUTER_ATOL`、`OUTER_DTOL`、
`OUTER_MAX_IT` 以及对应的 `INNER_*` 环境变量。`run_case.sh` 的四个尾部位置参数
优先级最高。

## 单算例

```bash
./work/run_case.sh \
  MPI_RANKS GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE \
  [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]
```

例如：

```bash
./work/run_case.sh \
  4 \
  work/input/square_2d.txt \
  k_refine 5 16,16 \
  greville left_gmres_fixed_jacobi \
  work/results/k_refine_p5_n16x16_left_gmres_fixed_jacobi.log
```

合法枚举值：

```text
REFINEMENT = p_refine | h_refine | k_refine
LOR_POINTS = greville | demko
LOR_PC = left_cg_lu | right_gmres_lu | left_gmres_cg |
         right_gmres_cg | right_fgmres_cg | right_fgmres_amg |
         right_fgmres_jacobi | left_gmres_amg | left_gmres_k_jacobi |
         left_gmres_k_gamg | right_gmres_k_gamg |
         left_gmres_fixed_jacobi
INNER_PC = jacobi | gamg  # 仅用于三个 *_cg 方案
```

## 批处理

三个几何入口各自保存 case 配置并复用 `run_batch_common.sh`。二维矩形的默认配置为：

```bash
refinement="k_refine"
degrees=(3 4 5 6 7)
element_cases=("4,4" "16,16" "128,128")

lor_points="greville"
# lor_points="demko"

mpi_rank=${MPI_RANK:-default}
# 其他八个完整方案及其配套参数在实际脚本中以注释块列出。
# 当前只暴露方案 B：
lor_pc="left_gmres_fixed_jacobi"
outer_rtol="1.0e-10"
outer_atol="1.0e-10"
outer_dtol="1.0e4"
outer_max_it="10000"
fixed_jacobi_steps="4"
fixed_jacobi_omega="0.6666666666666667"

compute_solution=true
compute_l2_error=true
compute_condition_number=false

svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
```

实际的三个几何入口都把九种方案写成独立配置块。切换时应先注释当前整个方案 B 块，
再取消注释目标块中的方案名和配套参数；不要同时暴露两个 `lor_pc` 赋值。公共执行器
会在 dry-run 中打印当前方案真正使用的参数。每个方案块都显式给出外层
`outer_rtol`、`outer_atol`、`outer_dtol` 和 `outer_max_it`，可以直接用于批处理精度扫描。

单算例也可以用环境变量覆盖 SVD 设置，例如：

```bash
COMPUTE_SOLUTION=true \
COMPUTE_L2_ERROR=false \
COMPUTE_CONDITION_NUMBER=true \
SVD_MAX_IT=200000 SVD_NCV=160 \
./work/run_case.sh ...
```

`COMPUTE_L2_ERROR=true` 要求 `COMPUTE_SOLUTION=true`。条件数分析与求解、L2
误差相互独立；默认关闭，以免普通批处理中隐式启动四次昂贵的 SLEPc 奇异值求解。

批处理 summary 的 `elapsed_s` 是包括几何、组装、求解和误差分析在内的整例时间。
求解器的 setup、外层时间与嵌套预条件成本保存在每个 case 的结果文件
和控制台日志中，适合用于论文中的求解阶段对比。

分别预览三个批处理而不启动计算：

```bash
BATCH_DRY_RUN=1 ./work/run_batch_square_2d.sh
BATCH_DRY_RUN=1 ./work/run_batch_cube_3d.sh
BATCH_DRY_RUN=1 ./work/run_batch_curved_quadrilateral_2d.sh
```

正式运行时选择对应几何入口，例如：

```bash
./work/run_batch_square_2d.sh
```

每个入口使用独立的 `work/results/<geometry>/`、`work/logs/<geometry>/` 和
`work/batch_summary_<geometry>.log`，case 名包含预条件方向、内层预条件器及
solve/L2/condition 三个开关。
单个 case 失败后脚本继续余下计划，最后以非零状态报告整个 batch 存在失败。

三个入口统一默认执行组装、求解和 L2 误差分析，并关闭条件数分析。三者使用同一个
非零仿射 Dirichlet lifting；轴对齐几何和曲边几何分别使用正弦与多项式内部 bubble，
二维自然忽略第三坐标。
