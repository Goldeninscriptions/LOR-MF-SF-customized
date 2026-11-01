# Work inputs and scripts

`work/` 保存可迁移到测试机器的几何输入和 shell 工作流。`results/`、`logs/` 与
各几何的 `batch_summary_*.log` 是运行时产物，不属于算法源码。

## 文件职责

| 文件 | 当前职责 | 关键内容 |
|---|---|---|
| `input/square_2d.txt` | 最小二维轴对齐矩形几何 | 初始 `p=1, n=(1,1)`，物理域 `[0,2] x [0,1]` |
| `input/cube_3d.txt` | 最小三维轴对齐长方体几何 | 初始 `p=1, n=(1,1,1)`，物理域 `[0,2] x [0,1.5] x [0,1]` |
| `input/curved_quadrilateral_2d.txt` | 精确多项式双三次曲边 Patch | 初始 `p=3, n=(1,1)`；使用非零仿射迹与曲边多项式 bubble 完成求解、L2 和条件数分析 |
| `run_case.sh` | 验证参数并以 `mpirun -n MPI_RANKS` 运行一个 case | 传入 rank 数、refinement、p/n、point set、LOR 模式、结果路径和可选容差；环境变量控制三个独立功能 |
| `run_batch_square_2d.sh` | 二维矩形批处理入口 | `p={3,...,7}`，`n={(4,4),(16,16),(128,128)}`；默认求解并计算 L2 |
| `run_batch_cube_3d.sh` | 三维长方体批处理入口 | `p={3,...,7}`，`n={(4,4,4),(8,8,8),(16,16,16)}`；限制三维默认规模的立方增长 |
| `run_batch_curved_quadrilateral_2d.sh` | 双三次曲边 Patch 批处理入口 | 与二维矩形采用相同 p/n sweep 和功能默认值；默认完成求解与 L2 |
| `run_batch_common.sh` | 三个入口共享的执行器 | 统一参数验证、逐 case rank 规划、运行、日志和 TSV summary；不直接运行 |
| `REMOTE_BATCH_GUIDE.md` | 远端 macOS/Linux 构建与批处理清单 | 说明 dry-run、`MPI_RANK`、`MAX_MPI_RANKS`、launcher 覆盖、输出和失败恢复 |

几何文件中的 p/n 注释供用户阅读；解析器只从 knot vectors 推断实际初始值。控制点
带显式 tensor index，文本书写顺序不影响几何。

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
  greville gmres_shell \
  work/results/k_refine_p5_n16x16_greville_gmres_shell.log
```

合法枚举值：

```text
REFINEMENT = p_refine | h_refine | k_refine
LOR_POINTS = greville | demko
LOR_PC     = cg_lu | gmres_shell
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
lor_pc="gmres_shell"
# lor_pc="cg_lu"

compute_solution=true
compute_l2_error=true
compute_condition_number=false

svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
```

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
`work/batch_summary_<geometry>.log`，case 名还包含 solve/L2/condition 三个开关。
单个 case 失败后脚本继续余下计划，最后以非零状态报告整个 batch 存在失败。

三个入口统一默认执行组装、求解和 L2 误差分析，并关闭条件数分析。三者使用同一个
非零仿射 Dirichlet lifting；轴对齐几何和曲边几何分别使用正弦与多项式内部 bubble，
二维自然忽略第三坐标。
