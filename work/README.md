# Work inputs and scripts

`work/` 保存可迁移到测试机器的几何输入和 shell 工作流。`results/`、`logs/` 与
`batch_summary.log` 是运行时产物，不属于算法源码。

## 文件职责

| 文件 | 当前职责 | 关键内容 |
|---|---|---|
| `input/square_2d.txt` | 最小二维轴对齐矩形几何 | 初始 `p=1, n=(1,1)`，物理域 `[0,2] x [0,1]` |
| `input/cube_3d.txt` | 最小三维轴对齐长方体几何 | 初始 `p=1, n=(1,1,1)`，物理域 `[0,2] x [0,1.5] x [0,1]` |
| `input/curved_quadrilateral_2d.txt` | 精确多项式双三次曲边 Patch | 初始 `p=3, n=(1,1)`；当前 assembled 制造解 Driver 不运行它 |
| `run_case.sh` | 验证参数并以 `mpirun -n MPI_RANKS` 运行一个 case | 传入 rank 数、refinement、p/n、point set、LOR 模式、结果路径和可选容差 |
| `run_batch.sh` | 顺序生成并运行一组合法 case | 配置 `mpi_ranks`；默认 GMRES-shell；保存结果、console log 和 TSV summary |

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

`run_batch.sh` 顶部保留纯 p 和纯 h 的注释模板，当前激活配置为：

```bash
refinement="k_refine"
degrees=(3 4 5 6 7)
element_cases=("4,4" "16,16" "128,128")

lor_points="greville"
# lor_points="demko"

mpi_ranks=4
lor_pc="gmres_shell"
# lor_pc="cg_lu"

svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
```

单算例也可以用环境变量覆盖 SVD 设置，例如：

```bash
SVD_MAX_IT=200000 SVD_NCV=160 ./work/run_case.sh ...
```

先预览 15 个 case 而不启动求解：

```bash
BATCH_DRY_RUN=1 ./work/run_batch.sh
```

正式运行：

```bash
./work/run_batch.sh
```

每个 case 的名称包含 MPI rank 数、refinement、实际目标 p/n、point set 和 LOR 模式。数值结果写入
`work/results/`，终端输出复制到 `work/logs/`，`work/batch_summary.log` 逐项记录
success/failed、耗时和文件路径。单个 case 失败后脚本继续余下计划，最后以非零状态
报告整个 batch 存在失败。

三维批处理只需切换几何文件，并把 element tuple 改为三项，例如 `4,4,4`。
