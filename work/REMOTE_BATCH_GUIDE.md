# Remote batch workflow

本页用于把项目复制到另一台 macOS/Linux 机器后，构建并顺序运行三个几何的批处理。
所有运行脚本都从自身位置解析项目根目录，不依赖开发机器的绝对路径。

## 1. 配置与构建

先根据远端 PETSc/SLEPc 安装修改 `conf/system_lib_loading.cmake`，然后执行：

```bash
mkdir -p build
cd build
cmake ../examples -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
```

应生成：

```text
build/assembled_driver
build/case_planner
```

如文件通过不保留执行权限的传输方式复制，可恢复脚本权限：

```bash
chmod +x work/run_case.sh work/run_batch_square_2d.sh \
  work/run_batch_cube_3d.sh work/run_batch_curved_quadrilateral_2d.sh
```

## 2. 先检查计划

dry-run 会执行参数验证和精确自由度/rank 规划，但不会启动 MPI 求解：

```bash
BATCH_DRY_RUN=1 ./work/run_batch_square_2d.sh
BATCH_DRY_RUN=1 ./work/run_batch_cube_3d.sh
BATCH_DRY_RUN=1 ./work/run_batch_curved_quadrilateral_2d.sh
```

默认按每 10000 个自由度一个 rank 自动选择，并以当前机器逻辑 CPU 数为上限。集群
作业分配的核心数可能小于整台机器，因此提交脚本应设置：

```bash
MAX_MPI_RANKS=32 ./work/run_batch_square_2d.sh
```

若希望所有 case 固定使用同一数量，则设置：

```bash
MPI_RANK=8 ./work/run_batch_square_2d.sh
```

`MPI_RANK` 优先于自动规划。若集群的 MPI launcher 不叫 `mpirun`，可同时指定：

```bash
MPIEXEC=mpiexec MPI_RANK=8 ./work/run_batch_square_2d.sh
```

## 3. 运行三个几何

三个入口应顺序运行，避免它们同时争用同一节点的 CPU 和内存：

```bash
MAX_MPI_RANKS=32 ./work/run_batch_square_2d.sh
MAX_MPI_RANKS=32 ./work/run_batch_cube_3d.sh
MAX_MPI_RANKS=32 ./work/run_batch_curved_quadrilateral_2d.sh
```

每个脚本默认执行组装、两次求解和两次 L2 积分，并关闭昂贵的条件数分析：

```bash
compute_solution=true
compute_l2_error=true
compute_condition_number=false
```

需要条件数时，在对应几何入口中显式改为 `compute_condition_number=true`。并行条件数
中的精确 LOR 逆要求远端 PETSc 配置包含 MUMPS。

## 4. 输出与失败恢复

三个几何分别写入：

```text
work/results/<geometry>/
work/logs/<geometry>/
work/batch_summary_<geometry>.log
```

summary 是制表符分隔文件，记录每个 case 的状态、实际 rank 数、功能开关、耗时及
结果路径。一个 case 失败后脚本会继续后续 case，最终以非零退出码报告整个 batch
存在失败；重新运行前可根据 summary 缩小 case 列表或单独调用 `run_case.sh`。

批处理脚本固定 `OMP_NUM_THREADS=1`，避免每个 MPI rank 内的数学库线程再次过度占用
CPU。后续正式启用 OpenMP 时，应同时重新设计 MPI×OpenMP 的资源划分。
