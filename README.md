# LOR-MF-SF customized

面向 `Matrix-free-IGA` 论文的单机 CPU 最小原型。当前版本已经贯通普通 B-spline
高阶 assembled Laplace 系统、Greville/Demko LOR 网格、Q1 LOR 预条件矩阵、两种
KSP 应用方式、SLEPc 条件数和 L2 误差分析。Matrix-free 与 sum factorization
尚未实现，当前 assembled 路径是后续优化的正确性基线。

## 当前能力

- 2D/3D、单 patch、普通非有理 B-spline 几何；
- 各方向相同次数、独立 knot vector 和独立 \(n_i\)；
- 数学正确的 p-/h-/k-refinement；
- 全边界齐次 Dirichlet 条件；
- 高阶与 LOR Q1 的刚度、质量、荷载独立按需组装；
- `greville|demko` LOR point set；
- `cg_lu|gmres_shell` 两种 LOR 求解配置；
- `cond_2(K)`、`cond_2(K_LOR^{-1}K)`、外层迭代数和 L2 误差输出；
- 单算例与顺序批处理脚本。

当前不支持 NURBS、周期边界、多 patch、MPI 分布式计算、MF/SF 算子或曲边制造解。

## 构建

```bash
mkdir -p build
cd build
cmake ../examples -DCMAKE_BUILD_TYPE=Release
make -j
```

可执行文件为 `build/assembled_driver`。新机器首先维护
`conf/system_lib_loading.cmake`，或用 `SYSTEM_LIB_LOADING_FILE` 指定外部配置。

## 运行

```bash
./work/run_case.sh \
  work/input/square_2d.txt \
  k_refine 5 16,16 \
  greville cg_lu \
  work/results/example.log
```

批处理预览和正式运行：

```bash
BATCH_DRY_RUN=1 ./work/run_batch.sh
./work/run_batch.sh
```

## 顶层文件和目录

| 路径 | 当前职责 |
|---|---|
| `.gitignore` | 忽略 `.DS_Store` 和整个 `build/` 目录 |
| `CMakeLists.txt` | 定义 `lor_core`、`lor_petsc`、`lor_solver` 和 examples 构建入口 |
| `DEVELOPMENT_GUIDE.md` | 当前数学范围、真实开发状态、架构约束与下一阶段 |
| `conf/` | 本机/示例机器配置以及 PETSc、SLEPc 查找逻辑 |
| `include/` | IO、Mesh、Element、Assembly、Solver 的公开接口 |
| `src/` | 上述接口的实现 |
| `examples/` | 当前唯一 assembled 实验驱动器 |
| `work/` | 最小几何输入、单算例脚本、批处理脚本和运行产物 |

每个代码文件的职责和对象关系见各目录 README。完整开发现状见
`DEVELOPMENT_GUIDE.md`。
