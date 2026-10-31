# Examples: current assembled driver

`examples/` 当前只保留一个端到端程序，用它建立后续 matrix-free 和
sum-factorization 的数值参考。

| 文件 | 当前职责 |
|---|---|
| `CMakeLists.txt` | 支持从项目根构建，也支持用户要求的 `cmake ../examples` 独立入口；生成 `assembled_driver` 并链接 `lor_solver` |
| `assembled_driver.cpp` | 初始化 SLEPc，调用 root-only 集体几何读取，解析 options，并把 2D/3D 分派给 `AssembledLaplaceExperiment`；不实现 refinement、组装或求解 |

## Driver 对象关系

主程序按以下顺序连接模块：

1. `DistributedGeometryInput` 只在 rank 0 读取文件并广播最低必要几何；
2. Solver 的 `AssembledLaplaceExperiment` 调用 Mesh `Refinement`，只在 rank 0 构造
   实际分析 Patch，再通过 `DistributedPatch` 广播；
3. `Patch::boundaries -> LexicographicOrdering -> DirichletBC -> ID` 建立全边界
   齐次约束；
4. `TensorSlabPartition` 在最慢张量方向均衡划分高阶单元、LOR cells 和连续 ID
   所有权；
5. 预计算逐方向 `BSplineBasis1D` Gauss 表，通过 `BSplineIEN` 和
   `GlobalAssembly` 的 `stiffness+load` 融合入口组装本算例需要的高阶 K 与 F；
6. `GrevillePointSet|DemkoPointSet -> LORMesh -> LORIEN -> GlobalAssembly`
   只组装作为预条件器的 Q1 LOR 刚度矩阵；
7. 分别调用无预条件和 LOR 预条件 Solver；
8. `ErrorAnalysis` 和 `ConditionNumber` 以 MPI collective 生成六项数值指标，rank 0
   写结果文件。

Solver 实验对象使用的制造解是

\[
u(x)=\prod_d\sin\!\left(\frac{\pi(x_d-a_d)}{b_d-a_d}\right),\qquad
f(x)=\left[\sum_d\left(\frac{\pi}{b_d-a_d}\right)^2\right]u(x).
\]

它只适用于轴对齐矩形/长方体。曲边输入能够被 IO、Mesh 和 refinement 使用，但
当前 Driver 会在组装前拒绝该制造解不适用的几何。

## 构建

```bash
mkdir -p build
cd build
cmake ../examples -DCMAKE_BUILD_TYPE=Release
make -j
```

## 直接运行

```bash
mpirun -n 4 ./build/assembled_driver \
  -geometry work/input/square_2d.txt \
  -refinement k_refine \
  -p 5 -n 16,16 \
  -lor_points greville \
  -lor_pc gmres_shell \
  -outer_rtol 1e-10 -outer_atol 1e-10 \
  -output work/results/example.log
```

### Refinement options

- `p_refine`：只改变共同次数；`-n` 必须等于输入几何的初始单元数；
- `h_refine`：只插入 knots；`-p` 必须等于输入几何的初始次数；
- `k_refine`：先升到共同目标次数，再分别达到每方向目标 `n_i`。

### LOR options

- `-lor_points greville|demko`；
- `-lor_pc gmres_shell|cg_lu`，省略时默认为 `gmres_shell`；
- `-svd_tol`、`-svd_max_it`、`-svd_ncv` 控制四个 SLEPc 极端奇异值求解；
- `cg_lu` 使用外层 CG 和精确 LOR LU；
- `gmres_shell` 使用外层 restarted GMRES 和 PCSHELL 内层 CG/Jacobi。

## 结果格式

结果文件记录实际 refinement、实际 p/n，然后记录：

```text
mpi_ranks
cond_2(K)
iterations(K)
L2_error(K)
cond_2(K_LOR^{-1}K)
iterations(K_LOR^{-1}K)
L2_error(K_LOR^{-1}K)
```

条件数来自独立 SLEPc SVD，不由 CG/GMRES 迭代投影估计。两个 L2 error 都针对同一
高阶系统 `K u=F`；LOR 系统只作为预条件器，不作为另一个 PDE 解进行误差比较。
质量矩阵接口仍由 Assembly 提供，但该 Laplace Driver 不请求它，因此运行时不会
产生高阶或 LOR 质量矩阵。
