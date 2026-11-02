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
   约束拓扑，制造解控制系数作为独立非零 Dirichlet lift；
4. `TensorSlabPartition` 在最慢张量方向均衡划分高阶单元、LOR cells 和连续 ID
   所有权；
5. 预计算逐方向 `BSplineBasis1D` Gauss 表，通过 `GlobalAssembly` 组装高阶 K；仅在
   `compute_solution=true` 时融合组装 F，并逐单元加入 `-K_fc g_c` 消元贡献；
6. `GrevillePointSet|DemkoPointSet -> LORMesh -> LORIEN -> GlobalAssembly`
   只组装作为预条件器的 Q1 LOR 刚度矩阵；
7. `compute_solution=true` 时分别调用无预条件和 LOR 预条件 Solver；
8. `compute_l2_error=true` 时调用 `ErrorAnalysis`，
   `compute_condition_number=true` 时显式调用 `GlobalAssembly` 条件数入口；rank 0
   只写实际启用的数值字段。

Solver 实验对象使用的制造解是

\[
u(x)=1+\sum_{d=1}^{\mathrm{Dim}}x_d+\psi(x),\qquad
f(x)=-\Delta\psi(x).
\]

二维实例只包含 x、y，等价于令 z=0。矩形/长方体上的 \(\psi\) 是按物理边长
缩放的张量正弦函数；曲边 Patch 上的 \(\psi\) 是四条精确多项式边界隐式因子的
乘积。两者都在边界严格为零，因此公共边界迹为仿射函数
\(g=1+\sum_d x_d\)，其精确 B-spline 系数为 `1 + sum(control_point[d])`。
Assembly 在约化系统右端加入 `-K_fc g_c`，ErrorAnalysis 再恢复边界系数；内部
bubble 则产生非零源项和可用于收敛比较的非平凡 L2 误差。

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
  -lor_pc left_gmres_fixed_jacobi \
  -fixed_jacobi_steps 4 \
  -fixed_jacobi_omega 0.6666666666666667 \
  -compute_solution true \
  -compute_l2_error true \
  -compute_condition_number false \
  -outer_rtol 1e-10 -outer_atol 1e-10 \
  -output work/results/example.log
```

### Refinement options

- `p_refine`：只改变共同次数；`-n` 必须等于输入几何的初始单元数；
- `h_refine`：只插入 knots；`-p` 必须等于输入几何的初始次数；
- `k_refine`：先升到共同目标次数，再分别达到每方向目标 `n_i`。

### LOR options

- `-lor_points greville|demko`；
- `-lor_pc` 选择完整求解方案：`left_cg_lu|right_gmres_lu|left_gmres_cg|`
  `right_gmres_cg|right_fgmres_cg|left_gmres_fixed_jacobi`；默认最后一项；
- 预条件方向由方案名称唯一确定；`preconditioner_side` 只作为结果元数据；
- `-inner_pc jacobi|gamg` 及 `-inner_*` 仅属于三个 `*_cg` 方案；
- `-fixed_jacobi_steps` 和 `-fixed_jacobi_omega` 仅属于固定线性方案，默认 `4` 和
  `2/3`；向其他模式传入这些不相容参数会直接报错；
- `-svd_tol`、`-svd_max_it`、`-svd_ncv` 控制四个 SLEPc 极端奇异值求解；
- `left_cg_lu`/`right_gmres_lu` 使用精确 LOR LU；
- `left_gmres_cg`/`right_gmres_cg` 是原标准 GMRES 对照；
- `right_fgmres_cg` 是方案 A1，允许变化的容差控制内层 CG；
- `left_gmres_fixed_jacobi` 是方案 B，对固定线性近似逆应用标准左 GMRES；
- 三个 `*_cg` 方案的外层/内层默认容差均参考 Parallel-Computing 实际生效的
  `rtol=1e-10, atol=1e-10`，也可用命令行逐次覆盖。

### Independent feature options

- `-compute_solution true|false`：是否组装荷载并执行无预条件/LOR 两次求解；
- `-compute_l2_error true|false`：是否对两个解做 L2 积分，启用时要求 solution 为 true；
- `-compute_condition_number true|false`：是否调用四次 SLEPc 极端奇异值求解；
- 三者默认分别为 `true`、`true`、`false`。

## 结果格式

结果文件先记录实际 refinement、p/n 和三个功能开关，然后只记录启用项：

```text
mpi_ranks
lor_pc
preconditioner_side
[inner_pc]                                      # 仅 *_cg
compute_solution
compute_l2_error
compute_condition_number
[outer_rtol, outer_atol, outer_dtol, outer_max_it]
[inner_rtol, inner_atol, inner_dtol, inner_max_it] # 仅 *_cg
[fixed_jacobi_steps, fixed_jacobi_omega]           # 仅固定 Jacobi
predicted_nnz(K), used_nnz(K), allocated_nnz(K), allocation_calls(K)
estimated_minimum_bytes(K), memory_bytes(K)
predicted_nnz(K_LOR), used_nnz(K_LOR), allocated_nnz(K_LOR)
allocation_calls(K_LOR), estimated_minimum_bytes(K_LOR), memory_bytes(K_LOR)
[cond_2(K), cond_2(K_LOR^{-1}K)]
[iterations(K), iterations(<scheme>)]
[setup_time_seconds(K), outer_solve_time_seconds(K)]
[setup_time_seconds(<scheme>), outer_solve_time_seconds(<scheme>)]
[pc_applications(<scheme>)]
[inner_iterations_total(<scheme>), inner_iterations_average(<scheme>)]
[inner_solve_time_seconds(<scheme>), inner_solve_time_average_seconds(<scheme>)]
[fixed_jacobi_steps_total(<scheme>), fixed_jacobi_time_seconds(<scheme>)]
[residual(K), relative_residual(K)]
[residual(<scheme>), relative_residual(<scheme>)]
[L2_error(K), L2_error(<scheme>)]
```

条件数仍以理想精确 LOR 算子 `K_LOR^{-1}K` 或 `KK_LOR^{-1}` 标记；求解、时间、
真实残差与 L2 字段使用完整 `<scheme>` 名称，避免把实际近似逆误记成精确逆。

未预条件和预条件求解都记录 setup 与完整外层时间。`*_cg` 方案输出内层 CG 成本，
固定 Jacobi 方案输出应用次数、总步数和时间；这些子成本已经包含在预条件外层时间中，
不能二者相加。所有并行时间均取最大 rank 的 wall time。

矩阵存储字段始终输出；`estimated_minimum_bytes` 是 values、column indices 和 MPIAIJ
row pointers 的可移植下界，`memory_bytes` 是当前 PETSc 实现的报告值，某些版本可能
返回 0。方括号字段只在对应开关启用时出现。条件数来自独立 SLEPc SVD，不由 CG/GMRES
迭代投影估计。两个 L2 error 都针对同一
高阶系统 `K u=F`；LOR 系统只作为预条件器，不作为另一个 PDE 解进行误差比较。
质量矩阵接口仍由 Assembly 提供，但该 Laplace Driver 不请求它，因此运行时不会
产生高阶或 LOR 质量矩阵。
