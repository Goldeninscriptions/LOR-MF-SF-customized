# LOR-MF-SF 最小论文原型：当前开发指南

## 1. 项目定位

本项目是 `Matrix-free-IGA` 论文的单机 CPU 最小原型，研究二维/三维单 patch
高阶 B-spline 离散求解 Poisson/Laplace 方程时的 LOR 预条件效果，并为后续
matrix-free（MF）和 sum factorization（SF）实现提供清晰基线。它不是通用 IGA
框架，也不迁移旧仓库中的 CUDA、多 patch、MPI 分区和无关应用代码。

当前模型问题为

\[
-\Delta u=f\quad\text{in }\Omega,
\qquad u=0\quad\text{on }\partial\Omega.
\]

当前代码支持普通非有理、张量积、open-knot B-spline 几何；多个参数方向使用
相同次数 \(p\)，但每个方向始终保留独立 knot vector 和独立单元数 \(n_i\)。

## 2. 当前可运行的数据流

```text
geometry txt
  -> GeometryInput
  -> minimal BSplinePatch
  -> PRefinement / HRefinement / KRefinement
  -> refined analysis BSplinePatch
  -> boundary multi_index -> DirichletBC -> ID
  -> BSplineIEN -> BSplineElement -> LocalAssembly -> 按需组装 PETSc K / M / F
  -> GrevillePointSet / DemkoPointSet
  -> LORMesh -> LORIEN -> FEMQuad4/FEMHex8 -> 按需组装 PETSc K_LOR / M_LOR / F_LOR
  -> cg_lu / gmres_shell
  -> SLEPc condition numbers + L2 error + result log
```

`examples/assembled_driver.cpp` 是目前唯一主程序。它负责连接上述对象，不把基函数、
单元积分、网格拓扑或求解器算法重新实现在主程序中。

## 3. 当前已经实现的功能

### 3.1 几何、网格和细化

- 二维与三维单 patch 普通 B-spline 几何；
- 显式张量索引控制点输入，文件中的控制点书写顺序不影响内部排列；
- direction 0 最快变化的统一 lexicographic ordering；
- 边界保存为四条边或六个面的 basis `multi_index`；
- 几何保持的 Boehm knot insertion；
- 数学上嵌套的三种细化对象：
  - `PRefinement`：提高共同次数，并同步提高既有节点重复度，保持既有连续性和
    单元划分；
  - `HRefinement`：保持次数和已有节点重复度，仅插入兼容的新节点；
  - `KRefinement`：先 p-refinement，后 h-refinement，新节点具有升阶空间允许的
    最大连续性。

几何输入只描述最低必要几何。`square_2d.txt` 与 `cube_3d.txt` 分别从双线性和
三线性单元开始；曲边四边形由一个双三次多项式 B-spline patch 精确描述。

### 3.2 高阶 IGA 单元

- `BSplineBasis1D` 同时提供任意参数位置的 `evaluate_N(xi)` 和 Gauss 点上的
  `N_Gauss`/`N_der1_Gauss` 预计算表；
- 每个参数方向只构造一张一维积分表，二维/三维数据由 `BSplineElement` 做张量积；
- `IGAElement<Dim>` 提供只读的基函数、参数/物理梯度、坐标、Jacobian、
  inverse Jacobian、`det_Jac` 和 `quad_weight` 接口；
- `quad_weight` 与 `det_Jac` 分开保存，物理积分在 Assembly 中显式相乘；
- Element 不构造任何离散问题矩阵或向量。

### 3.3 LOR 网格和低阶单元

- `GrevillePointSet` 和 Remez 型 `DemkoPointSet`；
- 每个方向根据自己的 knot vector 独立生成 point set；
- `LORMesh` 用高阶几何映射只计算 LOR 顶点物理坐标；
- 单元内部几何由 `FEMQuad4` 或 `FEMHex8` 的 Q1 映射定义，因此 LOR 分片几何
  通常不等于高阶 patch 几何；
- `FEMQ1Basis` 保存所有低阶单元共享的 `[-1,1]^Dim` 参考表；
- 高阶空间和 LOR 空间分别拥有 `BSplineIEN` 与 `LORIEN`，但 coefficient-aligned
  点集使二者可以使用同一个 `ID` 代数编号。

### 3.4 边界、组装和求解

- `LexicographicOrdering` 独占 `multi_index <-> global_index` 转换；
- `DirichletBC` 只消费 Patch 给出的边界 `multi_index` 和一个 Ordering；
- `ID` 将全边界齐次 Dirichlet 自由度标为 `-1`，其余自由度连续编号；
- `LocalAssembly` 分别提供局部刚度、相容质量和荷载接口；每次调用只积分并分配
  被请求的一个局部对象；
- `LocalAssembly` 和 `GlobalAssembly` 还提供 `stiffness+load`、`mass+load` 与
  `stiffness+mass+load` 三种融合入口，在调用者需要多个对象时复用单元构造、
  Gauss 遍历、几何测度和索引转换；
- `GlobalAssembly` 对高阶 IGA 和 LOR Q1 使用相同的
  gather -> element -> local assembly -> IEN/ID scatter 结构，并分别提供刚度、
  质量和荷载入口；
- 当前 Driver 通过融合入口请求高阶刚度与荷载，并单独请求作为预条件器的 LOR
  刚度；质量矩阵和 LOR 荷载不会被隐式构造；
- `cg_lu`：外层 CG，LOR 矩阵用 LU 精确应用；
- `gmres_shell`：外层 restarted GMRES，`PCSHELL` 内部使用 CG+Jacobi 近似应用
  LOR 逆；
- SLEPc 分别计算 `cond_2(K)` 与 `cond_2(K_LOR^{-1}K)`；
- SVD 默认允许 100000 次迭代，并为最小奇异值使用最多 100 维搜索子空间；
  `-svd_max_it`、`-svd_ncv` 和 `-svd_tol` 可以逐次覆盖；
- 制造解模块和误差模块计算高阶数值解的物理域 L2 误差。

## 4. 当前限制

以下功能尚未实现，README 和论文实验不能把它们写成当前能力：

- matrix-free 高阶算子；
- sum-factorized 高阶算子；
- Element/Assembly 中实际启用的 OpenMP 并行循环；
- NURBS、有理权重和精确圆弧；
- 周期、非齐次 Dirichlet、Neumann 或 Robin 边界；
- 多 patch、分布式网格和 MPI 并行求解；
- k-/h-/p-refinement 的自动收敛阶分析；
- AMG 或不完全分解型 LOR 求解；
- 统一可切换的 `float/double/long double` 数值别名。当前离散核心使用 `double`，
  PETSc/SLEPc 接口使用其配置提供的标量与实数类型。

曲边 B-spline 输入、几何求值和三种细化已经可用，但当前制造解
\(u=\prod_d\sin(\pi(x_d-a_d)/(b_d-a_d))\) 只适用于轴对齐矩形/长方体。
`assembled_driver` 因此会主动拒绝曲边算例；需要为曲边边界设计零迹解析解后才能
进入同一求解流程。

## 5. 核心数学与索引约定

### 5.1 三类 index

- `multi_index`：保留张量积结构的二维/三维指标；
- `global_index`：由 `LexicographicOrdering` 得到的完整系数空间编号；
- `ID`：施加 Dirichlet 约束后的代数自由度编号，受约束项为 `-1`。

所有张量结构均令 direction 0 最快变化。其他模块不得自行复制正逆映射公式。

### 5.2 IEN

`IEN(e,a)` 把 element/cell 的局部 basis/node `a` 映射到完整空间的
`global_index`。IEN 不知道边界条件，Assembly 再通过 `ID(global_index)` 决定是否
散射到 PETSc 矩阵和向量。

### 5.3 积分测度

高阶单元的

```text
quad_weight(q) = tensor Gauss weight * reference-to-knot-span scale
det_Jac(q)     = det(dx/dxi)
```

低阶 Q1 单元的 `quad_weight` 是 `[-1,1]^Dim` 上的张量 Gauss 权重，`det_Jac` 是
`det(dx/dref)`。两者都由 LocalAssembly 手动相乘。

## 6. 细化模式和批处理含义

几何文件头部注释记录初始 \(p\) 与 \(n_i\)，但程序从 knot vector 自动推断，
不解析这些注释。

- `p_refine` 只允许改变 \(p\)，输入 `-n` 必须等于几何初始单元数；
- `h_refine` 只允许改变 \(n_i\)，输入 `-p` 必须等于几何初始次数；
- `k_refine` 可以同时指定目标共同次数和各方向目标单元数。

从最低阶矩形几何构造论文中的高连续性 \(p\ge3\)、多单元空间时，应使用
`k_refine`。固定最终 \(p\) 改变 \(n_i\) 仍可作为 h-study，尽管每个独立 case 都
从最小几何重新执行 k-refinement。

`work/run_batch.sh` 默认运行

```text
p = {3,4,5,6,7}
n = {(4,4),(16,16),(128,128)}
refinement = k_refine
```

point set 可选 `greville|demko`，LOR 求解模式可选 `cg_lu|gmres_shell`。脚本为每个
case 保存数值结果、控制台日志和批处理汇总；`BATCH_DRY_RUN=1` 只打印计划。

## 7. 目录和依赖关系

```text
include/IO, src/IO             输入、实验参数和结果输出
include/Mesh, src/Mesh         Patch、细化、point set、LOR mesh、IEN、ID
include/Element, src/Element   高阶与Q1局部基函数及几何积分数据
include/Assembly, src/Assembly ordering、边界条件、局部/全局组装
include/Solver, src/Solver     PETSc/SLEPc求解、条件数和误差
examples/                      当前assembled主程序
work/                          几何输入与运行脚本
conf/                          机器路径和PETSc/SLEPc查找
```

依赖方向保持为：

```text
IO -> Mesh
Element -> Mesh
Assembly -> Element + Mesh
Solver -> Assembly + Element + Mesh
Driver -> IO + Mesh + Assembly + Solver
```

Mesh 和 Element 不持有 PETSc 对象。PETSc 全局对象从 `GlobalAssembly` 开始出现。

## 8. 构建目标和第三方库

- `lor_core`：Mesh、Element、Geometry/Patch IO、Ordering、DirichletBC 和
  LocalAssembly；
- `lor_petsc`：PETSc GlobalAssembly；
- `lor_solver`：ExperimentIO、PETSc KSP、SLEPc 条件数和误差分析；
- `assembled_driver`：当前端到端程序。

PETSc 和 SLEPc 是当前主程序的必需依赖。OpenMP 可以由 CMake 查找并链接，但当前
核心循环尚未并行化。PETSc 在本机由 MPICH wrapper 构建，因此编译器使用
`mpicc/mpicxx` 保证 ABI 一致；程序仍按单进程原型运行。

机器相关路径位于被 Git 忽略的 `conf/system_lib_loading.cmake`，Linux 模板位于
`conf/system_lib_loading.linux.example.cmake`。可通过
`SYSTEM_LIB_LOADING_FILE` 指向仓库外的机器配置。

## 9. 开发和注释规则

- `.hpp` 必须说明类、公开接口和成员数据的数学意义、排列、所有权与不变量；
- `.cpp` 每个具有单一目的的连续代码段上方必须说明该段功能；
- Element 只保存与方程无关的局部数据，离散矩阵计算必须留在 Assembly；
- index 转换只能调用 `LexicographicOrdering`；
- IEN/ID 属于 Mesh/约束层，不回填到 Element；
- 每个方向即使次数相同，也不得假设 knot vector 相同；
- 新优化必须先与 assembled 参考实现比较数值结果；
- 修改接口、文件职责或当前能力后，同步更新本指南和对应目录 README。

## 10. 下一阶段

当前 assembled + LOR 基线已经贯通。下一阶段合理顺序是：

1. 固化小规模数值回归检查，包括细化前后几何不变性、矩阵对称性、正 Jacobian、
   制造解误差和两种 point set；
2. 把高阶算子作用从全局组装中抽象出来，增加朴素 matrix-free 参考实现；
3. 实现张量 sum factorization，并逐向量比较 assembled、MF、MF+SF；
4. 在串行结果一致后加入 OpenMP，并分别记录 setup、operator apply、LOR apply 和
   总求解时间；
5. 为曲边几何设计与全边界齐次 Dirichlet 相容的制造解。

各目录中每个头文件和源文件的当前职责见对应 `README.md`。
