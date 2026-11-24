# Derivative-space Greville mass 刚度预条件器：仓库设计映射

本文记录 `TECHNICAL_SPEC.md` v1.0 在当前仓库中的最小增量实现。数学定义以该规范为唯一依据。

## 1. 复用的仓库组件

- `Element/BSplineBasis1D`：局部非零 B-spline 值、Gauss 表；新增任意点值和一阶参数导数的联合求值。
- `Element/GaussQuadrature`：独立高阶刚度和精确导数空间质量积分。
- `Assembly/BandedMassCholesky1D`：复用独立的一维 B-spline mass/stiffness 积分作为参考，不从新预条件器反推 `K_s`。
- `Assembly/LexicographicOrdering`、`Mesh/ID`、`DirichletBC`：现有 benchmark 的 direction-0-fastest 编号和原始系数空间边界消元保持不变。
- PETSc AIJ、`MatMatMult`、`MatPtAP`：显式稀疏形成 `B=T(DR)` 与 `P=B^T M_g B`，从不形成显式逆。
- `Solver/GeneralizedConditionNumber`：小型实验使用 SPD GHEP `K v=lambda P v`。
- `LinearSolver`/现有 assembled benchmark：新矩阵源通过 `derivative_greville_mass` 接入，非齐次 Dirichlet lift 与右端移项仍走原路径。

## 2. 新模块

公开头文件：`include/Assembly/DerivativeGrevilleStiffness.hpp`

实现文件：`src/Assembly/DerivativeGrevilleStiffness.cpp`

主要 API：

```cpp
assemble_derivative_greville_stiffness_1d(...)
assemble_derivative_greville_stiffness_2d(...)
destroy_derivative_greville_result(...)
derivative_space_knots(...)
strict_greville_points(...)
derivative_greville_id_2d(...)
```

一维结果保留 `D,R,DR,T,M_g,M_exact,B,P,K_s`；二维结果分别保留 x/y 两套 `D,T,M_g,M_exact,B,P_component`。生产 benchmark 可关闭参考矩阵、精确质量阵并在形成 `P` 后释放中间矩阵。

## 3. 数学和尺寸

一维：

```text
n = E+p
D:       (n-1) x n
R:       n x (n-2)
T, Mg:   (n-1) x (n-1)
B:       (n-1) x (n-2)
P, K:    (n-2) x (n-2)
```

代码逐行使用

```text
D[a,a]   = -p/(t[a+p+1]-t[a+1])
D[a,a+1] =  p/(t[a+p+1]-t[a+1])
```

导数 knot vector 严格删除原 knot vector 首尾各一个 knot。`T` 在该 `p-1` 次空间自身的 Greville 点上组装，`M_g` 是这些点之间的 P1 consistent mass。

二维固定：

```text
id(i,j) = i + nx*j
Dx: ((nx-1)ny) x (nx ny)
Dy: (nx(ny-1)) x (nx ny)
Tx, Mgx: ((nx-1)ny) square, degree (p-1,p)
Ty, Mgy: (nx(ny-1)) square, degree (p,p-1)
P, K: ((nx-2)(ny-2)) square
```

`R` 只删除原始 `(p,p)` 系数空间的四周边界。导数空间的所有行、Greville 顶点及 Q1 mass 节点全部保留。

二维 `T_x,T_y` 通过局部基函数值乘积逐行稀疏插入，但没有组装任何全局 Kronecker 矩阵。两套 Q1 consistent mass 通过四节点矩形单元阵直接装配。

物理矩形使用参考域 knot：

```text
w_x = Ly/Lx
w_y = Lx/Ly
```

同一权重同时用于独立高阶 `K_s` 和新 `P`。`Lx=2,Ly=1` 时为 `1/2,2`。

## 4. 高阶参考矩阵

- 一维 `K_s` 复用独立 Gauss 导数积分。
- 二维 `K_s` 逐 knot-span、逐 Gauss 点直接装配；每方向采用 `p+1` 点。
- 二维精确混合次数质量阵也逐单元直接装配，仅用于验证
  `K=D_x^T W_x D_x+D_y^T W_y D_y`。
- 这些路径不复用 Greville Q1 mass，避免构造与参考共享同一错误。

## 5. benchmark 集成

`-preconditioner_matrix_source derivative_greville_mass` 是新的明确选项：

- 当前 assembled benchmark 只允许二维、`p>=2`、同次数、坐标对齐仿射矩形。
- 完整 control net 会与 Greville 仿射坐标核对；曲边和非对角几何明确拒绝。
- 新 `P` 与现有约化高阶 `K_s` 使用相同 x-fast 自由度顺序。
- 原 benchmark 的非齐次边界系数和 `f_f-K_fb c_b` 逻辑未修改。
- 可继续选择现有 LU、CG、AMG 等矩阵预条件器应用方式。

## 6. 验证程序

`derivative_greville_validation` 编写了以下检查（本轮未执行）：

- 一维、二维所有规定尺寸；
- `D,Dx,Dy` 每行恰好两个非零项；
- 配置矩阵局部支撑上界；
- 任意点精确导数表达；
- 原始 p 次 Greville 间距与差分尺度 `delta_a`；
- 一维 `K=D^T M_exact D`；
- 一维 `p=2` 的 `P=K`；
- 一维 `p=2` 的 SPD GHEP 条件数约为 1；
- 二维非正方形 `(Ex,Ey)=(4,7)` 的 x/y stride；
- 二维 `Lx=2,Ly=1` 的 `1/2,2` 权重；
- 二维两个 Q1 分量能量之和等于 `c^T P c`；
- 二维精确混合质量恒等式；
- `K,P` 对称正定；
- 测试内部临时构造已删除旧公式作为 negative control，确认 `p>=3` 的新 `P` 不会误走 `T_p^T K_g T_p`；该旧公式不作为任何 public API 或 solver option 暴露；
- 二维 `p=2` 只验证能量和 SPD，不断言 `P=K`。

推荐容差：代数恒等式和能量检查 `2e-12` 到 `5e-11`；任意点导数检查 `5e-12` 到 `1e-11`。GHEP 是可选 SLEPc 验证，由 drivers 的 `-compute_spectrum` 控制。

## 7. 有意保留的限制

- 仅 1D/2D；无 3D。
- 显式稀疏组装；无 matrix-free、Kronecker、FDM、ADI。
- 二维只处理坐标对齐仿射矩形和对角常系数度量。
- 无 Neumann 零空间、变系数、一般 knot multiplicity 或 `p=1`。
- 不声称条件数对 `p` 或 `h` 无关；扫描脚本只负责产生数据。
- standalone driver 的直接 Cholesky GHEP/CG 分析当前要求单 rank；矩阵装配和 assembled benchmark 仍使用 PETSc communicator。
