# Build configuration files

`conf/` 将机器路径与通用构建逻辑分开。这里的 MPI compiler wrapper 是为了匹配
PETSc 的构建 ABI，不表示当前代码实现了分布式计算。

| 文件 | 当前职责 |
|---|---|
| `system_lib_loading.cmake` | 被 `.gitignore` 排除的本机私有配置；保存当前机器的 PETSc、SLEPc、编译器和 OpenMP 绝对路径 |
| `system_lib_loading.linux.example.cmake` | 迁移到 Linux 工作站时可复制/覆盖的路径模板 |
| `project_build_setup.cmake` | 选择机器配置和兼容编译器，设置 C++17/Release 默认值，按需发现 PETSc、SLEPc、OpenMP、Eigen，并打印配置摘要 |
| `FindPETSc.cmake` | 从 `PETSC_DIR/PETSC_ARCH` 查找 headers/library，创建 imported target `LOR::PETSc` |
| `FindSLEPc.cmake` | 查找与当前 PETSc 匹配的 SLEPc，创建依赖 `LOR::PETSc` 的 `LOR::SLEPc` |
| `petsc_compat.hpp` | 兼容 PETSc 3.19 前后的 null-pointer 宏名称；当前主线很少需要直接包含它 |

新 clone 可以复制 Linux 示例为本机的 `system_lib_loading.cmake`，或者使用仓库外配置：

```bash
cmake ../examples \
  -DSYSTEM_LIB_LOADING_FILE=/absolute/path/to/system_lib_loading.cmake
```

不要把另一台机器的绝对路径写进通用 CMake 查找文件。
