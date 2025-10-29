# 紧束缚模型计算工具包 (TB_model)

## 项目概述

这是一个基于JAX和NumPy的紧束缚模型(Tight-Binding Model)计算工具包，专为凝聚态物理和材料科学研究设计。该工具包提供了高效的电子结构计算功能，特别利用了JAX的自动微分和JIT编译能力，支持GPU加速，适用于大规模k空间积分和物理量计算。

## 主要功能

- 构建和处理紧束缚哈密顿量
- 计算k空间中的能量本征值和本征向量
- 计算态密度(DOS)和部分态密度(PDOS)
- 使用Kubo公式计算电导率等输运性质
- 支持费米能级扫描计算
- 构建k网格和高对称路径用于能带结构计算
- 内置多种物理算符(S、L、LS耦合等)
- 支持GPU并行计算和批处理

## 项目结构

```
TB_model/
├── TB_model.py      # 核心紧束缚模型类定义
├── TB_io.py         # 模型输入输出功能
├── calculators.py   # 各种物理量计算器实现
├── K_space.py       # k空间相关功能
├── operators.py     # 物理算符定义
├── utility.py       # 辅助功能和工具函数
├── run.py           # 示例运行脚本
├── TB.npz           # 紧束缚模型数据文件
├── operator.npz     # 预定义算符数据文件
└── documentation.md # 项目文档
```

## 核心模块说明

### 1. TB_model.py

定义了`TB_model`核心数据类，包含紧束缚模型的所有基本参数和方法：

- **数据属性**：
  - `HR`: 实空间哈密顿量 (nR, nwan, nwan)
  - `cell_center`: 实空间格点位置 (nR, 3)
  - `real_lattice`: 实空间晶格基矢 (3, 3)
  - `recip_lattice`: 倒空间晶格基矢 (3, 3)
  - `center_list`: Wannier函数中心位置 (nwan, 3)
  - `AA_R`: 实空间规范势 (nR, nwan, nwan, 3)

- **主要方法**：
  - `set_EF(file_name)`: 设置费米能级
  - `set_E_fermis(E_btm, E_top, interval)`: 设置费米能级扫描范围
  - `add_operator_kubo(operator, name)`: 添加用于Kubo计算的算符
  - `add_operator_inner(operator, name)`: 添加用于内积计算的算符
  - `make_k_hamiltonian_calculator()`: 创建k空间哈密顿量计算器
  - `make_calc_inner(kbT)`: 创建内积计算器(如DOS)
  - `make_calc_kubo(alpha, beta, Gamma, kbT)`: 创建Kubo公式计算器
  - `make_k_mesh(nkx, nky, nkz)`: 构建k网格
  - `make_k_path(k_points, ticks_label, length)`: 构建高对称k路径
  - `run_all(batch_size, tabulate)`: 运行所有计算

### 2. calculators.py

实现了各种物理量的计算功能：

- `make_k_hamiltonian_calculator(model)`: 创建计算k空间哈密顿量的函数
- `make_k_hamiltonian_velocity_calculator(model, use_A)`: 创建计算哈密顿量和速度算符的函数
- `make_calc_inner(model, kbT)`: 创建计算内积的函数(可用于DOS计算)
- `make_calc_kubo(model, alpha, beta, Gamma, kbT, use_A)`: 创建Kubo公式计算器，用于计算电导率等输运性质

### 3. K_space.py

处理k空间相关操作：

- `build_k_path(model, k_points, ticks_label, length)`: 构建高对称k路径用于能带计算
- `build_k_mesh(model, nkx, nky, nkz)`: 构建均匀k网格用于积分计算

### 4. operators.py

定义各种物理算符：

- `operator_matrix_all`类：管理轨道算符
  - 支持s、p、d、f轨道
  - 内置自旋(S)、轨道角动量(L)、自旋-轨道耦合(LS)等算符
  - 提供投影算符功能

### 5. utility.py

提供辅助功能：

- `fermi_dirac(E, mu, kBT)`: 费米-狄拉克分布函数
- `fermi_dirac_vamp`: 向量化的费米分布函数
- `cal_center_matrix(center_list)`: 计算Wannier中心差异矩阵
- `find_EF(file_name)`: 从量子化学计算输出文件中提取费米能级

### 6. TB_io.py

处理模型的输入输出：

- `read_TB_model(file_name)`: 从npz或pickle文件读取紧束缚模型
- `save_TB_model(model, file_name)`: 保存模型到pickle文件

### 7. run.py

提供示例运行脚本：

- `run_kps(kps, calculator, batch_size, tabulate)`: 批量处理k点计算
- 示例展示如何设置模型、添加算符、创建计算器和运行计算

## 使用指南

### 基本使用流程

以下是推荐的简化使用流程，基于TB_model类提供的易用接口：

```python
from TB_io import read_TB_model
import operators as ops
import jax.numpy as jnp

# 1. 加载模型
model = read_TB_model("TB.npz")

# 2. 设置费米能级
model.set_EF("Cu.scf.out")  # 从SCF输出文件读取费米能级
model.set_E_fermis(-10, 8, 0.05)  # 设置费米能级扫描范围

# 3. 添加算符
# 添加内积算符(用于DOS计算)
model.add_operator_inner(jnp.eye(model.nwan), "cumDOS")

# 添加Kubo计算用算符
matrix_all = ops.operator_matrix_all(['spd'])
model.add_operator_kubo(matrix_all.pre_prepared_operators["Lz"], "Lz")

# 4. 创建计算器
model.make_calc_inner(kbT=0.026)  # 创建内积计算器
model.make_calc_kubo(kbT=0.026)   # 创建Kubo计算器

# 5. 设置k网格
model.make_k_mesh(nkx=100, nky=100, nkz=100)

# 6. 一键运行所有计算 (推荐的简化方法)
results = model.run_all(batch_size=1000, tabulate=False)
```

> **提示**：`run_all()`方法是推荐的简化接口，它会自动处理所有已添加的算符和计算器，无需手动传递k点和计算器。

### 替代方式：单独运行计算

如需单独运行特定计算，也可以使用以下方式：

```python
from run import run_kps

# 获取模型中的k网格
_, k_points = model.make_k_mesh(nkx=100, nky=100, nkz=100)
k_mesh_flatten = k_points.reshape(-1, 3)

# 单独运行Kubo计算
kubo_calculator = model.make_calc_kubo(kbT=0.026)
res_kubo = run_kps(k_mesh_flatten, kubo_calculator, batch_size=1000, tabulate=False)
```

### 高级功能

#### 1. 投影态密度(PDOS)计算

通过添加特定轨道的投影算符，可以计算部分态密度：

```python
# 创建s轨道的投影算符
P_s = matrix_all.make_P_easy("s")
model.add_operator_inner(P_s, "s_PDOS")
```

#### 2. 高对称路径能带计算

构建高对称k路径并计算能带：

```python
# 定义高对称点 (以立方晶体为例)
k_points = jnp.array([
    [0.0, 0.0, 0.0],    # Γ
    [0.5, 0.0, 0.0],    # X
    [0.5, 0.5, 0.0],    # M
    [0.0, 0.0, 0.0],    # Γ
])
ticks_label = ["Γ", "X", "M", "Γ"]

# 构建k路径
k_path, k_path_x, ticks, ticks_label = model.make_k_path(
    k_points, ticks_label, length=100
)
```

#### 3. GPU加速

该工具包自动检测并使用可用的GPU：

```python
# 在run_kps函数中会自动检测GPU
# 确保batch_size能被GPU数量整除
batch_size = 1000  # 调整以适应您的GPU内存
```

## 技术特点

1. **高性能计算**：利用JAX的JIT编译和自动向量化实现高效计算
2. **GPU支持**：内置GPU并行计算能力，可显著加速大规模计算
3. **模块化设计**：清晰的代码结构和模块化设计，便于扩展和修改
4. **批处理计算**：支持大批量k点并行计算，可有效管理内存使用
5. **丰富的算符库**：内置多种物理算符，支持自定义算符

## 依赖项

- Python 3.11+
- JAX 2.0+
- NumPy
- tqdm (进度显示)
- Matplotlib (可选，用于可视化)

## 性能优化建议

1. 对于大规模计算，调整`batch_size`以平衡计算速度和内存使用
2. 使用GPU时，确保batch_size是GPU数量的倍数
3. 对于高分辨率k网格，考虑减小batch_size避免内存溢出
4. 对于只需要特定能量范围的计算，适当设置费米能级扫描范围

## 示例应用

1. 计算材料的电子结构和能带
2. 计算态密度和部分态密度
3. 使用Kubo公式计算电导率等输运性质
4. 研究自旋-轨道耦合效应
5. 分析材料的轨道特征和对称性

## 注意事项

1. 确保TB.npz文件包含正确格式的紧束缚模型数据
2. 使用GPU时注意内存限制，必要时减小batch_size
3. 对于低温计算，设置合适的kbT值以避免数值不稳定
4. 在使用Kubo公式计算时，调整Gamma参数以获得合适的展宽

## 扩展与开发

该工具包设计为可扩展的，您可以通过以下方式进行扩展：

1. 添加新的物理算符到`operators.py`
2. 实现新的计算器功能到`calculators.py`
3. 扩展`TB_model`类以支持新的模型特性
4. 开发新的输入/输出功能以支持更多文件格式

## 许可证

MIT许可证

---

## 联系方式

如有问题或建议，请联系项目维护者。
wenchenyue5678@buaa.edu.cn