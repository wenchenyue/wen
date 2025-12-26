# 紧束缚模型计算工具包 (TB_model)

## 项目概述

这是一个基于JAX和NumPy的紧束缚模型(Tight-Binding Model)计算工具包，专为凝聚态物理和材料科学研究设计。该工具包提供了高效的电子结构计算功能，特别利用了JAX的自动微分和JIT编译能力，支持GPU加速，适用于大规模k空间积分和物理量计算。

## 主要功能

- 构建和处理紧束缚哈密顿量
- 计算k空间中的能量本征值和本征向量
- 计算态密度(DOS)和部分态密度(PDOS)
- 使用Kubo公式计算电导率等输运性质
- 计算轨道磁化率（现代理论）
- 计算轨道贝里曲率(OBC)
- 支持费米能级扫描计算
- 构建k网格和高对称路径用于能带结构计算
- 内置多种物理算符(S、L、LS耦合等)
- 支持GPU并行计算和批处理
- 支持规范势(AA_R)项

## 项目结构

```
TB_model/
├── TB_model.py      # 核心紧束缚模型类定义
├── TB_io.py         # 模型输入输出功能
├── calculators.py   # 各种物理量计算器实现
├── K_space.py       # k空间相关功能
├── operators.py     # 物理算符定义
├── utility.py       # 辅助功能和工具函数
├── Hk.py            # 哈密顿量计算模块
├── run.py           # 计算执行模块
├── get_WB.py        # WannierBerri数据读取工具
├── job.py           # 示例运行脚本
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
  - `EF`: 费米能级
  - `E_fermis`: 费米能级扫描范围
  - `k_points`: k点集合
  - `k_mesh`: k网格

- **主要方法**：
  - `set_EF(file_name)`: 从文件设置费米能级
  - `set_E_fermis(E_btm, E_top, interval)`: 设置费米能级扫描范围
  - `add_calculator(calculator, name)`: 添加计算器
  - `clear_calculator()`: 清除所有计算器
  - `make_k_hamiltonian_calculator()`: 创建k空间哈密顿量计算器
  - `make_k_hamiltonian_velocity_calculator()`: 创建哈密顿量速度计算器
  - `make_calc_inner(name, operator)`: 创建内积计算器(如DOS)
  - `make_calc_kubo(name, alpha, beta, Gamma, Operator)`: 创建Kubo公式计算器
  - `make_calc_OBC(name, alpha, beta, gamma, Gamma)`: 创建轨道贝里曲率计算器
  - `make_calc_all(kbT, use_A)`: 创建统一计算器
  - `make_k_mesh(nkx, nky, nkz)`: 构建k网格
  - `make_k_path(k_points, ticks_label, length)`: 构建高对称k路径
  - `run_all(job_name, batch_size, tabulate)`: 运行所有计算

### 2. calculators.py

实现了各种物理量的计算功能：

- `make_calc_inner(model, Operator)`: 创建计算内积的函数(可用于DOS计算)
  - 支持计算各种物理量的内积，如态密度(DOS)和部分态密度(PDOS)
  - 通过添加不同的算符，可以灵活计算各种可观测量

- `make_calc_mag_modern(model, gamma)`: 使用现代理论计算轨道磁化率
  - 基于Pezo等人的理论 [PHYSICAL REVIEW B 106, 104414 (2022)]
  - 计算内禀轨道磁化矩
  - 公式：M_orb(n,k) = Im <∂u(n,k)/∂k|x (H_k - E_nk)|∂u(n,k)/∂k>

- `make_calc_mag_matrix(model, gamma)`: 计算轨道磁化矩阵
  - 计算轨道磁化矩阵元素 M_orb[n,m](k, gamma)
  - 基于Pezo等人的公式

- `make_calc_OBC_modern(model, alpha, beta, gamma, Gamma)`: 计算轨道贝里曲率
  - 用于轨道霍尔电导率(OHC)计算
  - 使用现代轨道磁化理论
  - 基于Pezo等人的公式 [PHYSICAL REVIEW B 106, 104414 (2022)]

- `make_calc_kubo(model, alpha, beta, Gamma, Operator)`: 创建Kubo公式计算器
  - 用于计算电导率等输运性质
  - alpha和beta参数定义了电导率张量的方向
  - Gamma参数控制展宽以避免数值发散
  - Operator参数可选，用于计算特定算符的贝里曲率

- `make_calc_all(model, kbT, use_A)`: 集成所有计算功能的统一接口
  - 整合了内积计算和Kubo公式计算
  - 支持费米能级扫描(kbT参数)
  - 使用use_A参数控制是否在速度算符中包含规范势项

- `make_calc_mag_scan(model, kbT, use_A, gamma)`: 磁化扫描计算器
  - 对费米能级进行扫描，计算磁化率
  - 将磁化计算分为能量依赖项和常数项

### 3. Hk.py

哈密顿量计算模块：

- `make_k_hamiltonian_calculator(model)`: 创建k空间哈密顿量计算器
  - 计算H(k) = Σ_R H(R) e^{ik·(R+τ_ij)}

- `make_calc_AA_k(model)`: 计算k空间规范势AA_k
  - 用于Wannier规范下的位置算符

- `make_k_hamiltonian_velocity_calculator(model, use_A)`: 创建哈密顿量速度计算器
  - 计算H(k)和速度算符v(k) = ∂H(k)/∂k
  - use_A参数控制是否包含规范势项

### 4. K_space.py

处理k空间相关操作：

- `build_k_path(model, k_points, ticks_label, length)`: 构建高对称k路径用于能带计算
- `build_k_mesh(model, nkx, nky, nkz)`: 构建均匀k网格用于积分计算

### 5. operators.py

定义各种物理算符：

- `operator_matrix_all`类：管理轨道算符
  - 支持s、p、d、f轨道
  - 内置自旋(S)、轨道角动量(L)、自旋-轨道耦合(LS)等算符
  - 提供投影算符功能
  - 方法：
    - `make_P_easy(select_name)`: 创建投影算符
    - `make_POP_easy(select_name, pre_perpared_name)`: 创建投影算符与预定义算符的组合
    - `make_POP_pre_prepared(P_mat, pre_perpared_name)`: 使用自定义投影算符

- `build_operator_all(orbital_list)`: 构建轨道算符列表
- `build_operator_LS(orbital_list, use_LS_base)`: 构建自旋和轨道角动量算符
  - 支持LS耦合基或实原子轨道基

### 6. utility.py

提供辅助功能：

- `fermi_dirac(E, mu, kBT)`: 费米-狄拉克分布函数
- `fermi_dirac_vamp`: 向量化的费米分布函数
- `fermi_dirac_derivative(E, mu, kBT)`: 费米-狄拉克分布的导数
- `fermi_dirac_derivative_vamp`: 向量化的费米分布导数
- `cal_center_matrix(center_list)`: 计算Wannier中心差异矩阵
- `find_EF(file_name)`: 从量子化学计算输出文件中提取费米能级

### 7. TB_io.py

处理模型的输入输出：

- `read_TB_model(file_name)`: 从npz文件读取紧束缚模型
- `save_TB_model(model, file_name)`: 保存模型到文件

### 8. run.py

提供计算执行功能：

- `run_kps(kps, calculator, batch_size, tabulate)`: 批量处理k点计算
  - 支持GPU并行计算
  - 支持批处理以管理内存
  - tabulate参数控制是否保存每个k点的结果

### 9. get_WB.py

WannierBerri数据读取工具：

- 从WannierBerri的_tb.dat文件读取数据
- 提取HR、cRvec、real_lattice、recip_lattice、Wan_center、AA_R等数据
- 保存为npz格式供TB_model使用

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
# 或直接设置
model.EF = 12.43
model.set_E_fermis(-10, 8, 0.05)  # 设置费米能级扫描范围

# 3. 清除之前的计算器（可选）
model.clear_calculator()

# 4. 添加算符和计算器
# 添加内积算符(用于DOS计算)
model.make_calc_inner("cumDOS", jnp.eye(model.nwan))

# 添加Kubo计算用算符（如轨道霍尔电导率）
matrix_all = ops.operator_matrix_all(['spd'])
Lz = matrix_all.pre_prepared_operators["Lz"]
model.make_calc_kubo("OHC_z", alpha=0, beta=1, Gamma=0.025, Operator=Lz)

# 添加轨道贝里曲率计算器
model.make_calc_OBC("OBC_z", alpha=0, beta=1, gamma=2, Gamma=0.025)

# 5. 创建统一计算器（推荐）
model.make_calc_all(
    kbT=0.026,        # 温度，单位eV，室温约为0.026 eV
    use_A=False       # 是否在速度算符中包含规范势项
)

# 6. 设置k网格
model.make_k_mesh(nkx=100, nky=100, nkz=100)

# 7. 一键运行所有计算 (推荐的简化方法)
results = model.run_all("job_name", batch_size=1000, tabulate=False)
```

### 示例：计算轨道霍尔电导率

```python
from TB_io import read_TB_model
import operators as ops

# 加载模型
model = read_TB_model("TB_Cu.npz")
model.EF = 12.43
model.set_E_fermis(-10, 8, 0.05)

# 创建轨道算符
matrix_all = ops.operator_matrix_all(['spd'])
Lz = matrix_all.pre_prepared_operators['Lz']

# 清除之前的计算器
model.clear_calculator()

# 添加轨道霍尔电导率计算器
BC_calculator = model.make_calc_kubo(
    "OHC_z",
    alpha=0,      # x方向
    beta=1,       # y方向
    Gamma=0.025,  # 展宽
    Operator=Lz   # 使用Lz算符
)

# 创建统一计算器
model.make_calc_all(kbT=0.003, use_A=False)

# 设置k网格
model.make_k_mesh(80, 80, 80)

# 运行计算
model.run_all("Cu", batch_size=100, tabulate=False)
```

### 高级功能

#### 1. 投影态密度(PDOS)计算

通过添加特定轨道的投影算符，可以计算部分态密度：

```python
# 创建s轨道的投影算符
P_s = matrix_all.make_P_easy("s")
model.make_calc_inner("s_PDOS", P_s)

# 创建p轨道的投影算符
P_p = matrix_all.make_P_easy("p")
model.make_calc_inner("p_PDOS", P_p)
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

#### 3. 轨道磁化率计算

使用现代理论计算轨道磁化率：

```python
# 创建轨道磁化率计算器
model.make_calc_OBC("OBC_z", alpha=0, beta=1, gamma=2, Gamma=0.025)

# 或使用磁化扫描计算器
mag_scan_calculator = model.make_calc_mag_scan(kbT=0.003, use_A=False, gamma=2)
model.add_calculator(mag_scan_calculator, "mag_scan")
```

#### 4. 使用规范势项

在速度算符中包含规范势项：

```python
# 创建包含规范势的速度算符
model.make_calc_all(kbT=0.026, use_A=True)
```

#### 5. GPU加速

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
6. **现代理论支持**：支持现代轨道磁化理论和轨道贝里曲率计算
7. **规范势支持**：支持在速度算符中包含规范势项

## 依赖项

- Python 3.11+
- JAX 2.0+
- NumPy
- tqdm (进度显示)
- Matplotlib (可选，用于可视化)
- WannierBerri (可选，用于数据转换)

## 性能优化建议

1. 对于大规模计算，调整`batch_size`以平衡计算速度和内存使用
2. 使用GPU时，确保batch_size是GPU数量的倍数
3. 对于高分辨率k网格，考虑减小batch_size避免内存溢出
4. 对于只需要特定能量范围的计算，适当设置费米能级扫描范围
5. 使用JIT编译的函数可以显著提高计算速度

## 示例应用

1. 计算材料的电子结构和能带
2. 计算态密度和部分态密度
3. 使用Kubo公式计算电导率等输运性质
4. 计算轨道霍尔电导率
5. 研究轨道磁化率
6. 计算轨道贝里曲率
7. 研究自旋-轨道耦合效应
8. 分析材料的轨道特征和对称性

## 注意事项

1. 确保TB.npz文件包含正确格式的紧束缚模型数据
2. 使用GPU时注意内存限制，必要时减小batch_size
3. 对于低温计算，设置合适的kbT值以避免数值不稳定
4. 在使用Kubo公式计算时，调整Gamma参数以获得合适的展宽
5. 规范势项(AA_R)的使用会影响速度算符的计算结果
6. 对于复杂的计算，建议先在小规模k网格上测试

## 扩展与开发

该工具包设计为可扩展的，您可以通过以下方式进行扩展：

1. 添加新的物理算符到`operators.py`
2. 实现新的计算器功能到`calculators.py`
3. 扩展`TB_model`类以支持新的模型特性
4. 开发新的输入/输出功能以支持更多文件格式
5. 添加新的k空间采样方法

## 许可证

MIT许可证

---

## 联系方式

如有问题或建议，请联系项目维护者。
wenchenyue5678@buaa.edu.cn

## 参考文献

1. Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022) - 现代轨道磁化理论
2. WannierBerri - Wannier插值和贝里相计算工具包