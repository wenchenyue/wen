# TB_model 紧束缚模型计算工具包

## 项目概述

TB_model 是一个基于 JAX 和 NumPy 的紧束缚模型(Tight-Binding Model)计算工具包，专为凝聚态物理和材料科学研究设计。该工具包利用 JAX 的 JIT 编译和自动向量化实现高效计算，支持 GPU 加速，适用于大规模 k 空间积分和物理量计算。

## 主要功能

- 构建和处理紧束缚哈密顿量
- 计算 k 空间中的能量本征值和本征向量
- 计算态密度(DOS)和部分态密度(PDOS)
- 使用 Kubo 公式计算电导率等输运性质
- 计算轨道磁化率（现代理论）
- 计算轨道贝里曲率(OBC)和轨道霍尔电导率(OHC)
- 支持费米能级扫描计算
- 构建k网格和高对称路径用于能带结构计算
- 内置多种物理算符（S、L、LS耦合等）
- 支持 GPU 并行计算和批处理
- 支持规范势(AA_R)项

## 项目结构

```
TB_model/
├── TB_model.py       # 核心紧束缚模型类定义
├── TB_io.py          # 模型输入输出功能
├── calculators.py    # 各种物理量计算器实现
├── K_space.py        # k空间相关功能
├── operators.py      # 物理算符定义
├── utility.py        # 辅助功能和工具函数
├── Hk.py             # 哈密顿量计算模块
├── run.py            # 计算执行模块
├── tb_parser.py      # 纯 NumPy *_tb.dat 解析器（推荐）
├── get_WB.py         # WannierBerri 数据读取工具（可选）
├── job.py            # 示例运行脚本
├── TB.npz            # 紧束缚模型数据文件
├── operator.npz      # 预定义算符数据文件
└── documentation.md  # 项目文档
```

## 核心模块说明

### 1. TB_model.py

定义了 `TB_model` 核心数据类，包含紧束缚模型的所有基本参数和方法。

#### 数据属性

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
- `all_calculators`: 所有已注册的计算器字典

#### 主要方法

- `set_EF(file_name)`: 从文件设置费米能级
- `set_E_fermis(E_btm, E_top, interval)`: 设置费米能级扫描范围
- `add_calculator(calculator, name)`: 添加计算器
- `clear_calculator()`: 清除所有计算器
- `make_k_hamiltonian_calculator()`: 创建k空间哈密顿量计算器
- `make_k_hamiltonian_velocity_calculator()`: 创建哈密顿量速度计算器
- `make_calc_inner(name, operator, H_gauge)`: 创建内积计算器
- `make_calc_kubo(name, alpha, beta, Gamma, Operator, H_gauge)`: 创建Kubo公式计算器
- `make_calc_OBC(name, alpha, beta, gamma, Gamma, H_gauge)`: 创建轨道贝里曲率计算器
- `make_calc_all(kbT, use_A, scanner_type)`: 创建统一计算器
- `make_k_mesh(nkx, nky, nkz)`: 构建k网格
- `make_k_path(k_points, ticks_label, length)`: 构建高对称k路径
- `run_all(job_name, batch_size, tabulate)`: 运行所有计算

### 2. calculators.py

实现了各种物理量的计算功能。

#### make_calc_inner

计算算符与波函数的内积 `<ψ_n(k)|O|ψ_n(k)>`，用于计算态密度和部分态密度。

#### make_calc_kubo

实现 Kubo 公式计算线性响应系数，可计算：
- 贝里曲率（当 Operator=None 时）
- 算符相关的贝里曲率（如自旋霍尔电导率）

公式：
```
Ω_n(α,β) = -2 * Im[ Σ_{m≠n} <n|J_α|m><m|v_β|n> / ((E_n - E_m)^2 + Γ^2) ]
```

#### make_calc_mag_matrix

计算完整的轨道磁化矩阵 M_orb[n,m](k, gamma)，基于 Pezo 等人理论 (PRB 106, 104414, 2022)。

#### make_calc_intra

计算带内贡献，公式为：
```
O_intra(k,n,β) = <u(n,k)|v_β|u(n,k)> * <u(n,k)|O|u(n,k)>
```
需要配合 `scanner_type='FD_deri'` 使用。

#### make_calc_all

创建统一计算器，自动计算所有已注册的计算器：
- 支持费米-狄拉克分布扫描
- 支持两种扫描模式：
  - `scanner_type="FD"`: 标准费米-狄拉克分布
  - `scanner_type="FD_deri"`: 费米-狄拉克分布导数（用于费米面积分）

#### make_calc_mag_scan

轨道磁化扫描计算器，将磁化计算分为能量依赖项和常数项两部分。

### 3. Hk.py

#### make_k_hamiltonian_calculator

创建k空间哈密顿量计算器：
```
H(k) = Σ_R e^{ik·(R+τ_j-τ_i)} H_{ij}(R)
```

#### make_calc_AA_k

计算k空间规范势 AA_k，用于Wannier规范下的位置算符。

#### make_k_hamiltonian_velocity_calculator

创建哈密顿量速度计算器：
```
v_α = ∂H(k)/∂k_α = i Σ_R e^{ik·(R+τ_ji)} (R_α + τ_ji,α) H(R)
```
支持通过 `use_A` 参数添加轨道霍尔效应中的异常位置项。

### 4. K_space.py

#### build_k_path

构建高对称k路径用于能带计算。

#### build_k_mesh

构建均匀k网格用于积分计算。

### 5. operators.py

#### operator_matrix_all

管理轨道算符的类：
- 支持s、p、d、f轨道
- 内置自旋(S)、轨道角动量(L)、自旋-轨道耦合(LS)等算符
- 提供投影算符功能

#### 主要方法

- `make_P_easy(select_name)`: 创建投影算符
- `make_POP_easy(select_name, pre_perpared_name)`: 创建投影算符与预定义算符的组合
- `make_POP_pre_prepared(P_mat, pre_perpared_name)`: 使用自定义投影算符

#### build_operator_LS

构建自旋和轨道角动量算符，支持LS耦合基或实原子轨道基。

### 6. utility.py

#### 费米-狄拉克分布相关函数

- `fermi_dirac(E, mu, kBT)`: 费米-狄拉克分布函数
- `fermi_dirac_vamp`: 向量化的费米分布函数
- `fermi_dirac_derivative(E, mu, kBT)`: 费米-狄拉克分布的导数
- `fermi_dirac_derivative_vamp`: 向量化的费米分布导数

#### 其他工具函数

- `cal_center_matrix(center_list)`: 计算Wannier中心差异矩阵
- `find_EF(file_name)`: 从量子化学计算输出文件中提取费米能级

### 7. TB_io.py

- `read_TB_model(file_name)`: 从npz或pickle文件读取紧束缚模型
- `save_TB_model(model, file_name)`: 保存模型到pickle文件

### 8. run.py

#### run_kps

批量处理k点计算的主函数：
- 自动检测并使用GPU或CPU
- 支持批处理以管理内存
- 支持多GPU并行计算
- `tabulate` 参数控制是否保存每个k点的结果

### 9. tb_parser.py

独立的 Wannier90 `*_tb.dat` 解析器，仅依赖 NumPy，不需要安装 WannierBerri。它读取实空间哈密顿量、位置矩阵元、晶格基矢、Wannier 中心和 R 向量，并保存为可由 `TB_io.read_TB_model` 直接读取的 NPZ 文件。

**推荐新的数据转换流程使用 `tb_parser.py`**，以避免为了解析一个 `*_tb.dat` 文件而引入完整的 WannierBerri 依赖。

默认行为：

- 解析 `HR` 和 `AA_R`；
- 将 `AA_R` 从 Wannier90 convention II 转换为当前 TB 模型使用的 convention I；
- 输出 `HR`、`AA_R`、`iRvec`、`cRvec`、`real_lattice`、`recip_lattice`、`Wan_center` 和 `num_wann`。

### 10. get_WB.py

WannierBerri 数据读取工具，从 `*_tb.dat` 文件读取数据并保存为 NPZ 格式。该脚本作为兼容和交叉验证方案保留，需要额外安装 WannierBerri；日常数据转换推荐使用 `tb_parser.py`。

## 使用指南

### 从 `*_tb.dat` 生成 NPZ（推荐）

使用纯 NumPy 解析器：

```powershell
python tb_parser.py Cu_tb.dat -o TB_Cu.npz
```

如果不指定 `-o`/`--output`，输出文件默认与输入文件同目录、同主文件名，例如 `Cu_tb.dat` 将生成 `Cu_tb.npz`。

可选参数：

- `--no-aa`：不保存 `AA_R`，仅适用于诊断或自定义流程。当前 `TB_io.read_TB_model` 需要 `AA_R`，因此常规使用不应添加该参数。
- `--no-conv`：不执行 `AA_R` convention II 到 convention I 的转换。当前 TB 模型数据流程应保持默认转换。

生成后可直接加载：

```python
from TB_io import read_TB_model

model = read_TB_model("TB_Cu.npz")
```

在 Cu 模型上，`tb_parser.py` 的结果已与 WannierBerri 生成的 `TB_Cu.npz` 交叉验证：`HR`、`AA_R`、`iRvec`、晶格和 Wannier 中心一致，`cRvec` 仅有约 `1.8e-15` 的浮点舍入误差。

### 基本使用流程

```python
from TB_io import read_TB_model
import operators as ops

# 1. 加载模型
model = read_TB_model("TB.npz")

# 2. 设置费米能级
model.EF = 12.43
model.set_E_fermis(-10, 8, 0.05)  # 设置费米能级扫描范围

# 3. 清除之前的计算器
model.clear_calculator()

# 4. 添加计算器
# 计算轨道霍尔电导率
matrix_all = ops.operator_matrix_all(['spd'])
Lz = matrix_all.pre_prepared_operators['Lz']
model.make_calc_kubo("OHC_z", alpha=0, beta=1, Gamma=0.025, Operator=Lz)

# 计算态密度
model.make_calc_inner("cumDOS", jnp.eye(model.nwan))

# 5. 创建统一计算器
model.make_calc_all(kbT=0.026, use_A=False)

# 6. 设置k网格
model.make_k_mesh(nkx=100, nky=100, nkz=100)

# 7. 运行计算
model.run_all("job_name", batch_size=1000, tabulate=False)
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
model.make_calc_kubo("OHC_z", alpha=0, beta=1, Gamma=0.025, Operator=Lz)

# 创建统一计算器
model.make_calc_all(kbT=0.003, use_A=False)

# 设置k网格
model.make_k_mesh(80, 80, 80)

# 运行计算
model.run_all("Cu", batch_size=100, tabulate=False)
```

### 高级功能

#### 1. 部分态密度(PDOS)计算

```python
# 创建s轨道的投影算符
P_s = matrix_all.make_P_easy("s")
model.make_calc_inner("s_PDOS", P_s)
```

#### 2. 高对称路径能带计算

```python
k_points = jnp.array([
    [0.0, 0.0, 0.0],    # Γ
    [0.5, 0.0, 0.0],    # X
    [0.5, 0.5, 0.0],    # M
    [0.0, 0.0, 0.0],    # Γ
])
ticks_label = ["Γ", "X", "M", "Γ"]

k_path, k_path_x, ticks, ticks_label = model.make_k_path(
    k_points, ticks_label, length=100
)
```

#### 3. 轨道磁化率计算

```python
# 使用现代理论计算轨道磁化矩阵
Lz_modern = make_calc_mag_matrix(model, gamma=2)
model.make_calc_kubo("modern", alpha=0, beta=1, Gamma=0.025, 
                     Operator=Lz_modern, H_gauge=True)
```

#### 4. 使用规范势项

```python
model.make_calc_all(kbT=0.026, use_A=True)
```

#### 5. 带内贡献计算

```python
model.make_calc_intra("intra_OHC", beta=2, Operator=Lz, H_gauge=False)
# 注意：需要使用 scanner_type='FD_deri'
model.make_calc_all(kbT=0.003, use_A=False, scanner_type="FD_deri")
```

## 技术特点

1. **高性能计算**: 利用 JAX 的 JIT 编译和自动向量化实现高效计算
2. **GPU支持**: 内置 GPU 并行计算能力，支持多GPU计算
3. **模块化设计**: 清晰的代码结构和模块化设计
4. **批处理计算**: 支持大批量k点并行计算，有效管理内存
5. **丰富的算符库**: 内置多种物理算符，支持自定义算符
6. **现代理论支持**: 支持现代轨道磁化理论和轨道贝里曲率计算
7. **规范势支持**: 支持在速度算符中包含规范势项

## 依赖项

- Python 3.11+
- JAX 2.0+
- NumPy
- tqdm (进度显示)
- WannierBerri（可选，仅用于 `get_WB.py` 兼容或交叉验证流程）

## 性能优化建议

1. 调整 `batch_size` 以平衡计算速度和内存使用
2. 使用GPU时，确保 `batch_size` 是GPU数量的倍数
3. 对于高分辨率k网格，考虑减小 `batch_size` 避免内存溢出
4. 对于只需要特定能量范围的计算，适当设置费米能级扫描范围
5. 使用JIT编译的函数可以显著提高计算速度

## 注意事项

1. 推荐使用 `tb_parser.py` 将 Wannier90 `*_tb.dat` 转换为 NPZ，并确保 NPZ 文件包含正确格式的紧束缚模型数据
2. 使用GPU时注意内存限制，必要时减小 `batch_size`
3. 对于低温计算，设置合适的 `kbT` 值以避免数值不稳定
4. 在使用Kubo公式计算时，调整 `Gamma` 参数以获得合适的展宽
5. 规范势项(AA_R)的使用会影响速度算符的计算结果
6. 带内贡献计算必须使用 `scanner_type='FD_deri'`

## 许可证

MIT许可证

## 联系方式

如有问题或建议，请联系项目维护者。
wenchenyue5678@buaa.edu.cn

## 参考文献

1. Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022) - 现代轨道磁化理论
2. WannierBerri - Wannier插值和贝里相计算工具包
