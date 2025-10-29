# 紧束缚模型计算工具包 (TB_model)

## 项目概述
这是一个用于紧束缚模型计算的工具包，提供了多种计算功能，包括哈密顿量计算、Kubo公式计算、态密度计算等。项目支持GPU加速，可以高效处理复杂的物理计算任务。

## 主要功能
- 构建k空间路径和网格
- 计算哈密顿量及其速度
- 实现Kubo公式计算
- 支持态密度(DOS)和投影态密度(PDOS)计算
- 提供高对称路径能带计算
- 支持GPU加速计算

## 项目结构
```
.
├── K_space.py          # k空间路径和网格构建
├── TB_io.py            # 模型数据读写
├── TB_model.py         # 紧束缚模型数据结构
├── calculators.py      # 计算器生成模块
├── documentation.md    # 项目文档
├── operators.py        # 算符构建模块
├── run.py              # 运行计算模块
├── utility.py          # 工具函数
```

## 核心模块说明

### TB_model.py
定义了`TB_model`数据类，用于存储紧束缚模型的基本信息，包括轨道信息、哈密顿量参数等。

### calculators.py
提供多种计算器生成函数：
- `make_k_hamiltonian_calculator`: 创建k空间哈密顿量计算器
- `make_k_hamiltonian_velocity_calculator`: 创建哈密顿量速度计算器
- `make_calc_inner`: 创建内积计算器（用于DOS计算）
- `make_calc_kubo`: 创建Kubo公式计算器

### K_space.py
包含k空间路径和网格构建函数：
- `build_k_path`: 根据给定高对称点构建k路径
- `build_k_mesh`: 构建三维k网格

### operators.py
提供算符构建功能：
- `operator_matrix_all`: 算符矩阵类
- `build_operator_all`: 构建完整算符矩阵
- `build_operator_LS`: 构建角动量算符

### utility.py
包含实用函数：
- `fermi_dirac`: 费米-狄拉克分布函数
- `cal_center_matrix`: 计算中心矩阵
- `find_EF`: 寻找费米能级

### TB_io.py
实现模型数据的读写功能：
- `read_TB_model`: 从文件读取模型数据
- `save_TB_model`: 保存模型数据到文件

### run.py
提供运行计算的接口函数：
- `run_kps`: 执行k空间点计算，支持批量处理和GPU加速

## 使用指南

### 基本使用流程
1. 读取模型：`model = read_TB_model("TB.npz")`
2. 创建计算器：`calculator = make_calc_kubo(model)`
3. 构建k路径：`kps = build_k_path(model, k_points, ticks_label)`
4. 运行计算：`results = run_kps(kps, calculator)`

### 添加内积算符(用于DOS计算)
```python
from utility import build_P_easy
from calculators import make_calc_inner

P = build_P_easy(model.orbital_list, "all")
calc_inner = make_calc_inner(model, P)
```

### 添加Kubo计算用算符
```python
from calculators import make_calc_kubo

calc_kubo = make_calc_kubo(model, alpha=0, beta=1, Gamma=0.025)
```

### 创建内积计算器
```python
calc_inner = make_calc_inner(model, kbT=0.01)
```

### 创建Kubo计算器
```python
calc_kubo = make_calc_kubo(model, alpha=0, beta=1, Gamma=0.025, kbT=0.01)
```

### 运行Kubo计算
```python
from K_space import build_k_mesh
from run import run_kps

k_mesh = build_k_mesh(model, 10, 10, 1)
results = run_kps(k_mesh, calc_kubo, batch_size=100)
```

### 或者使用模型的run_all方法运行所有计算
```python
results = model.run_all(kps, calculators)
```

## 高级功能

### 投影态密度(PDOS)计算
#### 创建s轨道的投影算符
```python
from utility import build_P_easy

P_s = build_P_easy(model.orbital_list, "s")
```

### 高对称路径能带计算
#### 定义高对称点 (以立方晶体为例)
```python
from K_space import build_k_path

k_points = jnp.array([
    [0.0, 0.0, 0.0],   # Γ点
    [0.5, 0.0, 0.0],   # X点
    [0.5, 0.5, 0.0],   # M点
    [0.0, 0.0, 0.0],   # Γ点
])
ticks_label = ["Γ", "X", "M", "Γ"]
```

#### 构建k路径
```python
k_path = build_k_path(model, k_points, ticks_label, length=100)
```

### GPU加速
在`run_kps`函数中会自动检测GPU，建议设置`batch_size`能被GPU数量整除以获得最佳性能。

## 技术特点
- 使用JAX进行自动微分和GPU加速
- 支持大规模并行计算
- 模块化设计，易于扩展
- 提供详细的文档和示例

## 依赖项
- Python 3.8+
- JAX
- NumPy
- dataclasses

## 性能优化建议
1. 尽量使用GPU进行计算
2. 合理设置batch_size以充分利用硬件资源
3. 对于重复计算，使用缓存机制
4. 在构建模型时，精简不必要的轨道信息

## 示例应用
请参考`documentation.md`中的示例代码。

## 注意事项
1. 确保输入文件格式正确
2. 在使用GPU加速时，注意内存限制
3. 对于复杂计算，适当调整精度设置

## 扩展与开发
项目结构设计为高度模块化，可以方便地添加新的计算功能和物理模型。建议遵循现有代码风格进行扩展开发。

## 许可证
本项目采用MIT许可证。

## 联系方式
请参考`documentation.md`中的联系方式。