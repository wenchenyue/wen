# TB_model - 紧束缚模型计算工具包

## 简介

TB_model 是一个用于紧束缚模型计算的 Python 工具包，提供多种物理量计算功能，支持 GPU 加速。

## 主要功能

- 能带结构计算
- 态密度 (DOS) 和投影态密度 (PDOS)
- Kubo 公式输运计算
- 轨道磁化率计算
- 轨道贝里曲率 (OBC) 计算
- k 空间路径和网格构建

## 快速开始

### 安装依赖

```bash
pip install jax numpy tqdm
```

### 基本使用

```python
from TB_io import read_TB_model
from K_space import build_k_mesh

# 读取模型
model = read_TB_model("TB.npz")

# 设置费米能级
model.EF = 12.43

# 创建 k 网格
k_mesh = build_k_mesh(model, 100, 100, 100)

# 运行计算
results = model.run_all("job_name", batch_size=1000)
```

## 文档

详细文档请参阅 [documentation.md](documentation.md)

## 许可证

MIT License
