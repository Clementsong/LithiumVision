# LithiumVision：基于MatterGen的锂离子超导体快速筛选与评估

## 项目概述

LithiumVision是一个利用微软研究院开发的先进生成式AI模型MatterGen，进行锂离子超导体快速筛选与评估的项目。该项目旨在通过人工智能技术加速新型锂离子超导体材料的发现，为下一代固态电池研发提供候选材料。

## 背景与意义

锂离子超导体作为下一代固态电解质的关键材料，因其高离子导电率和优异的稳定性，成为全球研究的焦点。传统的材料发现方法，如实验试错和第一性原理计算，虽然精确，但成本高昂且周期漫长。本项目利用生成式AI模型MatterGen，能够以前所未有的速度和效率探索广阔的化学空间，快速生成潜在的锂离子超导体候选材料。

## 功能特点

- **基于化学体系的条件生成**：利用MatterGen的预训练模型，针对特定锂离子导体化学体系生成候选晶体结构
- **自动数据获取与分析**：自动调用Materials Project等开源数据库API获取材料性质数据
- **新颖性与稳定性评估**：评估生成结构的新颖性，并通过数据库匹配或MatterSim评估其稳定性
- **结构可视化**：支持对候选材料的晶体结构进行可视化
- **系统化筛选与评估**：提供完整的工作流程，从生成到筛选、评估和结果展示

## 安装指南

### 前提条件

- Python 3.10或更高版本
- CUDA支持（推荐用于加速生成过程）
- Git LFS（用于下载大型模型文件）

### 设置环境

1. 克隆本仓库：
```bash
git clone https://github.com/yourusername/LithiumVision.git
cd LithiumVision
```

2. 创建并激活虚拟环境：
```bash
# 使用conda（推荐）
conda env create -f environment.yml
conda activate lithiumvision

# 或使用venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

3. 设置MatterGen：
```bash
# 克隆MatterGen仓库
git clone https://github.com/microsoft/mattergen.git
cd mattergen

# 安装Git LFS并拉取模型文件
git lfs install
git lfs pull

# 安装MatterGen
pip install -e .
```

4. 设置Materials Project API密钥：
```bash
# 将你的MP API密钥设置为环境变量
export MP_API_KEY="your_api_key_here"  # Linux/Mac
set MP_API_KEY="your_api_key_here"     # Windows
```

## 使用指南

### 生成候选结构

```bash
# 使用scripts/generate_structures.py脚本生成特定化学体系的结构
python scripts/generate_structures.py --chem_sys "Li-P-S" --e_hull 0.05 --num_samples 100
```

### 分析生成的结构

```bash
# 分析生成的结构并匹配Materials Project数据
python scripts/analyze_structures.py --input_dir "data/generated/Li-P-S_ehull_0.05"
```

### 筛选候选材料

```bash
# 根据稳定性指标筛选候选材料
python scripts/filter_candidates.py --e_hull_max 0.1 --output "results/candidates/top_candidates.csv"
```

### 可视化结果

```bash
# 可视化筛选后的候选材料
python scripts/visualize_structures.py --candidates "results/candidates/top_candidates.csv" --top 10
```

## 项目结构

```
LithiumVision/
├── scripts/                  # 主要脚本
│   ├── setup_environment.py  # 环境设置脚本
│   ├── generate_structures.py # 使用MatterGen生成结构
│   ├── analyze_structures.py # 分析生成的结构
│   ├── filter_candidates.py  # 筛选候选材料
│   ├── visualize_structures.py # 结构可视化
│   └── summarize_results.py  # 汇总结果
├── configs/                  # 配置文件
│   ├── generation_configs.json # 生成配置
│   └── analysis_configs.json  # 分析配置
├── data/                     # 数据目录
│   ├── generated/            # 生成的结构
│   └── analyzed/             # 分析结果
├── results/                  # 结果输出
│   ├── figures/              # 图表
│   ├── tables/               # 表格
│   └── candidates/           # 候选材料
└── notebooks/                # Jupyter笔记本
    ├── exploration.ipynb     # 数据探索
    ├── visualization.ipynb   # 可视化笔记本
    └── presentation.ipynb    # 演示笔记本
```

## 参考资料

1. MatterGen: https://github.com/microsoft/mattergen
2. Materials Project: https://materialsproject.org/
3. Pymatgen: https://pymatgen.org/
4. MP-API: https://github.com/materialsproject/api

## 致谢

本项目使用了微软研究院开发的MatterGen生成模型，以及Materials Project提供的开放材料数据。我们对这些资源的贡献表示感谢。

## 许可

[MIT License](LICENSE) 