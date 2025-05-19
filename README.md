---
**`LithiumVision/README_CN.md`**
---
```markdown
# LithiumVision：基于MatterGen的锂离子超导体快速筛选与评估

## 项目概述

LithiumVision是一个利用微软研究院开发的先进生成式AI模型MatterGen，进行锂离子超导体快速筛选与评估的项目。该项目旨在通过人工智能技术加速新型锂离子超导体材料的发现，为下一代固态电池研发提供候选材料。

## 背景与意义

锂离子超导体作为下一代固态电解质的关键材料，因其高离子导电率和优异的稳定性，成为全球研究的焦点。传统的材料发现方法，如实验试错和第一性原理计算，虽然精确，但成本高昂且周期漫长。本项目利用生成式AI模型MatterGen，能够以前所未有的速度和效率探索广阔的化学空间，快速生成潜在的锂离子超导体候选材料。

## 功能特点

- **基于化学体系的条件生成**：利用MatterGen的预训练模型，针对特定锂离子导体化学体系生成候选晶体结构。
- **自动数据获取与分析**：自动调用Materials Project等开源数据库API获取材料性质数据。
- **新颖性与稳定性评估**：评估生成结构的新颖性，并通过数据库匹配或MatterSim（若集成）评估其稳定性。
- **结构可视化**：支持对候选材料的晶体结构进行可视化。
- **系统化筛选与评估**：提供完整的工作流程，从生成到筛选、评估和结果展示。

## 安装指南

### 前提条件

- Python 3.10或更高版本
- CUDA支持（推荐用于加速生成过程）
- Git LFS（用于下载大型模型文件）

### 设置环境

1.  克隆本仓库：
    ```bash
    git clone [https://github.com/yourusername/LithiumVision.git](https://github.com/yourusername/LithiumVision.git)
    cd LithiumVision
    ```

2.  创建并激活虚拟环境：
    ```bash
    # 使用conda（推荐）
    conda env create -f environment.yml
    conda activate lithiumvision

    # 或使用venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # .venv\Scripts\activate   # Windows
    pip install -r requirements.txt
    ```

3.  设置MatterGen：
    请遵循MatterGen官方GitHub仓库（microsoft/mattergen）的安装指南。通常包括：
    ```bash
    # 克隆MatterGen仓库 (如果setup_environment.py未处理)
    # git clone [https://github.com/microsoft/mattergen.git](https://github.com/microsoft/mattergen.git)
    # cd mattergen

    # 安装Git LFS并拉取模型文件
    # git lfs install
    # git lfs pull

    # 安装MatterGen
    # pip install -e .
    # cd ..
    ```
    建议运行 `python scripts/setup_environment.py` 脚本，它会提供指引。

4.  设置Materials Project API密钥：
    ```bash
    # 将你的MP API密钥设置为环境变量
    export MP_API_KEY="your_api_key_here"  # Linux/Mac
    # set MP_API_KEY="your_api_key_here"     # Windows
    ```
    或者，您也可以在 `configs/analysis_configs.json` 文件中设置。

5.  初始化项目结构与环境检查：
    ```bash
    python scripts/setup_environment.py
    ```

## 使用指南

### 运行完整工作流程

使用`run_workflow.py`脚本可以执行完整的工作流程，包括生成、分析、筛选和可视化步骤：

```bash
python scripts/run_workflow.py --output_report
可以通过各种参数控制工作流程的执行：

Bash

# 指定化学体系
python scripts/run_workflow.py --chemical_systems "Li-P-S" "Li-Si-S"

# 指定能量上凸包目标值
python scripts/run_workflow.py --e_hull_targets 0.0 0.05 0.1

# 跳过特定步骤
python scripts/run_workflow.py --skip_generation --skip_visualization
单独运行各个步骤
也可以单独运行工作流程中的各个步骤：

生成候选结构
Bash

# 使用scripts/generate_structures.py脚本生成特定化学体系的结构
python scripts/generate_structures.py --chem_sys "Li-P-S" --e_hull 0.05 --num_samples 100
分析生成的结构
Bash

# 分析生成的结构并匹配Materials Project数据
python scripts/analyze_structures.py --input_dir "data/generated/LiPS_ehull_0.05"
筛选候选材料
Bash

# 根据稳定性指标筛选候选材料
python scripts/filter_candidates.py --input_csv "data/analyzed/LiPS_ehull_0.05/analyzed_structures.csv" --e_hull_max 0.1
可视化结果
Bash

# 可视化筛选后的候选材料
python scripts/visualize_structures.py --candidates "results/candidates/top_candidates.csv" --top 10
生成报告
Bash

# 生成结果摘要报告
python scripts/summarize_results.py
配置文件
项目使用JSON格式的配置文件来控制各个步骤的参数：

configs/generation_configs.json：控制结构生成参数，包括化学体系、目标energy_above_hull值、MatterGen模型名称等。
configs/analysis_configs.json：控制分析和筛选参数，如稳定性阈值、API密钥、是否使用严格结构匹配等。
可以根据需要修改这些配置文件来调整工作流程。

项目结构
LithiumVision/
├── README.md
├── README_CN.md
├── configs/                  # 配置文件
│   ├── analysis_configs.json # 分析配置
│   └── generation_configs.json # 生成配置
├── data/                     # 数据目录 (脚本运行时自动创建)
│   ├── generated/            # 生成的结构 (例如: LiPS_ehull_0.05/generated_crystals_cif.zip)
│   └── analyzed/             # 分析结果 (例如: LiPS_ehull_0.05/analyzed_structures.csv)
├── environment.yml           # Conda环境配置
├── notebooks/                # Jupyter笔记本 (用于数据探索、自定义可视化和演示)
│   ├── exploration.ipynb
│   ├── presentation.ipynb
│   └── visualization.ipynb
├── requirements.txt          # Python依赖
├── results/                  # 结果输出 (脚本运行时部分自动创建)
│   ├── candidates/           # 候选材料CSV (例如: top_candidates.csv)
│   ├── figures/              # 图表和结构可视化图像
│   │   └── structures/       # 单个结构的图像
│   └── tables/               # 表格形式的摘要和日志 (例如: generation_campaign_log.csv)
└── scripts/                  # 主要脚本
    ├── analyze_structures.py # 分析生成的结构
    ├── filter_candidates.py  # 筛选候选材料
    ├── generate_structures.py # 使用MatterGen生成结构
    ├── run_workflow.py       # 完整工作流程脚本
    ├── setup_environment.py  # 环境设置脚本
    ├── summarize_results.py  # 汇总结果并生成报告
    └── visualize_structures.py # 结构可视化
示例结果
成功运行工作流程后，可以在以下位置找到结果：

生成的结构： data/generated/<化学体系_ehull条件>/generated_crystals_cif.zip
分析结果： data/analyzed/<化学体系_ehull条件>/analyzed_structures.csv
候选材料列表： results/candidates/top_candidates.csv
结构可视化： results/figures/structures/
摘要报告： results/tables/LithiumVision_summary_report.md
参考资料
MatterGen: https://github.com/microsoft/mattergen
Materials Project: https://materialsproject.org/
Pymatgen: https://pymatgen.org/
MP-API (Materials Project API): https://github.com/materialsproject/api
致谢
本项目使用了微软研究院开发的MatterGen生成模型，以及Materials Project提供的开放材料数据。我们对这些资源的贡献表示感谢。

许可
MIT License (如果使用MIT许可，请添加一个https://www.google.com/search?q=LICENSE文件)


---
**`LithiumVision/configs/analysis_configs.json`**
---
```json
{
  "default": {
    "mp_api_key": null,
    "max_workers": 4,
    "structure_matcher": false,
    "e_hull_threshold": 0.2,
    "novelty_criteria": {
      "formula_matching": true,
      "structure_matching_ltol": 0.2,
      "structure_matching_stol": 0.3,
      "structure_matching_angle_tol": 5
    }
  },
  "filtering": {
    "e_hull_max": 0.1,
    "only_novel": false,
    "only_containing_li": true,
    "top_n_candidates": 20,
    "score_weights": {
      "stability": 1.0,
      "novelty": 0.5
    }
  },
  "visualization": {
    "top_candidates_to_visualize": 10,
    "image_format": "png",
    "image_dpi": 300
  },
  "stability_scoring": {
    "ranges": [
      {"max_e_hull": 0.01, "score": 100},
      {"max_e_hull": 0.05, "score": 80},
      {"max_e_hull": 0.1, "score": 50},
      {"max_e_hull": 0.2, "score": 20}
    ],
    "novel_structure_default_score": 20
  },
  "batch_analysis": {
    "auto_analyze_after_generation": true,
    "auto_filter_after_analysis": true
  }
}