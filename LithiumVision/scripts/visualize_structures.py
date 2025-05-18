#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化晶体结构

此脚本用于可视化筛选出的候选材料的晶体结构，
生成高质量的结构图像，用于报告和演示。
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize')

# 检查所需库
try:
    from pymatgen.core import Structure
    from pymatgen.vis.structure_vtk import StructureVis
    from pymatgen.vis.plotters import StructurePlotter
    from pymatgen.io.ase import AseAtomsAdaptor
    logger.info("依赖库加载成功")
except ImportError as e:
    logger.error(f"依赖库导入失败: {e}")
    logger.error("请确保已安装pymatgen和必要的可视化依赖")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='可视化晶体结构')
    
    parser.add_argument('--candidates', type=str, required=True,
                        help='候选材料CSV文件路径')
    
    parser.add_argument('--structures_dir', type=str, default=None,
                        help='包含CIF文件的目录，如果未指定则尝试从candidates文件中的路径推断')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认为results/figures/structures')
    
    parser.add_argument('--top', type=int, default=5,
                        help='要可视化的顶级候选材料数量，默认: 5')
    
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'jpg', 'svg', 'pdf'],
                        help='输出图像格式，默认: png')
    
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出图像DPI，默认: 300')
    
    return parser.parse_args()

def setup_paths(args):
    """设置输入和输出路径"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 候选材料文件路径
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        logger.error(f"候选材料文件 {candidates_path} 不存在")
        sys.exit(1)
    
    # 设置结构目录路径
    if args.structures_dir:
        structures_dir = Path(args.structures_dir)
    else:
        # 尝试从候选材料文件中推断
        try:
            # 读取CSV文件的第一行
            df = pd.read_csv(candidates_path, nrows=1)
            if 'cif_file' in df.columns and 'source_dir' in df.columns:
                # 假设source_dir字段包含了生成目录的名称
                source_dir = df['source_dir'].iloc[0]
                structures_dir = project_root / "data" / "generated" / source_dir / "extracted_cifs"
                
                if not structures_dir.exists():
                    # 尝试上一级目录
                    structures_dir = project_root / "data" / "generated" / source_dir
                    if not structures_dir.exists():
                        logger.warning(f"无法找到结构目录 {structures_dir}")
                        structures_dir = None
            else:
                logger.warning("候选材料文件中没有cif_file或source_dir列")
                structures_dir = None
        except Exception as e:
            logger.warning(f"从候选材料文件推断结构目录失败: {e}")
            structures_dir = None
        
        # 如果无法推断，使用默认目录
        if structures_dir is None:
            structures_dir = project_root / "data" / "generated"
            logger.warning(f"使用默认结构目录: {structures_dir}")
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results" / "figures" / "structures"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return candidates_path, structures_dir, output_dir

def load_candidates(candidates_path, top_n):
    """加载候选材料数据"""
    try:
        df = pd.read_csv(candidates_path)
        logger.info(f"从 {candidates_path} 加载了 {len(df)} 条候选材料记录")
        
        # 选择前N个候选材料
        top_candidates = df.head(top_n)
        logger.info(f"选择了前 {len(top_candidates)} 个候选材料进行可视化")
        
        return top_candidates
    except Exception as e:
        logger.error(f"加载候选材料文件失败: {e}")
        sys.exit(1)

def find_cif_file(cif_filename, structures_dir):
    """查找CIF文件"""
    if cif_filename is None:
        return None
    
    # 首先尝试直接路径
    direct_path = Path(cif_filename)
    if direct_path.exists():
        return direct_path
    
    # 如果不是绝对路径，在structures_dir中查找
    # 尝试精确匹配
    exact_match = structures_dir / cif_filename
    if exact_match.exists():
        return exact_match
    
    # 尝试递归查找匹配的文件名
    try:
        for cif_path in structures_dir.glob('**/*.cif'):
            if cif_path.name == cif_filename:
                return cif_path
    except Exception as e:
        logger.warning(f"递归查找CIF文件时出错: {e}")
    
    logger.warning(f"无法找到CIF文件: {cif_filename}")
    return None

def visualize_structure(structure, output_path, title=None, format='png', dpi=300):
    """可视化晶体结构并保存图像"""
    try:
        # 使用StructurePlotter创建结构图
        plotter = StructurePlotter()
        fig = plotter.get_plot(structure)
        
        # 添加标题
        if title:
            plt.title(title)
        
        # 保存图像
        plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return True
    except Exception as e:
        logger.error(f"可视化结构失败: {e}")
        logger.debug(traceback.format_exc())
        return False

def generate_structure_visualizations(candidates, structures_dir, output_dir, format, dpi):
    """为候选材料生成结构可视化"""
    results = []
    
    for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc="可视化结构"):
        try:
            # 获取CIF文件名
            cif_filename = row.get('cif_file')
            
            # 查找CIF文件
            cif_path = find_cif_file(cif_filename, structures_dir)
            if cif_path is None:
                logger.warning(f"跳过候选材料 {idx+1}，无法找到CIF文件")
                continue
            
            # 读取结构
            structure = Structure.from_file(cif_path)
            
            # 构建输出文件名
            formula = row.get('formula', f"structure_{idx+1}")
            output_filename = f"{formula}_{Path(cif_filename).stem}.{format}"
            output_path = output_dir / output_filename
            
            # 构建标题
            title = f"{formula}"
            if 'is_novel' in row and 'is_stable' in row:
                novel_str = "新颖" if row['is_novel'] else "已知"
                stable_str = "稳定" if row['is_stable'] else "亚稳定"
                title += f" ({novel_str}, {stable_str})"
            
            if 'e_above_hull' in row and pd.notna(row['e_above_hull']):
                title += f"\nE_hull = {row['e_above_hull']:.4f} eV/atom"
            
            # 可视化并保存
            success = visualize_structure(structure, output_path, title, format, dpi)
            
            # 记录结果
            result = {
                "index": idx,
                "formula": formula,
                "cif_file": str(cif_path),
                "output_file": str(output_path),
                "success": success
            }
            results.append(result)
            
            logger.info(f"已生成结构可视化: {output_path}")
            
        except Exception as e:
            logger.error(f"处理候选材料 {idx+1} 时出错: {e}")
            logger.debug(traceback.format_exc())
    
    return results

def create_summary_visualization(visualization_results, output_dir, format, dpi):
    """创建汇总可视化"""
    if not visualization_results:
        logger.warning("没有成功的可视化结果，跳过汇总可视化")
        return None
    
    try:
        # 过滤成功的结果
        successful_results = [r for r in visualization_results if r['success']]
        
        if len(successful_results) == 0:
            logger.warning("没有成功的可视化结果，跳过汇总可视化")
            return None
        
        # 每行显示3个结构
        cols = min(3, len(successful_results))
        rows = (len(successful_results) + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 填充图像
        for i, result in enumerate(successful_results):
            if i < len(axes):
                # 读取生成的图像
                img_path = result['output_file']
                img = plt.imread(img_path)
                
                # 显示图像
                axes[i].imshow(img)
                axes[i].set_title(result['formula'])
                axes[i].axis('off')
        
        # 隐藏空白子图
        for i in range(len(successful_results), len(axes)):
            axes[i].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存汇总图像
        summary_path = output_dir / f"summary_visualization.{format}"
        plt.savefig(summary_path, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"已生成汇总可视化: {summary_path}")
        return summary_path
    
    except Exception as e:
        logger.error(f"创建汇总可视化失败: {e}")
        logger.debug(traceback.format_exc())
        return None

def main():
    args = parse_arguments()
    
    # 设置路径
    candidates_path, structures_dir, output_dir = setup_paths(args)
    
    # 加载候选材料
    candidates = load_candidates(candidates_path, args.top)
    
    # 生成结构可视化
    results = generate_structure_visualizations(
        candidates, structures_dir, output_dir, args.format, args.dpi
    )
    
    # 创建汇总可视化
    summary_path = create_summary_visualization(results, output_dir, args.format, args.dpi)
    
    # 保存可视化结果信息
    results_info = {
        "candidates_file": str(candidates_path),
        "structures_dir": str(structures_dir),
        "output_dir": str(output_dir),
        "format": args.format,
        "dpi": args.dpi,
        "total_candidates": len(candidates),
        "successful_visualizations": sum(1 for r in results if r['success']),
        "failed_visualizations": sum(1 for r in results if not r['success']),
        "summary_visualization": str(summary_path) if summary_path else None,
        "visualizations": results
    }
    
    # 保存结果信息
    info_path = output_dir / "visualization_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"可视化完成，成功: {results_info['successful_visualizations']}, "
               f"失败: {results_info['failed_visualizations']}")
    
    if results_info['successful_visualizations'] > 0:
        return 0
    else:
        logger.error("所有可视化尝试都失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 